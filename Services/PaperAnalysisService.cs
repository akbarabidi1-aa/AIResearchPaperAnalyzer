using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using AIResearchPaperAnalyzer.Models;
using Microsoft.Extensions.Options;

namespace AIResearchPaperAnalyzer.Services
{
    public enum AnalysisFailure
    {
        None,
        PdfNotFound,
        PythonNotFound,
        ScriptNotFound,
        Timeout,
        ProcessFailed,
        NoOutput,
        InvalidJson,
        EngineError
    }

    public sealed class PaperAnalysisResult
    {
        public bool            Succeeded   => Failure == AnalysisFailure.None;
        public AnalysisFailure Failure     { get; init; }

        // Set only when Succeeded.
        public AiEngineResult? Data        { get; init; }

        // Safe to show to any user: no machine paths, no stack traces.
        public string?         UserMessage { get; init; }

        // Development diagnostics only (paths, stderr, raw output). Never show to normal users.
        public string?         Diagnostics { get; init; }

        public int?            ExitCode    { get; init; }
        public string          StandardError { get; init; } = string.Empty;
    }

    public interface IPaperAnalysisService
    {
        Task<PaperAnalysisResult> AnalyzeAsync(string pdfPath, CancellationToken cancellationToken = default);
    }

    // Owns the ASP.NET -> Python handoff: runs AIEngine/ai_engine.py on a PDF and returns a typed result.
    public class PaperAnalysisService : IPaperAnalysisService
    {
        private const int MaxDiagnosticChars = 4000;

        // The one engine error that is meaningful (and safe) to show to users as-is.
        private const string NoTextPrefix = "No text found in PDF";

        private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

        private readonly AiEngineOptions               _options;
        private readonly IWebHostEnvironment           _env;
        private readonly ILogger<PaperAnalysisService> _logger;

        public PaperAnalysisService(
            IOptions<AiEngineOptions>     options,
            IWebHostEnvironment           env,
            ILogger<PaperAnalysisService> logger)
        {
            _options = options.Value;
            _env     = env;
            _logger  = logger;
        }

        public async Task<PaperAnalysisResult> AnalyzeAsync(string pdfPath, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(pdfPath) || !File.Exists(pdfPath))
                return Fail(AnalysisFailure.PdfNotFound,
                    "The uploaded file could not be found. Please upload it again.",
                    $"PDF not found: {pdfPath}");

            var script = ResolveFromContentRoot(_options.ScriptPath);
            if (!File.Exists(script))
                return Fail(AnalysisFailure.ScriptNotFound,
                    "The analysis engine is not installed correctly. Please contact the administrator.",
                    $"AI engine script not found: {script} (AIEngine:ScriptPath = \"{_options.ScriptPath}\")");

            var python = ResolvePythonExecutable(_options.PythonExecutable);
            if (IsPathLike(python) && !File.Exists(python))
                return PythonNotFound(python, null);

            var timeout = TimeSpan.FromSeconds(
                _options.TimeoutSeconds > 0 ? _options.TimeoutSeconds : AiEngineOptions.DefaultTimeoutSeconds);

            var psi = new ProcessStartInfo
            {
                FileName               = python,
                WorkingDirectory       = _env.ContentRootPath,
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding  = Encoding.UTF8,
                UseShellExecute        = false,
                CreateNoWindow         = true
            };
            // ArgumentList quotes each argument itself, so paths with spaces or quotes are passed intact.
            psi.ArgumentList.Add(script);
            psi.ArgumentList.Add(Path.GetFullPath(pdfPath));
            psi.Environment["PYTHONIOENCODING"] = "utf-8";

            using var proc = new Process { StartInfo = psi };
            try
            {
                proc.Start();
            }
            catch (Exception ex) when (ex is Win32Exception or FileNotFoundException or DirectoryNotFoundException)
            {
                return PythonNotFound(python, ex);
            }

            // Read both pipes while the process runs; otherwise a full stderr buffer would block Python.
            var stdoutTask = proc.StandardOutput.ReadToEndAsync();
            var stderrTask = proc.StandardError.ReadToEndAsync();

            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(timeout);

            try
            {
                await proc.WaitForExitAsync(timeoutCts.Token);
            }
            catch (OperationCanceledException)
            {
                KillProcessTree(proc);
                var partialError = await ReadOrEmptyAsync(stderrTask);
                await ReadOrEmptyAsync(stdoutTask);

                // The caller gave up (e.g. the browser disconnected): not a timeout.
                cancellationToken.ThrowIfCancellationRequested();

                return Fail(AnalysisFailure.Timeout,
                    $"The analysis took longer than {timeout.TotalSeconds:0} seconds and was stopped. " +
                    "Please try again or use a smaller PDF.",
                    $"Python process timed out after {timeout.TotalSeconds:0}s and was terminated.",
                    stderr: partialError);
            }

            var stdout   = await stdoutTask;
            var stderr   = await stderrTask;
            var exitCode = proc.ExitCode;

            if (exitCode != 0)
                return Fail(AnalysisFailure.ProcessFailed,
                    "The analysis engine failed while processing this PDF.",
                    $"Python exited with code {exitCode}.\nstdout: {Truncate(stdout)}",
                    exitCode, stderr);

            if (string.IsNullOrWhiteSpace(stdout))
                return Fail(AnalysisFailure.NoOutput,
                    "The analysis engine returned no result for this PDF.",
                    "Python exited with code 0 but wrote nothing to stdout.",
                    exitCode, stderr);

            var data = ParseEngineOutput(stdout, out var jsonError);
            if (data == null)
                return Fail(AnalysisFailure.InvalidJson,
                    "The analysis engine returned an unreadable result.",
                    $"Invalid JSON on stdout: {jsonError}\nstdout: {Truncate(stdout)}",
                    exitCode, stderr);

            if (!string.Equals(data.Status, "success", StringComparison.OrdinalIgnoreCase))
            {
                var engineMessage = data.Message ?? string.Empty;
                var userMessage   = engineMessage.StartsWith(NoTextPrefix, StringComparison.OrdinalIgnoreCase)
                    ? "No selectable text was found in this PDF. Scanned or image-only PDFs are not supported yet."
                    : "The analysis engine could not process this PDF.";

                return Fail(AnalysisFailure.EngineError, userMessage,
                    $"AI engine reported status \"{data.Status}\": {engineMessage}",
                    exitCode, stderr);
            }

            if (!string.IsNullOrWhiteSpace(stderr))
                _logger.LogDebug("AI engine succeeded with stderr output: {Stderr}", Truncate(stderr));

            return new PaperAnalysisResult { Data = data, ExitCode = exitCode, StandardError = stderr };
        }

        // ── Path resolution ───────────────────────────────────────────────────
        private string ResolveFromContentRoot(string path) =>
            Path.GetFullPath(Path.IsPathRooted(path) ? path : Path.Combine(_env.ContentRootPath, path));

        // "python" is looked up on PATH by the OS; anything with a directory part is treated as a path.
        private string ResolvePythonExecutable(string configured)
        {
            var value = string.IsNullOrWhiteSpace(configured) ? "python" : configured.Trim();
            return IsPathLike(value) ? ResolveFromContentRoot(value) : value;
        }

        private static bool IsPathLike(string value) =>
            Path.IsPathRooted(value) ||
            value.Contains(Path.DirectorySeparatorChar) ||
            value.Contains(Path.AltDirectorySeparatorChar);

        // ── Output parsing ────────────────────────────────────────────────────
        // The CLI prints one JSON object. If a library printed extra lines first, fall back to the last JSON line.
        private static AiEngineResult? ParseEngineOutput(string stdout, out string? error)
        {
            error = null;
            var text = stdout.Trim();

            if (TryDeserialize(text, out var result, out error))
                return result;

            var lastLine = text.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                               .LastOrDefault();
            if (lastLine != null && lastLine != text && lastLine.StartsWith('{') &&
                TryDeserialize(lastLine, out result, out _))
            {
                error = null;
                return result;
            }

            return null;
        }

        private static bool TryDeserialize(string json, out AiEngineResult? result, out string? error)
        {
            try
            {
                result = JsonSerializer.Deserialize<AiEngineResult>(json, JsonOptions);
                error  = result == null ? "JSON value was null." : null;
                return result != null;
            }
            catch (JsonException ex)
            {
                result = null;
                error  = ex.Message;
                return false;
            }
        }

        // ── Helpers ───────────────────────────────────────────────────────────
        private void KillProcessTree(Process proc)
        {
            try
            {
                if (!proc.HasExited)
                    proc.Kill(entireProcessTree: true);
                proc.WaitForExit(5000);
            }
            catch (Exception ex) when (ex is InvalidOperationException or Win32Exception or NotSupportedException)
            {
                _logger.LogWarning(ex, "Could not terminate the AI engine process cleanly.");
            }
        }

        // After a kill the pipes close; do not let a stuck read keep the request alive.
        private static async Task<string> ReadOrEmptyAsync(Task<string> readTask)
        {
            try
            {
                return await readTask.WaitAsync(TimeSpan.FromSeconds(5));
            }
            catch (Exception ex) when (ex is TimeoutException or IOException or ObjectDisposedException)
            {
                return string.Empty;
            }
        }

        private PaperAnalysisResult PythonNotFound(string python, Exception? ex) =>
            Fail(AnalysisFailure.PythonNotFound,
                "The analysis engine is not available right now. Please contact the administrator.",
                $"Python executable could not be started: \"{python}\" " +
                $"(AIEngine:PythonExecutable = \"{_options.PythonExecutable}\"). {ex?.Message}".TrimEnd());

        private PaperAnalysisResult Fail(
            AnalysisFailure failure, string userMessage, string diagnostics, int? exitCode = null, string stderr = "")
        {
            if (!string.IsNullOrWhiteSpace(stderr))
                diagnostics += $"\nstderr: {Truncate(stderr)}";

            _logger.LogWarning("Paper analysis failed ({Failure}). {Diagnostics}", failure, diagnostics);

            return new PaperAnalysisResult
            {
                Failure       = failure,
                UserMessage   = userMessage,
                Diagnostics   = diagnostics,
                ExitCode      = exitCode,
                StandardError = stderr
            };
        }

        private static string Truncate(string value)
        {
            value = value.Trim();
            return value.Length <= MaxDiagnosticChars ? value : value[..MaxDiagnosticChars] + " …[truncated]";
        }
    }
}
