using System.Diagnostics;
using AIResearchPaperAnalyzer.Services;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace AIResearchPaperAnalyzer.Tests
{
    // Runs the real service against small stand-in Python scripts (standard library only),
    // so every process outcome can be produced on demand. Requires "python" on PATH.
    public class PaperAnalysisServiceTests : IDisposable
    {
        private const string SuccessJson =
            "{\"status\": \"success\", \"file_name\": \"x.pdf\", \"summary\": \"S\", \"keywords\": [\"k\"], " +
            "\"important_points\": [\"P1 is a key point.\", \"P2 is a key point.\"], \"flow\": [\"Intro\"], \"tables\": []}";

        private readonly TempDir _root = new("content root");   // space on purpose
        private readonly string  _pdf;

        public PaperAnalysisServiceTests()
        {
            _pdf = _root.WriteFile("Uploads/sample file.pdf", TestData.PdfBytes());
        }

        public void Dispose() => _root.Dispose();

        private PaperAnalysisService Service(string script = "engine.py", string python = "python", int timeout = 60) =>
            new(Options.Create(new AiEngineOptions
                {
                    PythonExecutable = python,
                    ScriptPath       = script,
                    TimeoutSeconds   = timeout
                }),
                new FakeWebHostEnvironment(_root.Path),
                NullLogger<PaperAnalysisService>.Instance);

        private void Script(string body, string name = "engine.py") => _root.WriteFile(name, body);

        private static string PrintJson(string json) =>
            $"import sys\nsys.stdout.write({ToPythonLiteral(json)})\n";

        private static string ToPythonLiteral(string value) =>
            "'" + value.Replace("\\", "\\\\").Replace("'", "\\'").Replace("\n", "\\n") + "'";

        [Fact]
        public async Task Successful_run_returns_typed_data_with_key_points()
        {
            Script(PrintJson(SuccessJson));

            var result = await Service().AnalyzeAsync(_pdf);

            Assert.True(result.Succeeded, result.Diagnostics);
            Assert.Equal(0, result.ExitCode);
            Assert.Equal("S", result.Data!.Summary);
            Assert.Equal(new[] { "P1 is a key point.", "P2 is a key point." }, result.Data.ImportantPoints);
            Assert.Equal("x.pdf", result.Data.FileName);
        }

        [Fact]
        public async Task Pdf_path_with_spaces_reaches_python_as_one_argument()
        {
            Script("import sys, json\n" +
                   "print(json.dumps({'status': 'success', 'summary': sys.argv[1], 'keywords': [str(len(sys.argv))]}))\n");

            var result = await Service().AnalyzeAsync(_pdf);

            Assert.True(result.Succeeded, result.Diagnostics);
            Assert.Equal(Path.GetFullPath(_pdf), result.Data!.Summary);
            Assert.Equal(new[] { "2" }, result.Data.Keywords);
        }

        [Fact]
        public async Task Missing_python_executable_on_path_is_reported()
        {
            Script(PrintJson(SuccessJson));

            var result = await Service(python: "python-that-does-not-exist-airpa").AnalyzeAsync(_pdf);

            Assert.False(result.Succeeded);
            Assert.Equal(AnalysisFailure.PythonNotFound, result.Failure);
            Assert.Null(result.Data);
        }

        [Fact]
        public async Task Missing_python_executable_path_is_reported()
        {
            Script(PrintJson(SuccessJson));
            var missing = Path.Combine(_root.Path, "no-such-dir", "python.exe");

            var result = await Service(python: missing).AnalyzeAsync(_pdf);

            Assert.Equal(AnalysisFailure.PythonNotFound, result.Failure);
            Assert.DoesNotContain(_root.Path, result.UserMessage);
            Assert.Contains(missing, result.Diagnostics);
        }

        [Fact]
        public async Task Missing_engine_script_is_reported()
        {
            var result = await Service(script: "AIEngine/ai_engine.py").AnalyzeAsync(_pdf);

            Assert.Equal(AnalysisFailure.ScriptNotFound, result.Failure);
            Assert.DoesNotContain(_root.Path, result.UserMessage);
            Assert.Contains("ai_engine.py", result.Diagnostics);
        }

        [Fact]
        public async Task Missing_pdf_is_reported_without_starting_python()
        {
            Script("open('ran.txt', 'w').close()\n");

            var result = await Service().AnalyzeAsync(Path.Combine(_root.Path, "Uploads", "gone.pdf"));

            Assert.Equal(AnalysisFailure.PdfNotFound, result.Failure);
            Assert.False(File.Exists(Path.Combine(_root.Path, "ran.txt")));
        }

        [Fact]
        public async Task Timeout_stops_the_python_process_and_reports_timeout()
        {
            // The script records its PID next to the PDF, then hangs.
            Script("import sys, os, time\n" +
                   "open(sys.argv[1] + '.pid', 'w').write(str(os.getpid()))\n" +
                   "time.sleep(120)\n" +
                   "print('{\"status\": \"success\"}')\n");

            var watch  = Stopwatch.StartNew();
            var result = await Service(timeout: 3).AnalyzeAsync(_pdf);
            watch.Stop();

            Assert.Equal(AnalysisFailure.Timeout, result.Failure);
            Assert.Null(result.Data);
            Assert.Contains("3 seconds", result.UserMessage);
            Assert.True(watch.Elapsed < TimeSpan.FromSeconds(30), $"took {watch.Elapsed}");

            var pid = int.Parse(File.ReadAllText(_pdf + ".pid"));
            Assert.True(HasExited(pid), "the Python child process is still running");
        }

        [Fact]
        public async Task Caller_cancellation_stops_python_and_is_not_reported_as_timeout()
        {
            Script("import time\ntime.sleep(120)\n");
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));

            await Assert.ThrowsAnyAsync<OperationCanceledException>(
                () => Service(timeout: 60).AnalyzeAsync(_pdf, cts.Token));
        }

        [Fact]
        public async Task Invalid_json_on_stdout_is_reported()
        {
            Script("print('this is not json')\n");

            var result = await Service().AnalyzeAsync(_pdf);

            Assert.Equal(AnalysisFailure.InvalidJson, result.Failure);
            Assert.Null(result.Data);
            Assert.DoesNotContain("this is not json", result.UserMessage);
            Assert.Contains("this is not json", result.Diagnostics);
        }

        [Fact]
        public async Task Empty_stdout_is_reported()
        {
            Script("pass\n");

            var result = await Service().AnalyzeAsync(_pdf);

            Assert.Equal(AnalysisFailure.NoOutput, result.Failure);
        }

        [Fact]
        public async Task Failed_process_keeps_stderr_for_diagnostics_only()
        {
            Script("import sys\n" +
                   "sys.stderr.write('Traceback: boom at C:\\\\secret\\\\path\\\\engine.py')\n" +
                   "sys.exit(3)\n");

            var result = await Service().AnalyzeAsync(_pdf);

            Assert.Equal(AnalysisFailure.ProcessFailed, result.Failure);
            Assert.Equal(3, result.ExitCode);
            Assert.Contains("boom", result.StandardError);
            Assert.Contains("boom", result.Diagnostics);
            Assert.DoesNotContain("boom", result.UserMessage);
            Assert.DoesNotContain("secret", result.UserMessage);
        }

        [Fact]
        public async Task Non_zero_exit_fails_even_when_stdout_holds_success_json()
        {
            Script(PrintJson(SuccessJson) + "sys.exit(1)\n");

            var result = await Service().AnalyzeAsync(_pdf);

            Assert.Equal(AnalysisFailure.ProcessFailed, result.Failure);
            Assert.Null(result.Data);
        }

        [Fact]
        public async Task Engine_error_status_fails_and_hides_the_raw_message()
        {
            Script(PrintJson("{\"status\": \"error\", \"message\": \"Resource punkt_tab not found in C:/Users/someone/nltk_data\"}"));

            var result = await Service().AnalyzeAsync(_pdf);

            Assert.Equal(AnalysisFailure.EngineError, result.Failure);
            Assert.Null(result.Data);
            Assert.DoesNotContain("someone", result.UserMessage);
            Assert.Contains("punkt_tab", result.Diagnostics);
        }

        [Fact]
        public async Task Engine_no_text_error_gets_a_readable_message()
        {
            Script(PrintJson("{\"status\": \"error\", \"message\": \"No text found in PDF. Make sure it is not a scanned image.\"}"));

            var result = await Service().AnalyzeAsync(_pdf);

            Assert.Equal(AnalysisFailure.EngineError, result.Failure);
            Assert.Contains("No selectable text", result.UserMessage);
        }

        [Fact]
        public async Task Json_without_status_is_not_success()
        {
            Script(PrintJson("{\"summary\": \"S\"}"));

            var result = await Service().AnalyzeAsync(_pdf);

            Assert.False(result.Succeeded);
        }

        [Fact]
        public async Task Extra_lines_before_the_json_line_are_tolerated()
        {
            Script("print('[nltk_data] Downloading package punkt...')\n" + PrintJson(SuccessJson));

            var result = await Service().AnalyzeAsync(_pdf);

            Assert.True(result.Succeeded, result.Diagnostics);
            Assert.Equal(2, result.Data!.ImportantPoints!.Length);
        }

        [Fact]
        public async Task Large_stderr_does_not_block_the_process()
        {
            Script("import sys\n" +
                   "sys.stderr.write('w' * 500000)\n" +
                   PrintJson(SuccessJson).Replace("import sys\n", ""));

            var result = await Service(timeout: 30).AnalyzeAsync(_pdf);

            Assert.True(result.Succeeded, result.Diagnostics);
            Assert.Equal(500000, result.StandardError.Length);
        }

        private static bool HasExited(int pid)
        {
            try
            {
                using var p = Process.GetProcessById(pid);
                return p.WaitForExit(5000);
            }
            catch (ArgumentException)
            {
                return true;    // no such process
            }
        }
    }
}
