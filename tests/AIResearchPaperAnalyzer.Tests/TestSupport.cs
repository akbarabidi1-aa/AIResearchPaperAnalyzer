using System.Text;
using AIResearchPaperAnalyzer.Services;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Options;

namespace AIResearchPaperAnalyzer.Tests
{
    // A throwaway directory that is removed when the test finishes.
    public sealed class TempDir : IDisposable
    {
        public string Path { get; }

        public TempDir(string? name = null)
        {
            Path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(), "airpa-tests", Guid.NewGuid().ToString("N"), name ?? "root");
            Directory.CreateDirectory(Path);
        }

        public string WriteFile(string relativePath, string content)
        {
            var full = System.IO.Path.Combine(Path, relativePath);
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(full)!);
            File.WriteAllText(full, content, new UTF8Encoding(false));
            return full;
        }

        public string WriteFile(string relativePath, byte[] content)
        {
            var full = System.IO.Path.Combine(Path, relativePath);
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(full)!);
            File.WriteAllBytes(full, content);
            return full;
        }

        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    public sealed class FakeWebHostEnvironment : IWebHostEnvironment
    {
        public FakeWebHostEnvironment(string contentRoot, string environmentName = "Development")
        {
            ContentRootPath = contentRoot;
            WebRootPath     = contentRoot;
            EnvironmentName = environmentName;
        }

        public string        ApplicationName         { get; set; } = "AIResearchPaperAnalyzer";
        public string        EnvironmentName         { get; set; }
        public string        ContentRootPath         { get; set; }
        public string        WebRootPath             { get; set; }
        public IFileProvider ContentRootFileProvider { get; set; } = new NullFileProvider();
        public IFileProvider WebRootFileProvider     { get; set; } = new NullFileProvider();
    }

    public sealed class FakeSession : ISession
    {
        private readonly Dictionary<string, byte[]> _store = new();

        public bool   IsAvailable => true;
        public string Id          => "test-session";
        public IEnumerable<string> Keys => _store.Keys;

        public void Clear() => _store.Clear();
        public Task CommitAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task LoadAsync(CancellationToken cancellationToken = default)   => Task.CompletedTask;
        public void Remove(string key) => _store.Remove(key);
        public void Set(string key, byte[] value) => _store[key] = value;
        public bool TryGetValue(string key, out byte[] value) => _store.TryGetValue(key, out value!);
    }

    public sealed class FakeAnalysisService : IPaperAnalysisService
    {
        private readonly PaperAnalysisResult _result;

        public FakeAnalysisService(PaperAnalysisResult result) => _result = result;

        public int     Calls             { get; private set; }
        public string? LastPath          { get; private set; }
        public bool    FileExistedAtCall { get; private set; }

        public Task<PaperAnalysisResult> AnalyzeAsync(string pdfPath, CancellationToken cancellationToken = default)
        {
            Calls++;
            LastPath          = pdfPath;
            FileExistedAtCall = File.Exists(pdfPath);
            return Task.FromResult(_result);
        }
    }

    public static class TestData
    {
        // Smallest content the validator accepts: the PDF magic bytes plus filler.
        public static byte[] PdfBytes(int totalLength = 64)
        {
            var bytes = Enumerable.Repeat((byte)' ', totalLength).ToArray();
            "%PDF-1.4\n"u8.CopyTo(bytes);
            return bytes;
        }

        public static IFormFile FormFile(byte[] content, string fileName) =>
            new FormFile(new MemoryStream(content), 0, content.Length, "File", fileName);

        public static PdfUploadValidator Validator(long maxBytes = UploadOptions.DefaultMaxFileSizeBytes) =>
            new(Options.Create(new UploadOptions { MaxFileSizeBytes = maxBytes }));

        // The repository root: the folder that holds the web project and AIEngine/.
        public static string RepoRoot()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null && !File.Exists(System.IO.Path.Combine(dir.FullName, "AIResearchPaperAnalyzer.csproj")))
                dir = dir.Parent;
            return dir?.FullName ?? throw new InvalidOperationException("Repository root not found.");
        }

        public static string DigitalFixture() =>
            System.IO.Path.Combine(RepoRoot(), "tests", "fixtures", "digital", "synthetic_digital.pdf");
    }
}
