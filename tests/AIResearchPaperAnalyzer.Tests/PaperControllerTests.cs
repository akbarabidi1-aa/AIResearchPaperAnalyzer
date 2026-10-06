using AIResearchPaperAnalyzer.Controllers;
using AIResearchPaperAnalyzer.Models;
using AIResearchPaperAnalyzer.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace AIResearchPaperAnalyzer.Tests
{
    public class PaperControllerTests : IDisposable
    {
        private const int TestUserId = 7;

        private readonly TempDir      _root = new();
        private readonly AppDbContext _db;

        public PaperControllerTests()
        {
            _db = new AppDbContext(new DbContextOptionsBuilder<AppDbContext>()
                .UseInMemoryDatabase(Guid.NewGuid().ToString())
                .Options);
        }

        public void Dispose()
        {
            _db.Dispose();
            _root.Dispose();
        }

        private string UploadsDir => Path.Combine(_root.Path, "Uploads");

        private string[] StoredUploads =>
            Directory.Exists(UploadsDir) ? Directory.GetFiles(UploadsDir) : Array.Empty<string>();

        private PaperController Controller(
            IPaperAnalysisService analysis, string environment = "Development", long maxBytes = 1_048_576)
        {
            var upload = Options.Create(new UploadOptions { MaxFileSizeBytes = maxBytes });
            var http   = new DefaultHttpContext { Session = new FakeSession() };
            http.Session.SetInt32("UserId", TestUserId);

            return new PaperController(
                _db, analysis, new PdfUploadValidator(upload), upload,
                new FakeWebHostEnvironment(_root.Path, environment),
                NullLogger<PaperController>.Instance)
            {
                ControllerContext = new ControllerContext { HttpContext = http }
            };
        }

        private static PaperAnalysisResult Success() => new()
        {
            Data = new AiEngineResult
            {
                Status          = "success",
                Summary         = "A short summary.",
                Keywords        = new[] { "network", "depth" },
                ImportantPoints = new[] { "First key point.", "Second key point." },
                Flow            = new[] { "Abstract" }
            }
        };

        private static PaperAnalysisResult Failure(AnalysisFailure failure) => new()
        {
            Failure     = failure,
            UserMessage = "The analysis engine could not process this PDF.",
            Diagnostics = @"Traceback at C:\secret\engine.py"
        };

        private static UploadViewModel Upload(byte[] content, string name = "paper.pdf") =>
            new() { File = TestData.FormFile(content, name) };

        [Fact]
        public async Task Successful_analysis_saves_a_research_paper_and_shows_key_points()
        {
            var analysis = new FakeAnalysisService(Success());

            var result = await Controller(analysis).Upload(Upload(TestData.PdfBytes()));

            var view  = Assert.IsType<ViewResult>(result);
            var model = Assert.IsType<ResultViewModel>(view.Model);
            Assert.Equal("Result", view.ViewName);
            Assert.Equal(new[] { "First key point.", "Second key point." }, model.ImportantPoints);
            Assert.Null(model.ErrorMessage);

            var paper = Assert.Single(_db.ResearchPapers);
            Assert.Equal("paper.pdf", paper.Title);
            Assert.Equal("A short summary.", paper.Summary);
            Assert.Equal("network, depth", paper.Keywords);
            Assert.Equal(TestUserId, paper.UserId);

            Assert.True(analysis.FileExistedAtCall);
            Assert.Single(StoredUploads);       // kept, as in V1
        }

        [Theory]
        [InlineData(AnalysisFailure.EngineError)]
        [InlineData(AnalysisFailure.Timeout)]
        [InlineData(AnalysisFailure.ProcessFailed)]
        [InlineData(AnalysisFailure.InvalidJson)]
        [InlineData(AnalysisFailure.PythonNotFound)]
        public async Task Failed_analysis_saves_nothing_and_removes_the_upload(AnalysisFailure failure)
        {
            var analysis = new FakeAnalysisService(Failure(failure));
            var controller = Controller(analysis);

            var result = await controller.Upload(Upload(TestData.PdfBytes()));

            var view = Assert.IsType<ViewResult>(result);
            Assert.Equal("Index", view.ViewName);
            Assert.False(controller.ModelState.IsValid);
            Assert.Empty(_db.ResearchPapers);
            Assert.Equal(1, analysis.Calls);
            Assert.True(analysis.FileExistedAtCall);
            Assert.Empty(StoredUploads);
        }

        [Fact]
        public async Task Diagnostics_are_shown_in_development_only()
        {
            var dev = Controller(new FakeAnalysisService(Failure(AnalysisFailure.ProcessFailed)), "Development");
            await dev.Upload(Upload(TestData.PdfBytes()));
            Assert.Contains("secret", (string)dev.ViewData["AnalysisDiagnostics"]!);

            var prod = Controller(new FakeAnalysisService(Failure(AnalysisFailure.ProcessFailed)), "Production");
            await prod.Upload(Upload(TestData.PdfBytes()));
            Assert.False(prod.ViewData.ContainsKey("AnalysisDiagnostics"));
            Assert.DoesNotContain(prod.ModelState.Values.SelectMany(v => v.Errors),
                e => e.ErrorMessage.Contains("secret"));
        }

        [Fact]
        public async Task Invalid_pdf_is_rejected_before_python_runs()
        {
            var analysis   = new FakeAnalysisService(Success());
            var controller = Controller(analysis);

            var result = await controller.Upload(Upload("not a pdf at all"u8.ToArray()));

            Assert.Equal("Index", Assert.IsType<ViewResult>(result).ViewName);
            Assert.False(controller.ModelState.IsValid);
            Assert.Equal(0, analysis.Calls);
            Assert.Empty(StoredUploads);
            Assert.Empty(_db.ResearchPapers);
        }

        [Fact]
        public async Task Empty_and_oversized_files_are_rejected_before_python_runs()
        {
            var analysis = new FakeAnalysisService(Success());

            await Controller(analysis).Upload(Upload(Array.Empty<byte>()));
            await Controller(analysis, maxBytes: 1024).Upload(Upload(TestData.PdfBytes(4096)));
            await Controller(analysis).Upload(new UploadViewModel());

            Assert.Equal(0, analysis.Calls);
            Assert.Empty(StoredUploads);
            Assert.Empty(_db.ResearchPapers);
        }

        [Fact]
        public async Task Client_file_name_never_decides_where_the_upload_is_stored()
        {
            var analysis = new FakeAnalysisService(Success());

            await Controller(analysis).Upload(Upload(TestData.PdfBytes(), @"..\..\..\evil\..\paper.pdf"));

            var stored = Assert.Single(StoredUploads);
            Assert.Equal(Path.GetFullPath(UploadsDir), Path.GetDirectoryName(Path.GetFullPath(stored)));
            Assert.True(Guid.TryParse(Path.GetFileNameWithoutExtension(stored), out _));
            Assert.Equal("paper.pdf", Assert.Single(_db.ResearchPapers).Title);
        }

        [Fact]
        public async Task Two_uploads_of_the_same_file_get_different_stored_names()
        {
            var analysis = new FakeAnalysisService(Success());

            await Controller(analysis).Upload(Upload(TestData.PdfBytes()));
            await Controller(analysis).Upload(Upload(TestData.PdfBytes()));

            Assert.Equal(2, StoredUploads.Distinct().Count());
            Assert.Equal(2, _db.ResearchPapers.Count());
        }

        [Fact]
        public async Task Very_long_file_name_is_shortened_to_fit_the_title_column()
        {
            var name = new string('a', 400) + ".pdf";

            await Controller(new FakeAnalysisService(Success())).Upload(Upload(TestData.PdfBytes(), name));

            Assert.Equal(260, Assert.Single(_db.ResearchPapers).Title!.Length);
        }

        // End to end through the real service and the real AIEngine/ai_engine.py.
        // Needs Python with AIEngine/requirements.txt installed (see docs/V1_BASELINE.md section 8).
        [Fact]
        [Trait("Category", "Integration")]
        public async Task Real_engine_analyses_the_digital_fixture_and_returns_key_points()
        {
            var service = new PaperAnalysisService(
                Options.Create(new AiEngineOptions()),
                new FakeWebHostEnvironment(TestData.RepoRoot()),
                NullLogger<PaperAnalysisService>.Instance);

            // Content root for uploads is the temp dir; the engine itself runs from the repository.
            var controller = Controller(service);

            var result = await controller.Upload(
                Upload(File.ReadAllBytes(TestData.DigitalFixture()), "synthetic_digital.pdf"));

            var view = Assert.IsType<ViewResult>(result);
            Assert.True(view.ViewName == "Result",
                "analysis failed: " + controller.ViewData["AnalysisDiagnostics"]);
            var model = Assert.IsType<ResultViewModel>(view.Model);
            Assert.NotEmpty(model.ImportantPoints);
            Assert.NotEmpty(model.Keywords);
            Assert.False(string.IsNullOrWhiteSpace(model.Summary));
            Assert.Single(_db.ResearchPapers);
        }
    }
}
