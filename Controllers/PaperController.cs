using AIResearchPaperAnalyzer.Models;
using AIResearchPaperAnalyzer.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;

namespace AIResearchPaperAnalyzer.Controllers
{
    public class PaperController : Controller
    {
        private const int MaxTitleLength = 260;     // ResearchPaper.Title column size

        private readonly AppDbContext             _db;
        private readonly IPaperAnalysisService    _analysis;
        private readonly PdfUploadValidator       _validator;
        private readonly UploadOptions            _upload;
        private readonly IWebHostEnvironment      _env;
        private readonly ILogger<PaperController> _logger;

        public PaperController(
            AppDbContext             db,
            IPaperAnalysisService    analysis,
            PdfUploadValidator       validator,
            IOptions<UploadOptions>  upload,
            IWebHostEnvironment      env,
            ILogger<PaperController> logger)
        {
            _db        = db;
            _analysis  = analysis;
            _validator = validator;
            _upload    = upload.Value;
            _env       = env;
            _logger    = logger;
        }

        // ── Auth guard ────────────────────────────────────────────────────────
        private int? UserId => HttpContext.Session.GetInt32("UserId");

        private IActionResult RedirectToLogin()
        {
            TempData["Error"] = "Please log in to continue.";
            return RedirectToAction("Login", "Auth");
        }

        // ── GET /Paper/Index ──────────────────────────────────────────────────
        [HttpGet]
        public IActionResult Index()
        {
            if (UserId == null) return RedirectToLogin();
            return View(new UploadViewModel());
        }

        // ── POST /Paper/Upload ────────────────────────────────────────────────
        // The request size limit is configured in Program.cs from Upload:MaxFileSizeBytes.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Upload(UploadViewModel model)
        {
            if (UserId == null) return RedirectToLogin();

            // 1. Validate before anything touches the disk or Python
            var validation = _validator.Validate(model.File);
            if (!validation.IsValid)
            {
                ModelState.AddModelError("File", validation.Message!);
                return View("Index", model);
            }

            var file = model.File!;

            // 2. Save PDF to disk under a server-generated name (never the client file name)
            var savedPath = await SaveUploadAsync(file, HttpContext.RequestAborted);
            var keepFile  = false;

            try
            {
                // 3. Run Python AI engine
                var analysis = await _analysis.AnalyzeAsync(savedPath, HttpContext.RequestAborted);

                if (!analysis.Succeeded || analysis.Data == null)
                {
                    ModelState.AddModelError("File", analysis.UserMessage ?? "The analysis failed. Please try again.");
                    if (_env.IsDevelopment())
                        ViewData["AnalysisDiagnostics"] = $"{analysis.Failure}: {analysis.Diagnostics}";
                    return View("Index", model);
                }

                var ai = analysis.Data;

                // 4. Persist result to database
                var paper = new ResearchPaper
                {
                    Title      = SafeTitle(file.FileName),
                    Summary    = ai.Summary  ?? "Analysis pending.",
                    Keywords   = ai.Keywords != null ? string.Join(", ", ai.Keywords) : string.Empty,
                    UploadDate = DateTime.UtcNow,
                    UserId     = UserId!.Value
                };
                _db.ResearchPapers.Add(paper);
                await _db.SaveChangesAsync();
                keepFile = true;

                // 5. Render result view
                return View("Result", new ResultViewModel
                {
                    Title           = paper.Title ?? file.FileName,
                    Summary         = ai.Summary         ?? "No summary generated.",
                    Keywords        = ai.Keywords         ?? Array.Empty<string>(),
                    ImportantPoints = ai.ImportantPoints  ?? Array.Empty<string>(),
                    Flow            = ai.Flow             ?? Array.Empty<string>(),
                    Tables          = ai.Tables           ?? Array.Empty<List<List<string>>>()
                });
            }
            finally
            {
                // Uploads of successful analyses are kept (as in V1); anything else is removed.
                if (!keepFile) TryDeleteUpload(savedPath);
            }
        }

        // ── GET /Paper/History ────────────────────────────────────────────────
        [HttpGet]
        public IActionResult History()
        {
            if (UserId == null) return RedirectToLogin();

            var papers = _db.ResearchPapers
                .Where(p => p.UserId == UserId)
                .OrderByDescending(p => p.UploadDate)
                .ToList();

            return View(papers);
        }

        // ── Upload file helpers ───────────────────────────────────────────────
        private string UploadsDirectory =>
            Path.GetFullPath(Path.IsPathRooted(_upload.Directory)
                ? _upload.Directory
                : Path.Combine(_env.ContentRootPath, _upload.Directory));

        private async Task<string> SaveUploadAsync(IFormFile file, CancellationToken cancellationToken)
        {
            var uploadsDir = UploadsDirectory;
            Directory.CreateDirectory(uploadsDir);

            var savedPath = Path.Combine(uploadsDir, $"{Guid.NewGuid():N}.pdf");
            try
            {
                await using var fs = new FileStream(savedPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                await file.CopyToAsync(fs, cancellationToken);
            }
            catch
            {
                TryDeleteUpload(savedPath);
                throw;
            }
            return savedPath;
        }

        private void TryDeleteUpload(string path)
        {
            try
            {
                System.IO.File.Delete(path);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _logger.LogWarning(ex, "Could not delete upload {Path}", path);
            }
        }

        // Display only: the client file name is never used to build a path.
        private static string SafeTitle(string clientFileName)
        {
            var name = clientFileName.Replace('\\', '/');
            name = name[(name.LastIndexOf('/') + 1)..].Trim();
            if (name.Length == 0) name = "document.pdf";
            return name.Length <= MaxTitleLength ? name : name[..MaxTitleLength];
        }
    }
}
