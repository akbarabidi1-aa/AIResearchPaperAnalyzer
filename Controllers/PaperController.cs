using System.Diagnostics;
using System.Text.Json;
using AIResearchPaperAnalyzer.Models;
using Microsoft.AspNetCore.Mvc;

namespace AIResearchPaperAnalyzer.Controllers
{
    public class PaperController : Controller
    {
        private readonly AppDbContext   _db;
        private readonly IConfiguration _cfg;

        public PaperController(AppDbContext db, IConfiguration cfg)
        {
            _db  = db;
            _cfg = cfg;
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
        [HttpPost]
        [ValidateAntiForgeryToken]
        [RequestSizeLimit(52_428_800)]          // 50 MB max
        public async Task<IActionResult> Upload(UploadViewModel model)
        {
            if (UserId == null) return RedirectToLogin();

            // Validation
            if (model.File == null || model.File.Length == 0)
            {
                ModelState.AddModelError("File", "Please select a PDF file.");
                return View("Index", model);
            }
            if (!model.File.FileName.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase))
            {
                ModelState.AddModelError("File", "Only PDF files are accepted.");
                return View("Index", model);
            }

            // 1. Save PDF to disk
            var uploadsDir = Path.Combine(Directory.GetCurrentDirectory(), "Uploads");
            Directory.CreateDirectory(uploadsDir);
            var savedPath = Path.Combine(uploadsDir, $"{Guid.NewGuid()}.pdf");

            await using (var fs = new FileStream(savedPath, FileMode.Create))
                await model.File.CopyToAsync(fs);

            // 2. Run Python AI engine
            var ai = await RunPythonAi(savedPath);

            // 3. Persist result to database
            var paper = new ResearchPaper
            {
                Title      = Path.GetFileName(model.File.FileName),
                Summary    = ai.Summary  ?? "Analysis pending.",
                Keywords   = ai.Keywords != null ? string.Join(", ", ai.Keywords) : string.Empty,
                UploadDate = DateTime.UtcNow,
                UserId     = UserId!.Value
            };
            _db.ResearchPapers.Add(paper);
            await _db.SaveChangesAsync();

            // 4. Render result view
            return View("Result", new ResultViewModel
            {
                Title           = paper.Title ?? model.File.FileName,
                Summary         = ai.Summary         ?? "No summary generated.",
                Keywords        = ai.Keywords         ?? Array.Empty<string>(),
                ImportantPoints = ai.ImportantPoints  ?? Array.Empty<string>(),
                Flow            = ai.Flow             ?? Array.Empty<string>(),
                Tables          = ai.Tables           ?? Array.Empty<List<List<string>>>(),
                ErrorMessage    = ai.Status == "error" ? ai.Message : null
            });
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

        // ── Python AI helper ──────────────────────────────────────────────────
        private async Task<AiEngineResult> RunPythonAi(string pdfPath)
        {
            var script = _cfg["AiEngine:ScriptPath"]
                         ?? Path.Combine(Directory.GetCurrentDirectory(), "AIEngine", "ai_engine.py");
            var python = _cfg["AiEngine:PythonExe"] ?? "python";

            var psi = new ProcessStartInfo
            {
                FileName               = python,
                Arguments              = $"\"{script}\" \"{pdfPath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
                UseShellExecute        = false,
                CreateNoWindow         = true
            };

            using var proc = new Process { StartInfo = psi };
            proc.Start();
            var stdout = await proc.StandardOutput.ReadToEndAsync();
            await proc.WaitForExitAsync();

            if (string.IsNullOrWhiteSpace(stdout))
                return new AiEngineResult { Status = "error", Message = "AI engine produced no output." };

            try
            {
                return JsonSerializer.Deserialize<AiEngineResult>(stdout,
                           new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
                       ?? new AiEngineResult { Status = "error", Message = "Empty AI response." };
            }
            catch (JsonException ex)
            {
                return new AiEngineResult { Status = "error", Message = ex.Message };
            }
        }
    }
}
