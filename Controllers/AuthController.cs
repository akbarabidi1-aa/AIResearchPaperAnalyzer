using System.Security.Cryptography;
using System.Text;
using AIResearchPaperAnalyzer.Models;
using Microsoft.AspNetCore.Mvc;

namespace AIResearchPaperAnalyzer.Controllers
{
    public class AuthController : Controller
    {
        private readonly AppDbContext _db;

        public AuthController(AppDbContext db) => _db = db;

        // ── GET /Auth/Login ───────────────────────────────────────────────────
        [HttpGet]
        public IActionResult Login()
        {
            if (HttpContext.Session.GetInt32("UserId") != null)
                return RedirectToAction("Index", "Paper");
            return View();
        }

        // ── POST /Auth/Login ──────────────────────────────────────────────────
        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult Login(LoginViewModel model)
        {
            if (!ModelState.IsValid)
                return View(model);

            var user = _db.Users.FirstOrDefault(u => u.Username == model.Username);
            if (user == null || user.PasswordHash != Hash(model.Password))
            {
                ModelState.AddModelError(string.Empty, "Invalid username or password.");
                return View(model);
            }

            HttpContext.Session.SetInt32("UserId",   user.Id);
            HttpContext.Session.SetString("Username", user.Username);
            return RedirectToAction("Index", "Paper");
        }

        // ── GET /Auth/Register ────────────────────────────────────────────────
        [HttpGet]
        public IActionResult Register() => View();

        // ── POST /Auth/Register ───────────────────────────────────────────────
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Register(RegisterViewModel model)
        {
            if (!ModelState.IsValid)
                return View(model);

            if (_db.Users.Any(u => u.Username == model.Username))
            {
                ModelState.AddModelError("Username", "That username is already taken.");
                return View(model);
            }

            _db.Users.Add(new User
            {
                Username     = model.Username,
                PasswordHash = Hash(model.Password)
            });
            await _db.SaveChangesAsync();

            TempData["Success"] = "Account created successfully. Please log in.";
            return RedirectToAction("Login");
        }

        // ── GET /Auth/Logout ──────────────────────────────────────────────────
        [HttpGet]
        public IActionResult Logout()
        {
            HttpContext.Session.Clear();
            return RedirectToAction("Login");
        }

        // ── Helper ────────────────────────────────────────────────────────────
        private static string Hash(string password)
        {
            using var sha = SHA256.Create();
            return Convert.ToBase64String(
                sha.ComputeHash(Encoding.UTF8.GetBytes(password)));
        }
    }
}
