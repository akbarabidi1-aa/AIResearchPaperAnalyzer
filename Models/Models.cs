using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;

namespace AIResearchPaperAnalyzer.Models
{
    // ═══════════════════════════════════════════════════════════════════════════
    //  DATABASE ENTITIES
    // ═══════════════════════════════════════════════════════════════════════════

    public class User
    {
        [Key]
        public int    Id           { get; set; }

        [Required, MaxLength(100)]
        public string Username     { get; set; } = string.Empty;

        [Required]
        public string PasswordHash { get; set; } = string.Empty;

        public ICollection<ResearchPaper> Papers { get; set; } = new List<ResearchPaper>();
    }

    public class ResearchPaper
    {
        [Key]
        public int      Id         { get; set; }

        [MaxLength(260)]
        public string?  Title      { get; set; }

        public string?  Summary    { get; set; }

        public string?  Keywords   { get; set; }

        public DateTime UploadDate { get; set; }

        [ForeignKey(nameof(User))]
        public int      UserId     { get; set; }

        public User?    User       { get; set; }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  DB CONTEXT
    // ═══════════════════════════════════════════════════════════════════════════

    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

        public DbSet<User>          Users          { get; set; }
        public DbSet<ResearchPaper> ResearchPapers { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<User>()
                .HasIndex(u => u.Username)
                .IsUnique();

            modelBuilder.Entity<ResearchPaper>()
                .HasOne(p => p.User)
                .WithMany(u => u.Papers)
                .HasForeignKey(p => p.UserId)
                .OnDelete(DeleteBehavior.Cascade);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  VIEW MODELS
    // ═══════════════════════════════════════════════════════════════════════════

    public class LoginViewModel
    {
        [Required(ErrorMessage = "Username is required.")]
        [Display(Name = "Username")]
        public string Username { get; set; } = string.Empty;

        [Required(ErrorMessage = "Password is required.")]
        [DataType(DataType.Password)]
        [Display(Name = "Password")]
        public string Password { get; set; } = string.Empty;
    }

    public class RegisterViewModel
    {
        [Required(ErrorMessage = "Username is required.")]
        [MinLength(3, ErrorMessage = "Username must be at least 3 characters.")]
        [MaxLength(50, ErrorMessage = "Username must be at most 50 characters.")]
        [Display(Name = "Username")]
        public string Username { get; set; } = string.Empty;

        [Required(ErrorMessage = "Password is required.")]
        [MinLength(6, ErrorMessage = "Password must be at least 6 characters.")]
        [DataType(DataType.Password)]
        [Display(Name = "Password")]
        public string Password { get; set; } = string.Empty;

        [Required(ErrorMessage = "Please confirm your password.")]
        [DataType(DataType.Password)]
        [Compare(nameof(Password), ErrorMessage = "Passwords do not match.")]
        [Display(Name = "Confirm Password")]
        public string ConfirmPassword { get; set; } = string.Empty;
    }

    public class UploadViewModel
    {
        [Display(Name = "PDF File")]
        public IFormFile? File { get; set; }
    }

    public class ResultViewModel
    {
        public string               Title           { get; set; } = string.Empty;
        public string               Summary         { get; set; } = string.Empty;
        public string[]             Keywords        { get; set; } = Array.Empty<string>();
        public string[]             ImportantPoints { get; set; } = Array.Empty<string>();
        public string[]             Flow            { get; set; } = Array.Empty<string>();
        public List<List<string>>[] Tables          { get; set; } = Array.Empty<List<List<string>>>();
        public string?              ErrorMessage    { get; set; }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  AI ENGINE RESULT DTO  (matches JSON output of ai_engine.py)
    // ═══════════════════════════════════════════════════════════════════════════

    public class AiEngineResult
    {
        public string                Status          { get; set; } = "success";
        public string?               Message         { get; set; }
        public string?               FileName        { get; set; }
        public string?               Summary         { get; set; }
        public string[]?             Keywords        { get; set; }
        public string[]?             ImportantPoints { get; set; }
        public string[]?             Flow            { get; set; }
        public List<List<string>>[]? Tables          { get; set; }
    }
}
