using Microsoft.Extensions.Options;

namespace AIResearchPaperAnalyzer.Services
{
    public enum PdfValidationError
    {
        None,
        Missing,
        Empty,
        TooLarge,
        NotPdfExtension,
        NotPdfContent
    }

    public sealed record PdfValidationResult(PdfValidationError Error, string? Message)
    {
        public bool IsValid => Error == PdfValidationError.None;

        public static readonly PdfValidationResult Valid = new(PdfValidationError.None, null);
    }

    // Checks an upload before it is written to disk or handed to the AI engine.
    public class PdfUploadValidator
    {
        // Every PDF starts with "%PDF-" (e.g. "%PDF-1.7").
        private static readonly byte[] PdfMagic = "%PDF-"u8.ToArray();

        private readonly UploadOptions _options;

        public PdfUploadValidator(IOptions<UploadOptions> options) => _options = options.Value;

        public long MaxFileSizeBytes => _options.EffectiveMaxFileSizeBytes;

        public PdfValidationResult Validate(IFormFile? file)
        {
            if (file == null)
                return new(PdfValidationError.Missing, "Please select a PDF file.");

            if (file.Length == 0)
                return new(PdfValidationError.Empty, "The selected file is empty. Please choose a valid PDF.");

            // Size and extension first, so an oversized upload is never opened.
            var early = ValidateNameAndLength(file.FileName, file.Length);
            if (!early.IsValid) return early;

            using var stream = file.OpenReadStream();
            return ValidateContent(stream);
        }

        public PdfValidationResult Validate(string? fileName, long length, Stream content)
        {
            if (length == 0)
                return new(PdfValidationError.Empty, "The selected file is empty. Please choose a valid PDF.");

            var early = ValidateNameAndLength(fileName, length);
            return early.IsValid ? ValidateContent(content) : early;
        }

        private PdfValidationResult ValidateNameAndLength(string? fileName, long length)
        {
            if (length > MaxFileSizeBytes)
                return new(PdfValidationError.TooLarge,
                    $"The file is too large. The maximum allowed size is {FormatSize(MaxFileSizeBytes)}.");

            if (!string.Equals(Path.GetExtension(fileName), ".pdf", StringComparison.OrdinalIgnoreCase))
                return new(PdfValidationError.NotPdfExtension, "Only PDF files are accepted.");

            return PdfValidationResult.Valid;
        }

        private static PdfValidationResult ValidateContent(Stream content)
        {
            var header = new byte[PdfMagic.Length];
            var read   = content.ReadAtLeast(header, header.Length, throwOnEndOfStream: false);

            if (read < header.Length || !header.AsSpan().SequenceEqual(PdfMagic))
                return new(PdfValidationError.NotPdfContent,
                    "This file is not a valid PDF. Please upload a real PDF document.");

            return PdfValidationResult.Valid;
        }

        private static string FormatSize(long bytes) =>
            bytes >= 1_048_576 ? $"{bytes / 1_048_576d:0.#} MB" : $"{bytes / 1024d:0.#} KB";
    }
}
