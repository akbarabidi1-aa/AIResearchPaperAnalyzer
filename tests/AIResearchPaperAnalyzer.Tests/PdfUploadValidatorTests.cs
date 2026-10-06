using System.Text;
using AIResearchPaperAnalyzer.Services;
using Xunit;

namespace AIResearchPaperAnalyzer.Tests
{
    public class PdfUploadValidatorTests
    {
        [Fact]
        public void Valid_pdf_is_accepted()
        {
            var result = TestData.Validator().Validate(TestData.FormFile(TestData.PdfBytes(), "paper.pdf"));

            Assert.True(result.IsValid);
            Assert.Null(result.Message);
        }

        [Fact]
        public void Real_fixture_pdf_is_accepted()
        {
            var bytes = File.ReadAllBytes(TestData.DigitalFixture());

            Assert.True(TestData.Validator().Validate(TestData.FormFile(bytes, "synthetic_digital.pdf")).IsValid);
        }

        [Fact]
        public void Pdf_extension_is_case_insensitive()
        {
            Assert.True(TestData.Validator().Validate(TestData.FormFile(TestData.PdfBytes(), "PAPER.PDF")).IsValid);
        }

        [Fact]
        public void Pdf_name_with_wrong_magic_bytes_is_rejected()
        {
            var notPdf = Encoding.ASCII.GetBytes("MZ this is really an executable, renamed to .pdf");

            var result = TestData.Validator().Validate(TestData.FormFile(notPdf, "paper.pdf"));

            Assert.Equal(PdfValidationError.NotPdfContent, result.Error);
            Assert.False(string.IsNullOrWhiteSpace(result.Message));
        }

        [Fact]
        public void File_shorter_than_the_magic_bytes_is_rejected()
        {
            var result = TestData.Validator().Validate(TestData.FormFile("%PD"u8.ToArray(), "paper.pdf"));

            Assert.Equal(PdfValidationError.NotPdfContent, result.Error);
        }

        [Fact]
        public void Magic_bytes_not_at_the_start_are_rejected()
        {
            var result = TestData.Validator().Validate(
                TestData.FormFile(Encoding.ASCII.GetBytes("junk %PDF-1.4 more"), "paper.pdf"));

            Assert.Equal(PdfValidationError.NotPdfContent, result.Error);
        }

        [Fact]
        public void Empty_file_is_rejected()
        {
            var result = TestData.Validator().Validate(TestData.FormFile(Array.Empty<byte>(), "paper.pdf"));

            Assert.Equal(PdfValidationError.Empty, result.Error);
        }

        [Fact]
        public void Missing_file_is_rejected()
        {
            Assert.Equal(PdfValidationError.Missing, TestData.Validator().Validate(null).Error);
        }

        [Fact]
        public void Oversized_file_is_rejected_with_the_configured_limit()
        {
            var validator = TestData.Validator(maxBytes: 1024);

            var result = validator.Validate(TestData.FormFile(TestData.PdfBytes(2048), "paper.pdf"));

            Assert.Equal(PdfValidationError.TooLarge, result.Error);
            Assert.Contains("1 KB", result.Message);
        }

        [Fact]
        public void File_exactly_at_the_limit_is_accepted()
        {
            var validator = TestData.Validator(maxBytes: 1024);

            Assert.True(validator.Validate(TestData.FormFile(TestData.PdfBytes(1024), "paper.pdf")).IsValid);
        }

        [Theory]
        [InlineData("paper.txt")]
        [InlineData("paper.pdf.exe")]
        [InlineData("paper")]
        public void Real_pdf_content_with_wrong_extension_is_rejected(string fileName)
        {
            var result = TestData.Validator().Validate(TestData.FormFile(TestData.PdfBytes(), fileName));

            Assert.Equal(PdfValidationError.NotPdfExtension, result.Error);
        }
    }
}
