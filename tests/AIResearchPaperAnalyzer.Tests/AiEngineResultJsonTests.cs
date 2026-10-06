using System.Text.Json;
using AIResearchPaperAnalyzer.Models;
using Xunit;

namespace AIResearchPaperAnalyzer.Tests
{
    public class AiEngineResultJsonTests
    {
        // Same shape and key names as json.dumps(result) in AIEngine/ai_engine.py.
        private const string EngineJson = """
            {"status": "success", "file_name": "abc.pdf", "summary": "A summary.",
             "keywords": ["network", "depth"],
             "important_points": ["First key point.", "Second key point."],
             "flow": ["Abstract", "1. Introduction"],
             "tables": [[["Model", "Acc", "F1"], ["Net-A", "0.9", "0.8"]]]}
            """;

        [Fact]
        public void Snake_case_important_points_is_mapped()
        {
            var result = JsonSerializer.Deserialize<AiEngineResult>(EngineJson)!;

            Assert.Equal(new[] { "First key point.", "Second key point." }, result.ImportantPoints);
        }

        [Fact]
        public void Every_engine_field_is_mapped()
        {
            var result = JsonSerializer.Deserialize<AiEngineResult>(EngineJson)!;

            Assert.Equal("success", result.Status);
            Assert.Equal("abc.pdf", result.FileName);
            Assert.Equal("A summary.", result.Summary);
            Assert.Equal(new[] { "network", "depth" }, result.Keywords);
            Assert.Equal(new[] { "Abstract", "1. Introduction" }, result.Flow);
            Assert.Single(result.Tables!);
            Assert.Equal("Net-A", result.Tables![0][1][0]);
        }

        [Fact]
        public void Error_payload_is_mapped()
        {
            var result = JsonSerializer.Deserialize<AiEngineResult>(
                """{"status": "error", "message": "No text found in PDF. Make sure it is not a scanned image."}""")!;

            Assert.Equal("error", result.Status);
            Assert.StartsWith("No text found", result.Message);
            Assert.Null(result.ImportantPoints);
        }

        [Fact]
        public void Missing_status_is_not_treated_as_success()
        {
            var result = JsonSerializer.Deserialize<AiEngineResult>("{}")!;

            Assert.NotEqual("success", result.Status);
        }
    }
}
