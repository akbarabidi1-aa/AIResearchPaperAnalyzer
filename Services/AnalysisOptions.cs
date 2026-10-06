namespace AIResearchPaperAnalyzer.Services
{
    // Bound to the "AIEngine" configuration section.
    public class AiEngineOptions
    {
        public const string SectionName = "AIEngine";
        public const int    DefaultTimeoutSeconds = 120;

        // A command on PATH ("python") or a path. Relative paths resolve from the content root.
        public string PythonExecutable { get; set; } = "python";

        // Relative paths resolve from the content root.
        public string ScriptPath       { get; set; } = "AIEngine/ai_engine.py";

        public int    TimeoutSeconds   { get; set; } = DefaultTimeoutSeconds;
    }

    // Bound to the "Upload" configuration section.
    public class UploadOptions
    {
        public const string SectionName = "Upload";
        public const long   DefaultMaxFileSizeBytes = 52_428_800;   // 50 MB

        public long   MaxFileSizeBytes { get; set; } = DefaultMaxFileSizeBytes;

        // Relative paths resolve from the content root.
        public string Directory        { get; set; } = "Uploads";

        public long EffectiveMaxFileSizeBytes =>
            MaxFileSizeBytes > 0 ? MaxFileSizeBytes : DefaultMaxFileSizeBytes;
    }
}
