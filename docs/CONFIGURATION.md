# Configuration (Phase 1)

`appsettings.json` contains only portable defaults. Nothing in the repository points at a specific machine.

## Settings

| Key | Default | Meaning |
|---|---|---|
| `AIEngine:PythonExecutable` | `python` | A command found on `PATH`, or a path to the interpreter |
| `AIEngine:ScriptPath` | `AIEngine/ai_engine.py` | The AI engine entry script |
| `AIEngine:TimeoutSeconds` | `120` | The Python process is terminated after this many seconds |
| `Upload:MaxFileSizeBytes` | `52428800` (50 MB) | Largest PDF accepted |
| `Upload:Directory` | `Uploads` | Where uploaded PDFs are stored |

Relative paths are resolved from the content root (`IWebHostEnvironment.ContentRootPath`): the folder with the
`.csproj` when using `dotnet run`, or the publish folder under IIS. Section and key names are case-insensitive.

## Machine-specific overrides

Use one of these when `python` is not on `PATH`, or a specific interpreter / virtual environment is needed.
Do not edit `appsettings.json` for this.

**1. `appsettings.Local.json`** (next to `appsettings.json`; git-ignored, loaded if present):

```json
{
  "AIEngine": {
    "PythonExecutable": "C:\\Path\\To\\Python310\\python.exe"
  }
}
```

**2. Environment variables** (`__` separates section and key; these win over all JSON files):

```powershell
$env:AIEngine__PythonExecutable = "C:\Path\To\Python310\python.exe"
$env:AIEngine__TimeoutSeconds   = "300"
dotnet run --launch-profile AIResearchPaperAnalyzer
```

Under IIS, add them as `<environmentVariable>` entries in `web.config`, or set them on the application pool.

Keep connection strings with passwords and any other secrets out of the repository; put them in
`appsettings.Local.json` or environment variables.

## Upload size

The server accepts request bodies up to `Upload:MaxFileSizeBytes` + 1 MB, so a file just over the limit gets a
readable validation message. Larger requests are refused by the server with HTTP 413. Under IIS the request
filtering limit (`maxAllowedContentLength`, 30 MB by default) also applies and must be raised in `web.config`
to allow larger files.

## Uploaded files

- Stored as `Uploads/<guid>.pdf`. The client file name is never used to build a path; it is only kept as the
  paper title.
- A file is validated (non-empty, size, `.pdf` extension, `%PDF-` magic bytes) before it is written to disk.
- If analysis fails, times out, or the database save fails, the stored file is deleted.
- Files of successful analyses are kept, as in V1.

## Tests

```powershell
dotnet test tests/AIResearchPaperAnalyzer.Tests
```

The tests need `python` on `PATH`. One test (`Category=Integration`) runs the real `AIEngine/ai_engine.py` and
needs the packages from `AIEngine/requirements.txt` plus the NLTK data listed in `docs/V1_BASELINE.md`.
To skip it: `dotnet test tests/AIResearchPaperAnalyzer.Tests --filter "Category!=Integration"`.
