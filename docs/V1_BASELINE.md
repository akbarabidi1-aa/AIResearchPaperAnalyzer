# AI Research Paper Analyzer — V1 Baseline (Phase 0)

Verified on 2026-10-07 on Windows 10, by reading the code and by running it. No application code was
changed in Phase 0.

## 0. Environment (as verified)

| Item | Required by project | Found on this machine |
|---|---|---|
| .NET | `net8.0` (`AIResearchPaperAnalyzer.csproj`), EF Core 8.0.8 | SDK **10.0.401**, runtimes **10.0.12 only** (no .NET 8 runtime) |
| Python | 3.10 (`__pycache__` is `cpython-310`; `appsettings.json` points at Python310) | 3.10.9 |
| Database | SQL Server LocalDB `(localdb)\mssqllocaldb`, DB `AIResearchPaperDB` | LocalDB `MSSQLLocalDB` 17.0 |
| Python packages | `AIEngine/requirements.txt` | pdfminer.six 20260107, nltk 3.10.3, scikit-learn 1.7.2, pandas 2.3.3 |
| NLTK data | `punkt`, `stopwords` (downloaded at import) **and `punkt_tab`** (see defect D1) | installed in `%APPDATA%\nltk_data` |
| EF tool | `dotnet-ef` 8.x | installed globally, 8.0.8 |

Configured values:
- Python executable: `appsettings.json` -> `C:\Users\CP\...\Python310\python.exe` (a different machine; does not exist here). `appsettings.Development.json` overrides it with `python` (PATH).
- AI engine path: `appsettings.json` -> `D:\PC\AIResearchPaperAnalyzer\AIEngine\ai_engine.py` (does not exist). `appsettings.Development.json` overrides it with `AIEngine/ai_engine.py`.
- `web.config` forces `ASPNETCORE_ENVIRONMENT=Production` under IIS, which would use the broken paths from `appsettings.json`.

Because only the .NET 10 runtime is installed, the net8.0 app is started with the process-level variable
`DOTNET_ROLL_FORWARD=Major`. The project file is unchanged. Installing the .NET 8 runtime removes the need for it.

## 1. Current architecture

```
Browser -> ASP.NET Core 8 MVC (Program.cs, in-memory Session)
             |- AuthController   (Login / Register / Logout)
             |- PaperController  (Index / Upload / History, RunPythonAi)
             |- Views            (Auth/*, Paper/Index|Result|History, Shared/_Layout)
             |- AppDbContext (EF Core) -> SQL Server LocalDB (Users, ResearchPapers)
             |- Uploads/<guid>.pdf  (disk; gitignored; never deleted)
             '- Process.Start("python ai_engine.py <pdf>") -> JSON on stdout
                  Python AIEngine: extraction -> preprocess -> keywords -> summary
```

## 2. C# -> Python communication

`PaperController.RunPythonAi(pdfPath)` uses `ProcessStartInfo` (`FileName = AiEngine:PythonExe`,
`Arguments = "\"<script>\" \"<pdf>\""`, stdout/stderr redirected, `UseShellExecute=false`). It reads all
of stdout, `WaitForExitAsync`, then `JsonSerializer.Deserialize<AiEngineResult>` with
`PropertyNameCaseInsensitive = true`. No REST, no temp files. stderr and the exit code are ignored, and there is no timeout.

## 3. Python processing pipeline (`AIEngine/ai_engine.py: run_analysis`)

1. `extraction.extract_pdf_content` — pdfminer `extract_text` (`LAParams(line_margin=0.5, char_margin=2.0, detect_vertical=True)`); regex heuristic tables; regex section lines (`flow_steps`, max 10).
2. If no text: prints `{"status":"error","message":"No text found in PDF..."}`.
3. `preprocess.clean_text` — NLTK `word_tokenize`, lowercase, drop stopwords, punctuation, non-alphabetic tokens.
4. `keyword_extractor.extract_keywords(top_n=10)` — sklearn `TfidfVectorizer` fitted on one document (IDF = 1.0 for every term, so this is ranking by word frequency).
5. `summarizer.summarize_large_text` — regex sentence split (> 40 chars), word-frequency scoring (4+ letter words, no stopword removal), top 5 sentences in original order.
6. `important_points` = the summary re-split into sentences (> 30 chars), first 5.
7. `flow` = `flow_steps`, or a fixed default list (Introduction ... Conclusion) when none are found; `tables[:3]`.
8. `print(json.dumps(result))`.

`pdf_reader.py` is a standalone prototype and is not part of the pipeline.

## 4. Database persistence

- Tables: `Users(Id, Username[100] unique, PasswordHash)`, `ResearchPapers(Id, Title[260], Summary, Keywords, UploadDate, UserId FK cascade)`.
- Migration: `20260319031318_InitialCreate`. Applied with `dotnet ef database update`; no `Migrate()` call at startup.
- Persisted per upload: Title (= original file name), Summary, Keywords (comma-joined), UploadDate, UserId.
- NOT persisted: important points, flow/structure, tables, the PDF path.
- Auth: session only (`UserId`, `Username`); passwords are unsalted SHA-256 (base64).

## 5. Supported capabilities (verified by running)

| Capability | Status | Evidence |
|---|---|---|
| Register / login / logout (session) | SUPPORTED | HTTP flow test |
| Upload PDF, save to `Uploads/` | SUPPORTED | 3 files saved |
| Digital single-column PDF analysis | SUPPORTED | CLI and web |
| Summary, keywords, section structure | SUPPORTED (basic, see section 6) | CLI and web |
| Result page, History page | SUPPORTED | HTTP flow test |
| `ResearchPaper` row saved | SUPPORTED | SQL query |
| Scanned PDFs / OCR | NOT SUPPORTED | error message only |
| Tables | PARTIALLY | heuristic, unreliable |
| Multi-column reading order | PARTIALLY | pdfminer defaults only |
| Charts, figures, captions, equations, LaTeX | NOT SUPPORTED | |

## 6. Known limitations

No OCR; no page/region/coordinate data; no figure, chart, caption or equation handling; references are included
in summary and keyword input; whole document processed in one pass (no chunking); keywords are unigram
frequency only; results are only viewable right after upload (no per-paper details page); the web request
blocks while Python runs; uploads accumulate forever; hard-coded foreign paths in `appsettings.json`;
unsalted SHA-256 passwords; GET logout; sessions lost on restart.

## 7. Defects reproduced (not fixed in Phase 0)

Fixtures: `tests/fixtures/*/synthetic_*.pdf` (synthetic, generated by `tests/fixtures/make_synthetic_fixtures.py`).

| ID | ISSUE | HOW TO REPRODUCE | EXPECTED | CURRENT RESULT | FILE RESPONSIBLE |
|---|---|---|---|---|---|
| D1 | **Fresh install fails on NLTK `punkt_tab`** | `pip install -r AIEngine/requirements.txt` (installs nltk 3.10.3) on a machine without that data, then run the CLI on any digital PDF | JSON `status: success` | `{"status":"error","message":"... Resource 'punkt_tab' not found ..."}` for every PDF with text. Worked around in this environment with `python -m nltk.downloader punkt punkt_tab stopwords` | `AIEngine/preprocess.py` (downloads only `punkt`, `stopwords`) |
| D2 | **Key Points never shown (snake_case mismatch)** | Upload `digital/synthetic_digital.pdf`. CLI JSON contains 5 `important_points`; the Result page has no "Key Points" card | Key Points card with up to 5 items | Card absent. C# DTO property `ImportantPoints` does not match JSON key `important_points`; `PropertyNameCaseInsensitive` does not ignore underscores. `file_name` -> `FileName` is also unbound (unused) | `Models/Models.cs` (`AiEngineResult`), `Controllers/PaperController.cs` (`RunPythonAi`) |
| D3 | **Scanned PDF fails; misleading UI and DB row** | Upload `scanned/synthetic_scanned.pdf` | Text recovered via OCR; at minimum a clear failure state | Python returns "No text found in PDF. Make sure it is not a scanned image." The page still says "Analysis Complete" with a warning banner, and a `ResearchPaper` row is saved with `Summary = "Analysis pending."` and empty keywords | `AIEngine/ai_engine.py` (no OCR); `PaperController.Upload` (saves on error) |
| D4 | **Table extraction unreliable** | (a) CLI on `digital/synthetic_digital.pdf`; (b) a PDF line `A. Smith    B. Jones    C. Lee` (scratch test, not in repo) | (a) the 4x4 results table; (b) no table | (a) `tables: []`. pdfminer emits the table column by column (one cell per line: `Model`, `Net-A`, `Net-B`...), so the whitespace heuristic never sees a row. (b) false positive: an author list becomes a "table". Captions are not linked | `AIEngine/extraction.py` (regex table block) |
| D5 | **Multi-column / structure** | CLI on `multicolumn/synthetic_two_column.pdf` | Left column then right column; detected sections | Column order was correct here, but this synthetic gap is wide and clean, so it does not prove real two-column papers work (no real fixture available). `flow` shows the fixed default list (Introduction ... Conclusion) although the document has no sections: a fake structure | `AIEngine/extraction.py` (`LAParams`), `AIEngine/ai_engine.py` (default `flow` fallback) |
| D6 | **Summary quality** | CLI on `digital/synthetic_digital.pdf` | Clean sentences from the content | Headings are glued into sentences ("Effect of Network Depth on Accuracy Abstract Neural networks are ..." and "Results The deeper network ..."); duplicate sentences are not de-duplicated; key points are the same sentences as the summary | `AIEngine/summarizer.py`, `AIEngine/ai_engine.py` (`important_points`) |
| D7 | **Keywords = word frequency** | Fit `TfidfVectorizer` on one document | Corpus-weighted keywords | Every IDF value is 1.0 (verified); ranking equals raw counts (accuracy 18, network 12, depth 11); singular and plural both appear (`network`, `networks`); no phrases | `AIEngine/keyword_extractor.py` |

Not a defect in code but relevant: the CLI exits 0 even for `status: error` (only a missing argument gives exit 1).

## 8. Exact run commands (PowerShell, from the repository root)

```powershell
# One-time: Python dependencies and NLTK data (punkt_tab is needed, see D1)
python -m pip install -r AIEngine/requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords

# One-time: EF tool and database
dotnet tool install --global dotnet-ef --version 8.0.8
sqllocaldb start MSSQLLocalDB

# Optional: regenerate synthetic test PDFs
python tests/fixtures/make_synthetic_fixtures.py

# CLI (independent of ASP.NET); stdout is a single JSON object
python AIEngine/ai_engine.py tests/fixtures/digital/synthetic_digital.pdf

# Build
dotnet build AIResearchPaperAnalyzer.csproj

# Database migration (applies the existing InitialCreate migration)
dotnet ef database update

# Run ASP.NET (Development profile -> https://localhost:7001, http://localhost:5001)
# Only needed while the .NET 8 runtime is not installed (only .NET 10 present):
$env:DOTNET_ROLL_FORWARD = "Major"
dotnet run --launch-profile AIResearchPaperAnalyzer
```

Run `dotnet ef` and `dotnet run` from the repository root (the folder containing the `.csproj`) so relative
paths in `appsettings.Development.json` resolve.
