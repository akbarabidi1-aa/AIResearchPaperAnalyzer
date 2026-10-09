"""Phase 5 check: turn every TABLE region into rows, columns and cells, write tables.json, print a summary.

    python AIEngine/tools/inspect_tables.py sample.pdf --output temp/sample
    python AIEngine/tools/inspect_tables.py sample.pdf --output temp/sample --visualize-tables
    python AIEngine/tools/inspect_tables.py temp/sample --table-mode hybrid

The input is a PDF or an existing working directory. Phase 2 (pages.json, page images), Phase 3
(layout.json) and Phase 4 (ocr.json) are reused when the working directory already holds them for this
PDF and these settings, and run otherwise; --force runs them again. Without --output a PDF goes to
AIEngine/output/<name>-<hash>/.
Exit code: 0 = ok, 2 = the input could not be processed.
"""
import argparse
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from inspect_ocr import add_work_dir_arguments, prepare_work_dir  # noqa: E402
from pipeline import (  # noqa: E402
    PaddleOCREngine,
    PdfPipelineError,
    SLANetPlusStructureModel,
    TableExtractor,
    load_layout_document,
    run_document_ocr,
    run_document_tables,
)
from pipeline.layout import load_pages_document  # noqa: E402
from pipeline.tables import MODES, load_ocr_document  # noqa: E402


def has_current_ocr(work_dir):
    """True when ocr.json of `work_dir` fits its layout.json."""
    try:
        load_ocr_document(work_dir, load_layout_document(work_dir, load_pages_document(work_dir)))
    except PdfPipelineError:
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Structured table extraction (Phase 5).")
    add_work_dir_arguments(parser)
    parser.add_argument("--table-mode", choices=MODES, default="auto",
                        help="auto (default): native geometry when the table is a regular grid of native text, "
                             "hybrid when it has native text but no regular grid, vision otherwise")
    parser.add_argument("--visualize-tables", action="store_true",
                        help="also save tables_debug/page_NNN_table_NNN_debug.png with the grid drawn")
    args = parser.parse_args(argv)

    try:
        work_dir, pages_reused, layout_reused = prepare_work_dir(args)
        engine = PaddleOCREngine(device=args.device)        # one engine for Phase 4 and for table cells
        ocr_reused = layout_reused and not args.force and has_current_ocr(work_dir)
        if not ocr_reused:
            run_document_ocr(work_dir, engine)
        extractor = TableExtractor(SLANetPlusStructureModel(device=args.device), engine)
        result = run_document_tables(work_dir, extractor, mode=args.table_mode, visualize=args.visualize_tables)
    except (PdfPipelineError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    document, summary, timings = result["document"], result["document"]["summary"], result["timings"]
    print(f"Pages: {summary['pages']}")
    print(f"Table regions: {summary['table_regions']}")
    print(f"Tables extracted: {summary['tables_extracted']}")
    print()
    print(f"Native: {summary['native']}")
    print(f"Vision: {summary['vision']}")
    print(f"Hybrid: {summary['hybrid']}")
    print()
    print(f"Cells: {summary['cells']}")
    print(f"Extraction failures: {summary['extraction_failures']}")
    print(f"Time: {timings['total_s']:.1f}s")
    print()
    for page in document["pages"]:
        for table in page["tables"]:
            if "error" in table:
                print(f"  {table['table_id']}: failed ({table['error']})")
                continue
            headers = "unknown" if table["headers"] is None else f"{len(table['header_rows'])} row(s)"
            merged = sum(1 for cell in table["cells"] if cell["row_span"] > 1 or cell["col_span"] > 1)
            print(f"  {table['table_id']}: {table['source']}, {table['n_rows']} x {table['n_columns']}, "
                  f"headers {headers}, merged cells {merged}, confidence {table['structure_confidence']:.2f}, "
                  f"caption {'yes' if table['caption'] else 'no'}")
    print(f"Tables with headers: {summary['with_headers']}; with caption: {summary['with_caption']}")
    print(f"Structure model: {extractor.structure_model.name}, "
          + ("loaded" if extractor.structure_model.loaded else "not loaded: no table needed it")
          + f"; OCR engine: {engine.name}, " + ("loaded" if engine.loaded else "not loaded: no text needed OCR"))
    print(f"Phase 2 pages: {'reused' if pages_reused else 'created'}; "
          f"Phase 3 layout: {'reused' if layout_reused else 'created'}; "
          f"Phase 4 OCR: {'reused' if ocr_reused else 'created'}")
    print(f"Tables: {result['tables_path']}")
    if result["visualizations"]:
        print(f"Visualizations: {os.path.dirname(result['visualizations'][0])} ({len(result['visualizations'])})")
    slowest = max(timings["table_s"].values(), default=0.0)
    print(f"Table time: model init {timings['model_init_s']:.2f}s, extraction {sum(timings['table_s'].values()):.2f}s "
          f"(slowest table {slowest:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
