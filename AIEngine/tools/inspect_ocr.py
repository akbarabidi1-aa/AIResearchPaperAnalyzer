"""Phase 4 check: give every layout region its text (native first, OCR as fallback), write ocr.json,
print a summary.

    python AIEngine/tools/inspect_ocr.py sample.pdf --output temp/sample
    python AIEngine/tools/inspect_ocr.py sample.pdf --output temp/sample --save-ocr-crops
    python AIEngine/tools/inspect_ocr.py temp/sample --ocr-tables

The input is a PDF or an existing working directory. Phase 2 (pages.json, page images) and Phase 3
(layout.json) are reused when the working directory already holds them for this PDF and these
settings, and run otherwise; --force runs them again. Without --output a PDF goes to
AIEngine/output/<name>-<hash>/.
Exit code: 0 = ok, 2 = the input could not be processed.
"""
import argparse
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from pipeline import (  # noqa: E402
    DEFAULT_DPI,
    DEFAULT_LAYOUT_THRESHOLD,
    DEFAULT_OCR_PADDING,
    PaddleOCREngine,
    PdfPipelineError,
    PPDocLayoutV3Detector,
    default_work_dir,
    detect_document_layout,
    has_current_pages,
    load_layout_document,
    process_pdf,
    run_document_ocr,
)
from pipeline.layout import check_threshold, load_pages_document  # noqa: E402
from pipeline.ocr import check_padding  # noqa: E402


def has_current_layout(work_dir, threshold):
    """True when layout.json of `work_dir` fits its pages.json and was made with this threshold."""
    try:
        layout = load_layout_document(work_dir, load_pages_document(work_dir))
    except PdfPipelineError:
        return False
    return layout["model"]["threshold"] == threshold


def main(argv=None):
    parser = argparse.ArgumentParser(description="Region text with selective OCR (Phase 4).")
    parser.add_argument("input", help="a PDF, or a working directory of Phase 2 / Phase 3")
    parser.add_argument("--output", help="working directory for a PDF input (default: AIEngine/output/<name>-<hash>)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help=f"render resolution for a PDF input (default {DEFAULT_DPI})")
    parser.add_argument("--layout-threshold", type=float, default=DEFAULT_LAYOUT_THRESHOLD,
                        help=f"minimum layout detection score, 0-1 (default {DEFAULT_LAYOUT_THRESHOLD})")
    parser.add_argument("--ocr-padding", type=int, default=DEFAULT_OCR_PADDING,
                        help=f"pixels of page added around a region before OCR (default {DEFAULT_OCR_PADDING})")
    parser.add_argument("--ocr-tables", action="store_true",
                        help="also OCR table regions without native text, as raw text lines (no cells)")
    parser.add_argument("--save-ocr-crops", action="store_true",
                        help="also save ocr_crops/page_NNN_region_NNN.png for every OCR'd region")
    parser.add_argument("--force", action="store_true", help="run Phase 2 and Phase 3 again even if present")
    parser.add_argument("--device", default="cpu", help="inference device (default cpu)")
    args = parser.parse_args(argv)

    try:
        threshold = check_threshold(args.layout_threshold)
        check_padding(args.ocr_padding)
        if os.path.isdir(args.input):
            work_dir, pages_reused = args.input, True
        else:
            work_dir = args.output or default_work_dir(args.input, os.path.join(ENGINE_DIR, "output"))
            pages_reused = not args.force and has_current_pages(work_dir, args.input, args.dpi)
            if not pages_reused:
                process_pdf(args.input, work_dir, dpi=args.dpi)
        layout_reused = pages_reused and not args.force and has_current_layout(work_dir, threshold)
        if not layout_reused:
            detect_document_layout(work_dir, PPDocLayoutV3Detector(device=args.device), threshold=threshold)

        engine = PaddleOCREngine(device=args.device)
        result = run_document_ocr(work_dir, engine, padding=args.ocr_padding, ocr_tables=args.ocr_tables,
                                  save_crops=args.save_ocr_crops)
    except (PdfPipelineError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    document, summary, timings = result["document"], result["document"]["summary"], result["timings"]
    confidence = summary["mean_ocr_confidence"]
    print(f"Pages: {summary['pages']}")
    print(f"Text regions: {summary['text_regions']}")
    print(f"Native text used: {summary['native_regions']}")
    print(f"OCR regions: {summary['ocr_regions']}")
    print(f"OCR failures: {summary['ocr_failures']}")
    print("Average OCR confidence: " + ("n/a" if confidence is None else f"{confidence:.2f}"))
    print(f"OCR time: {timings['ocr_s']:.1f}s")
    print()
    print(f"Regions without text (figure, equation, table without native text): {summary['skipped_regions']}")
    print(f"OCR regions without any text found: {summary['ocr_empty']}")
    for page in document["pages"]:
        for region in page["regions"]:
            if "error" in region:
                print(f"  failed: {region['region_id']} ({region['error']})")
    print(f"Engine: {engine.name} ({engine.detection_model} + {engine.recognition_model}), "
          f"device {args.device.upper()}, " + ("loaded" if engine.loaded else "not loaded: no region needed OCR"))
    print(f"Phase 2 pages: {'reused' if pages_reused else 'created'}; "
          f"Phase 3 layout: {'reused' if layout_reused else 'created'}")
    print(f"OCR: {result['ocr_path']}")
    if result["crops"]:
        print(f"Crops: {os.path.dirname(result['crops'][0])} ({len(result['crops'])})")
    slowest = max(timings["region_ocr_s"].values(), default=0.0)
    print(f"Time: engine init {timings['engine_init_s']:.2f}s, OCR {timings['ocr_s']:.2f}s "
          f"(slowest region {slowest:.2f}s), total {timings['total_s']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
