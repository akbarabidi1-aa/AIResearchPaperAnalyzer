"""Phase 6 check: turn every EQUATION region into LaTeX, write equations.json, print a summary.

    python AIEngine/tools/inspect_equations.py sample.pdf --output temp/sample
    python AIEngine/tools/inspect_equations.py sample.pdf --output temp/sample --save-equation-crops
    python AIEngine/tools/inspect_equations.py temp/sample --ocr-numbers

The input is a PDF or an existing working directory. Phase 2 (pages.json, page images) and Phase 3
(layout.json) are reused when the working directory already holds them for this PDF and these
settings, and run otherwise; --force runs them again. Without --output a PDF goes to
AIEngine/output/<name>-<hash>/.

Equation numbers are read from the native text of the PDF. On pages without native text they need the
OCR text of Phase 4: an ocr.json already in the working directory is used, and --ocr-numbers runs
Phase 4 when there is none.
Exit code: 0 = ok, 2 = the input could not be processed.
"""
import argparse
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from inspect_ocr import add_work_dir_arguments, prepare_work_dir  # noqa: E402
from inspect_tables import has_current_ocr  # noqa: E402
from pipeline import (  # noqa: E402
    DEFAULT_EQUATION_PADDING,
    PaddleOCREngine,
    PdfPipelineError,
    PPFormulaNetRecognizer,
    run_document_equations,
    run_document_ocr,
)
from pipeline.ocr import check_padding  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Equation recognition to LaTeX (Phase 6).")
    add_work_dir_arguments(parser)
    parser.add_argument("--equation-padding", type=int, default=DEFAULT_EQUATION_PADDING,
                        help=f"pixels of page added around an equation region (default {DEFAULT_EQUATION_PADDING})")
    parser.add_argument("--save-equation-crops", action="store_true",
                        help="also save equation_crops/page_NNN_eq_NNN.png for every equation "
                             "(equations that are not recognized always keep their crop)")
    parser.add_argument("--ocr-numbers", action="store_true",
                        help="run Phase 4 OCR if the working directory has no current ocr.json, so that "
                             "equation numbers of scanned pages can be read")
    args = parser.parse_args(argv)

    try:
        check_padding(args.equation_padding)
        work_dir, pages_reused, layout_reused = prepare_work_dir(args)
        ocr_state = "reused" if layout_reused and not args.force and has_current_ocr(work_dir) else "not available"
        if args.ocr_numbers and ocr_state != "reused":
            run_document_ocr(work_dir, PaddleOCREngine(device=args.device))
            ocr_state = "created"
        recognizer = PPFormulaNetRecognizer(device=args.device)     # loaded once, for all equations
        result = run_document_equations(work_dir, recognizer, padding=args.equation_padding,
                                        save_crops=args.save_equation_crops)
    except (PdfPipelineError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    document, summary, timings = result["document"], result["document"]["summary"], result["timings"]
    confidence = summary["mean_confidence"]
    print(f"Pages: {summary['pages']}")
    print(f"Equation regions: {summary['equation_regions']}")
    print(f"Recognized: {summary['recognized']}")
    print(f"Failed: {summary['invalid_latex'] + summary['recognition_failed']}")
    print(f"With equation numbers: {summary['with_equation_numbers']}")
    print("Average confidence: " + ("n/a (the model reports none)" if confidence is None else f"{confidence:.2f}"))
    print(f"Total time: {timings['total_s']:.1f}s")
    print()
    for page in document["pages"]:
        for equation in page["equations"]:
            number = equation["equation_number"] or "-"
            if equation["status"] == "success":
                detail = equation["latex"]
            elif equation["status"] == "invalid_latex":
                detail = f"INVALID ({'; '.join(equation['problems'])}): {equation['normalized_latex']}"
            else:
                detail = f"FAILED ({equation['error']})"
            print(f"  {equation['equation_id']} {number:>6}  {detail}")
    print(f"Invalid LaTeX: {summary['invalid_latex']}; recognition failed: {summary['recognition_failed']}")
    print(f"Recognizer: {recognizer.name}, device {args.device.upper()}, "
          + (f"loaded {recognizer.load_count} time(s)" if recognizer.loaded else "not loaded: no equation region"))
    print(f"Phase 2 pages: {'reused' if pages_reused else 'created'}; "
          f"Phase 3 layout: {'reused' if layout_reused else 'created'}; Phase 4 OCR: {ocr_state}")
    print(f"Equations: {result['equations_path']}")
    if result["crops"]:
        print(f"Crops: {os.path.dirname(result['crops'][0])} ({len(result['crops'])})")
    count = len(timings["equation_s"])
    average = timings["inference_s"] / count if count else 0.0
    print(f"Time: model init {timings['model_init_s']:.2f}s, inference {timings['inference_s']:.2f}s "
          f"({average:.2f}s/equation, slowest {max(timings['equation_s'].values(), default=0.0):.2f}s), "
          f"total {timings['total_s']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
