"""Phase 3 check: detect the layout regions of every page, write layout.json, print a summary.

    python AIEngine/tools/inspect_layout.py sample.pdf
    python AIEngine/tools/inspect_layout.py sample.pdf --output temp/sample --visualize-layout
    python AIEngine/tools/inspect_layout.py temp/sample --layout-threshold 0.4

The input is a PDF (Phase 2 runs first, into --output or AIEngine/output/<name>-<hash>/) or an
existing Phase 2 working directory (its pages.json and page images are reused).
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
    PdfPipelineError,
    PPDocLayoutV3Detector,
    count_region_types,
    default_work_dir,
    detect_document_layout,
    process_pdf,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect document layout regions on rendered pages (Phase 3).")
    parser.add_argument("input", help="a PDF, or a Phase 2 working directory")
    parser.add_argument("--output", help="working directory for a PDF input (default: AIEngine/output/<name>-<hash>)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help=f"render resolution for a PDF input (default {DEFAULT_DPI})")
    parser.add_argument("--layout-threshold", type=float, default=DEFAULT_LAYOUT_THRESHOLD,
                        help=f"minimum detection score, 0-1 (default {DEFAULT_LAYOUT_THRESHOLD})")
    parser.add_argument("--visualize-layout", action="store_true",
                        help="also save layout_debug/page_NNN_layout.png with the boxes drawn")
    parser.add_argument("--device", default="cpu", help="inference device (default cpu)")
    args = parser.parse_args(argv)

    try:
        if os.path.isdir(args.input):
            work_dir = args.input
        else:
            work_dir = args.output or default_work_dir(args.input, os.path.join(ENGINE_DIR, "output"))
            process_pdf(args.input, work_dir, dpi=args.dpi)
        detector = PPDocLayoutV3Detector(device=args.device)
        result = detect_document_layout(work_dir, detector, threshold=args.layout_threshold,
                                        visualize=args.visualize_layout)
    except (PdfPipelineError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    document, timings = result["document"], result["timings"]
    page_count = len(document["pages"])
    print(f"Pages: {page_count}")
    print()
    print("Detected regions:")
    counts = count_region_types(document)
    for region_type, count in counts.items():
        print(f"{region_type}: {count}")
    if not counts:
        print("(none)")
    print()
    print(f"Model: {document['model']['name']}")
    print(f"Device: {args.device.upper()}")
    print(f"Threshold: {document['model']['threshold']}")
    print(f"Overlap warnings: {len(result['issues'])}")
    for issue in result["issues"]:
        print(f"  {issue['kind']}: {', '.join(issue['region_ids'])} ({issue['value']:.2f})")
    print(f"Layout: {result['layout_path']}")
    if result["visualizations"]:
        print(f"Visualizations: {os.path.dirname(result['visualizations'][0])}")
    print("Time: model init {:.2f}s, inference {:.2f}s ({:.2f}s/page), total {:.2f}s".format(
        timings["model_init_s"], timings["inference_s"], timings["inference_s"] / page_count, timings["total_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
