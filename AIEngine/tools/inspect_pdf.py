"""Phase 2 check: render a PDF's pages, extract native page text, write pages.json, print a summary.

    python AIEngine/tools/inspect_pdf.py sample.pdf
    python AIEngine/tools/inspect_pdf.py sample.pdf --output temp/sample --dpi 180

Without --output the result goes to AIEngine/output/<name>-<hash>/.
Exit code: 0 = ok, 2 = the PDF or the output directory could not be processed.
"""
import argparse
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from pipeline import DEFAULT_DPI, PdfPipelineError, default_work_dir, process_pdf  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render PDF pages and extract native page text (Phase 2).")
    parser.add_argument("pdf", help="path to the PDF")
    parser.add_argument("--output", help="working directory (default: AIEngine/output/<name>-<hash>)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"render resolution (default {DEFAULT_DPI})")
    parser.add_argument("--copy-original", action="store_true", help="also store the PDF as original.pdf")
    args = parser.parse_args(argv)

    try:
        work_dir = args.output or default_work_dir(args.pdf, os.path.join(ENGINE_DIR, "output"))
        result = process_pdf(args.pdf, work_dir, dpi=args.dpi, copy_original=args.copy_original)
    except (PdfPipelineError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    info, timings = result["document"]["document"], result["timings"]
    print(f"Pages: {info['page_count']}")
    print(f"Rendered: {len(result['document']['pages'])}")
    print(f"Pages with native text: {info['pages_with_text_layer']}")
    print(f"Pages without native text: {info['pages_without_text_layer']}")
    print(f"Metadata: {result['metadata_path']}")
    print("Time: open {open_s:.3f}s, render {render_s:.3f}s, extract {extract_s:.3f}s, total {total_s:.3f}s"
          .format(**timings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
