"""Phase 7 check: extract every FIGURE region (crop, caption, type, text, chart data), write
figures.json, print a summary.

    python AIEngine/tools/inspect_figures.py sample.pdf --output temp/sample
    python AIEngine/tools/inspect_figures.py sample.pdf --output temp/sample --save-figure-debug
    python AIEngine/tools/inspect_figures.py temp/sample --no-figure-ocr

The input is a PDF or an existing working directory. Phase 2 (pages.json, page images), Phase 3
(layout.json) and Phase 4 (ocr.json) are reused when the working directory already holds them for this
PDF and these settings, and run otherwise; --force runs them again. Without --output a PDF goes to
AIEngine/output/<name>-<hash>/.

The crop of every figure is always saved (figure_images/); --save-figure-debug adds the padded crops
and the chart overlays (figure_crops/).
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
    DEFAULT_FIGURE_PADDING,
    FigureExtractor,
    PaddleOCREngine,
    PdfPipelineError,
    RuleBasedChartExtractor,
    run_document_figures,
    run_document_ocr,
)
from pipeline.ocr import check_padding  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Figure and chart extraction (Phase 7).")
    add_work_dir_arguments(parser)
    parser.add_argument("--figure-padding", type=int, default=DEFAULT_FIGURE_PADDING,
                        help=f"pixels of page added around a figure region (default {DEFAULT_FIGURE_PADDING})")
    parser.add_argument("--no-figure-ocr", action="store_true",
                        help="do not OCR figures that have no native text (their labels stay unread)")
    parser.add_argument("--save-figure-debug", action="store_true",
                        help="also save figure_crops/page_NNN_fig_NNN.png (the padded crop that was analysed) "
                             "and, for charts, page_NNN_fig_NNN_overlay.png")
    args = parser.parse_args(argv)

    try:
        check_padding(args.figure_padding)
        work_dir, pages_reused, layout_reused = prepare_work_dir(args)
        engine = PaddleOCREngine(device=args.device)        # one engine for Phase 4 and for figure text
        ocr_reused = layout_reused and not args.force and has_current_ocr(work_dir)
        if not ocr_reused:
            run_document_ocr(work_dir, engine)
        extractor = FigureExtractor(engine, RuleBasedChartExtractor(), padding=args.figure_padding,
                                    use_ocr=not args.no_figure_ocr)
        result = run_document_figures(work_dir, extractor, save_debug=args.save_figure_debug)
    except (PdfPipelineError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    document, summary, timings = result["document"], result["document"]["summary"], result["timings"]
    print(f"Pages: {summary['pages']}")
    print(f"Figure regions: {summary['figure_regions']}")
    print()
    print(f"Generic figures: {summary['generic_figures']}")
    print(f"Charts: {summary['charts']}")
    print()
    print(f"Captions matched: {summary['captions_matched']}")
    print(f"Figure numbers found: {summary['figure_numbers_found']}")
    print()
    print("Charts parsed:")
    for kind, count in summary["chart_types"].items():
        print(f"{kind.capitalize()}: {count}")
    if not summary["chart_types"]:
        print("none")
    print()
    print(f"Chart data extracted: {summary['chart_data_extracted']}/{summary['charts']}")
    print()
    print(f"Total time: {timings['total_s']:.1f}s")
    print()
    for page in document["pages"]:
        for figure in page["figures"]:
            if figure["status"] != "ok":
                print(f"  {figure['figure_id']}: FAILED ({figure['error']})")
                continue
            detail = f"{figure['figure_type']} ({figure['source_label']})"
            chart = figure["chart_data"]
            if chart:
                values = sum(len(series.get("bars", series.get("points", ()))) for series in chart["series"])
                detail += f", {chart['chart_type']}, {chart['status']}, {len(chart['series'])} series, {values} values"
                if chart["notes"]:
                    detail += f" [{'; '.join(chart['notes'])}]"
            text = f"{len(figure['ocr_text'])} text items ({figure['text_source']})" if figure["ocr_text"] else "no text"
            print(f"  {figure['figure_id']}: {detail}; {figure['figure_number'] or 'no number'}; "
                  f"caption {figure['caption_position'] or 'none'}; {text}")
    types = ", ".join(f"{kind} {count}" for kind, count in summary["figure_types"].items()) or "none"
    print(f"Figure types: {types}")
    print(f"Chart readings failed: {summary['chart_failures']}; figures failed: {summary['failures']}")
    print(f"Chart reader: {extractor.chart_extractor.name} (no model); OCR engine: {engine.name}, "
          f"device {args.device.upper()}, "
          + (f"loaded {engine.load_count} time(s)" if engine.loaded else "not loaded: no text needed OCR"))
    print(f"Phase 2 pages: {'reused' if pages_reused else 'created'}; "
          f"Phase 3 layout: {'reused' if layout_reused else 'created'}; "
          f"Phase 4 OCR: {'reused' if ocr_reused else 'created'}")
    print(f"Figures: {result['figures_path']}")
    if result["images"]:
        print(f"Figure images: {os.path.dirname(result['images'][0])} ({len(result['images'])})")
    if result["debug"]:
        print(f"Debug crops and overlays: {os.path.dirname(result['debug'][0])} ({len(result['debug'])})")
    count = len(timings["figure_s"])
    average = sum(timings["figure_s"].values()) / count if count else 0.0
    print(f"Time: OCR engine init {timings['engine_init_s']:.2f}s, figures {sum(timings['figure_s'].values()):.2f}s "
          f"({average:.2f}s/figure, slowest {max(timings['figure_s'].values(), default=0.0):.2f}s), "
          f"total {timings['total_s']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
