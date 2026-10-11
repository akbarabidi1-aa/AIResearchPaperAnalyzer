"""Phase 8: reconstruct the document (reading order, sections, tables, figures, equations) and write
document.json.

    python AIEngine/tools/build_document.py sample.pdf --output temp/sample
    python AIEngine/tools/build_document.py sample.pdf --output temp/sample --visualize-order
    python AIEngine/tools/build_document.py temp/sample --truth tests/fixtures/document/synthetic_papers.json

The input is a PDF or an existing working directory. Every earlier phase is reused when the working
directory already holds its result for this PDF and these settings, and run otherwise: Phase 2 (pages),
3 (layout), 4 (OCR), 5 (tables), 6 (equations), 7 (figures). --force runs all of them again. Without
--output a PDF goes to AIEngine/output/<name>-<hash>/.

--truth compares the result with the ground truth of a fixture (the entry of the JSON file named like
the PDF, or the one given with --truth-name) and prints reading-order and section metrics.
Exit code: 0 = ok, 2 = the input could not be processed or the document failed its checks.
"""
import argparse
import json
import os
import sys
import time

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from inspect_ocr import add_work_dir_arguments, prepare_work_dir  # noqa: E402
from inspect_tables import has_current_ocr  # noqa: E402
from pipeline import (  # noqa: E402
    FigureExtractor,
    PaddleOCREngine,
    PdfPipelineError,
    PPFormulaNetRecognizer,
    RuleBasedChartExtractor,
    SLANetPlusStructureModel,
    TableExtractor,
    build_document,
    evaluate_document,
    load_artifacts,
    run_document_equations,
    run_document_figures,
    run_document_ocr,
    run_document_tables,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Document reconstruction and reading order (Phase 8).")
    add_work_dir_arguments(parser)
    parser.add_argument("--visualize-order", action="store_true",
                        help="also save order_debug/page_NNN_order.png with the blocks numbered in reading order")
    parser.add_argument("--truth", help="ground-truth JSON of a fixture, to print reading-order and section metrics")
    parser.add_argument("--truth-name", help="name of the document inside --truth (default: the name of the PDF)")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    try:
        work_dir, pages_reused, layout_reused = prepare_work_dir(args)
        engine = PaddleOCREngine(device=args.device)        # one OCR engine for Phases 4, 5 and 7
        reused = {"Phase 2 pages": pages_reused, "Phase 3 layout": layout_reused}
        reused["Phase 4 OCR"] = layout_reused and not args.force and has_current_ocr(work_dir)
        if not reused["Phase 4 OCR"]:
            run_document_ocr(work_dir, engine)
        present = {}
        if reused["Phase 4 OCR"]:
            try:
                present = load_artifacts(work_dir)
            except PdfPipelineError:
                present = {}        # a result of another layout: Phases 5-7 run again
        steps = (
            ("Phase 5 tables", "tables", lambda: run_document_tables(
                work_dir, TableExtractor(SLANetPlusStructureModel(device=args.device), engine))),
            ("Phase 6 equations", "equations", lambda: run_document_equations(
                work_dir, PPFormulaNetRecognizer(device=args.device))),
            ("Phase 7 figures", "figures", lambda: run_document_figures(
                work_dir, FigureExtractor(engine, RuleBasedChartExtractor()))),
        )
        for label, key, run in steps:
            reused[label] = present.get(key) is not None
            if not reused[label]:
                run()
        prepared = time.perf_counter() - started
        result = build_document(work_dir, visualize=args.visualize_order)
    except (PdfPipelineError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    document, summary = result["document"], result["document"]["summary"]
    counts = document["provenance"]["counts"]
    print(f"Pages: {summary['pages']}")
    print(f"Sections: {summary['sections']}")
    print(f"Subsections: {summary['subsections']}")
    print(f"Paragraphs: {summary['paragraphs']}")
    print(f"Tables: {summary['tables']}")
    print(f"Figures: {summary['figures']}")
    print(f"Charts: {summary['charts']}")
    print(f"Equations: {summary['equations']}")
    print()
    print("Layout:")
    for layout, count in summary["layout"].items():
        print(f"{layout.replace('_', '-')} pages: {count}")
    print()
    print("Schema valid: yes")
    print(f"Total reconstruction time: {result['timings']['total_s']:.2f}s")
    print()
    print(f"Title: {document['document']['title']}")
    for section in document["sections"]:
        name = " ".join(filter(None, [section["number"], section["title"]])) or f"({section['section_type']})"
        pages = f"p. {section['page_start']}" + ("" if section["page_end"] == section["page_start"]
                                                 else f"-{section['page_end']}")
        print(f"  {'  ' * (section['level'] - 1)}{name} [{section['section_type']}] {len(section['blocks'])} blocks, {pages}")
    print(f"Blocks: {summary['blocks']} ({summary['blocks_in_reading_order']} in reading order); "
          f"headings {summary['headings']}")
    print(f"Captions attached: {summary['captions_attached']} (standalone {summary['captions_standalone']}); "
          f"equation numbers attached: {summary['equation_numbers_attached']}")
    print(f"Paragraph parts joined: {summary['paragraphs_merged']}")
    print(f"References in text: {summary['references_found']} ({summary['references_resolved']} resolved)")
    print(f"Kept out of the reading order: {summary['headers']} headers, {summary['footers']} footers, "
          f"{summary['page_numbers']} page numbers")
    print(f"Information loss check: {counts['blocks']}/{counts['layout_regions']} regions are blocks; text "
          f"native {counts['native_text_blocks']}/{counts['native_text_regions']}, "
          f"OCR {counts['ocr_text_blocks']}/{counts['ocr_text_regions']}; tables {counts['tables']}/"
          f"{counts['tables_source']}, equations {counts['equations']}/{counts['equations_source']}, "
          f"figures {counts['figures']}/{counts['figures_source']}; problems {len(counts['problems'])}")
    print("; ".join(f"{label}: {'reused' if state else 'created'}" for label, state in reused.items()))
    print(f"Document: {result['document_path']}")
    if result["visualizations"]:
        print(f"Reading-order images: {os.path.dirname(result['visualizations'][0])} ({len(result['visualizations'])})")
    print(f"Time: earlier phases {prepared:.1f}s, reconstruction {result['timings']['total_s']:.2f}s")

    if args.truth:
        name = args.truth_name or os.path.splitext(os.path.basename(document["document"]["source_pdf"] or ""))[0]
        try:
            with open(args.truth, encoding="utf-8") as fh:
                truth = json.load(fh)["documents"][name]
        except (OSError, ValueError, KeyError) as exc:
            print(f"Error: cannot read ground truth {name!r} from {args.truth}: {exc}", file=sys.stderr)
            return 2
        print_evaluation(evaluate_document(truth, document))
    return 0


def print_evaluation(result):
    def share(value):
        return "n/a" if value is None else f"{value:.3f}"

    print()
    for row in result["rows"]:
        marks = [name for name, key in (("type", "type_match"), ("section", "section_match"), ("flow", "flow_match"))
                 if not row[key] and not (key == "section_match" and row["flow"] == "furniture")]
        if marks or row["block_id"] is None:
            print(f"  {row['key']} (page {row['page_number']}): "
                  + ("not found" if row["block_id"] is None else f"{row['block_id']} wrong " + ", ".join(marks)))
    print(f"Blocks: {result['found']} of {result['blocks']} found")
    print(f"Pairwise order accuracy: {share(result['pairwise_order_accuracy'])}")
    print(f"Pages in exact order: {result['pages_in_exact_order']}/{result['pages']}; "
          f"whole document: {'yes' if result['exact_order'] else 'no'}")
    print(f"Layout detected as expected: {result['layout_pages_correct']}/{result['pages']} pages")
    print(f"Block type accuracy: {share(result['type_accuracy'])}")
    print(f"Heading detection accuracy: {share(result['heading_detection_accuracy'])} "
          f"({result['headings_detected']}/{result['headings']} found, {result['false_headings']} false)")
    print(f"Section assignment accuracy: {share(result['section_assignment_accuracy'])}")
    print(f"Section hierarchy accuracy: {share(result['section_hierarchy_accuracy'])} ({result['sections']} sections)")
    print(f"Captions and equation numbers attached: {result['attached_correct']}/{result['attached']}")
    print(f"Headers and page numbers kept out: {result['furniture_correct']}/{result['furniture']}")
    print(f"Paragraph continuations joined: {result['continuations_joined']}/{result['continuations']}")


if __name__ == "__main__":
    sys.exit(main())
