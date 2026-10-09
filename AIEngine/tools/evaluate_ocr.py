"""Compare the OCR text of one ocr.json with the text of a reference ocr.json: CER and WER.

    python AIEngine/tools/evaluate_ocr.py temp/scan --reference temp/digital

Both arguments are working directories (or ocr.json files). Typical use: the first is a scanned copy
of a document, the reference is the digital original, whose regions carry native text. Regions are
paired by position. An engineering check on fixtures with known text, not a benchmark.
Exit code: 0 = ok, 2 = a file could not be read or nothing could be compared.
"""
import argparse
import json
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

from pipeline import compare_ocr_documents, validate_ocr_document  # noqa: E402


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "metadata", "ocr.json")
    with open(path, encoding="utf-8") as fh:
        document = json.load(fh)
    problems = validate_ocr_document(document)
    if problems:
        raise ValueError(f"Invalid {path}: " + "; ".join(problems[:5]))
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description="CER / WER of OCR text against a reference (Phase 4).")
    parser.add_argument("ocr", help="working directory or ocr.json with OCR'd regions")
    parser.add_argument("--reference", required=True, help="working directory or ocr.json with the correct text")
    args = parser.parse_args(argv)

    try:
        result = compare_ocr_documents(load(args.reference), load(args.ocr))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if not result["rows"]:
        print("Error: no OCR'd region lies on a reference region with text", file=sys.stderr)
        return 2

    print(f"{'region':<12}{'reference':<12}{'chars':>6}{'CER':>8}{'WER':>8}{'conf':>7}")
    for row in result["rows"]:
        print(f"{row['region_id']:<12}{row['reference_region_id']:<12}{row['characters']:>6}"
              f"{row['cer']:>8.4f}{row['wer']:>8.4f}{row['confidence']:>7.3f}")
    print()
    print(f"Regions compared: {result['regions']}")
    print(f"OCR regions without a reference: {result['unmatched']}")
    print(f"Reference characters: {result['characters']}")
    print(f"CER: {result['cer']:.4f}")
    print(f"WER: {result['wer']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
