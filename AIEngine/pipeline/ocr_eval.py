"""Small evaluation helpers for OCR text: edit counts, character and word error rate.

Engineering checks on fixtures with known text. Not a benchmark.
"""
from .layout_eval import match_regions


def edit_counts(reference, hypothesis):
    """Fewest edits that turn `reference` into `hypothesis` (two sequences), as
    (substitutions, deletions, insertions). A deletion is a reference item missing from the hypothesis."""
    # previous[j] = (edits, substitutions, deletions, insertions) for reference[:i] -> hypothesis[:j]
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, expected in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, found in enumerate(hypothesis, 1):
            diagonal, above, left = previous[j - 1], previous[j], current[j - 1]
            if expected == found:
                best = diagonal
            else:
                best = min((diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                           (above[0] + 1, above[1], above[2] + 1, above[3]),
                           (left[0] + 1, left[1], left[2], left[3] + 1))
            current.append(best)
        previous = current
    return previous[-1][1:]


def normalize_for_eval(text):
    """Text as it is compared: all runs of whitespace (line breaks included) become one space."""
    return " ".join(text.split())


def cer(reference, hypothesis):
    """Character error rate (S + D + I) / N, N = characters of the reference. Whitespace is normalized
    on both sides; case and punctuation count. Can exceed 1. Raises ValueError for an empty reference."""
    return _error_rate(list(normalize_for_eval(reference)), list(normalize_for_eval(hypothesis)))


def wer(reference, hypothesis):
    """Word error rate (S + D + I) / N over whitespace-separated words, N = words of the reference."""
    return _error_rate(reference.split(), hypothesis.split())


def compare_ocr_documents(reference, hypothesis, iou_threshold=0.5):
    """Compare the OCR text of one ocr.json with the text of another, region by region.

    Typical use: `hypothesis` is the ocr.json of a scanned copy, `reference` that of the digital
    original (native text). Each OCR'd region of the hypothesis is paired with the reference region in
    the same place (match_regions, PDF boxes). Returns

        {"rows": [{"region_id", "reference_region_id", "characters", "cer", "wer", "confidence"}],
         "regions", "unmatched", "characters", "cer", "wer"}

    where the totals are over all paired regions ((S + D + I) summed, divided by N summed) and are
    None when nothing could be paired. `unmatched` counts OCR'd regions without a reference text.
    """
    def regions(document, wanted):
        return [{**region, "type": region["canonical_type"]}
                for page in document["pages"] for region in page["regions"] if wanted(region)]

    predicted = regions(hypothesis, lambda r: r["text_source"] == "ocr")
    truth = regions(reference, lambda r: r["text"])
    by_box = {(r["page_number"], tuple(r["pdf_bbox"])): r for r in predicted}

    rows, char_edits, chars, word_edits, words = [], 0, 0, 0, 0
    for expected, match in zip(truth, match_regions(truth, predicted, iou_threshold)):
        if match["predicted_bbox"] is None:
            continue
        found = by_box[(match["page_number"], tuple(match["predicted_bbox"]))]
        ref_chars, hyp_chars = list(normalize_for_eval(expected["text"])), list(normalize_for_eval(found["text"]))
        ref_words, hyp_words = expected["text"].split(), found["text"].split()
        region_char_edits, region_word_edits = sum(edit_counts(ref_chars, hyp_chars)), sum(edit_counts(ref_words, hyp_words))
        char_edits, chars = char_edits + region_char_edits, chars + len(ref_chars)
        word_edits, words = word_edits + region_word_edits, words + len(ref_words)
        rows.append({
            "region_id": found["region_id"],
            "reference_region_id": expected["region_id"],
            "characters": len(ref_chars),
            "cer": region_char_edits / len(ref_chars),
            "wer": region_word_edits / len(ref_words),
            "confidence": found["confidence"],
        })
    return {
        "rows": rows,
        "regions": len(rows),
        "unmatched": len(predicted) - len(rows),
        "characters": chars,
        "cer": char_edits / chars if chars else None,
        "wer": word_edits / words if words else None,
    }


def _error_rate(reference, hypothesis):
    if not reference:
        raise ValueError("the reference text is empty; an error rate is not defined")
    return sum(edit_counts(reference, hypothesis)) / len(reference)
