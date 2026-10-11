"""Small evaluation helpers for a reconstructed document: reading order, headings, sections, attached
captions, page furniture and joined paragraphs, against a fixture whose blocks are known in order.

Engineering checks on synthetic documents. Not a benchmark.
"""
from .document_builder import FLOW_ATTACHED, FLOW_FURNITURE, FLOW_MAIN, FLOW_MERGED
from .layout_eval import bbox_area
from .sections import TYPE_HEADING

BLOCK_MATCH_MIN = 0.5       # share of a ground-truth box that must lie inside the block it is matched to


def match_blocks(truth_blocks, document):
    """Pair each ground-truth block with the document block on the same page that covers most of it
    (at least BLOCK_MATCH_MIN of its area). Returns {truth key: block or None}. Two ground-truth
    blocks can get the same document block when the layout model drew one region around both."""
    matched = {}
    for expected in truth_blocks:
        box, best = expected["pdf_bbox"], (BLOCK_MATCH_MIN, None)
        for block in document["blocks"]:
            if block["page_number"] != expected["page_number"]:
                continue
            width = min(box[2], block["bbox"][2]) - max(box[0], block["bbox"][0])
            height = min(box[3], block["bbox"][3]) - max(box[1], block["bbox"][1])
            share = width * height / bbox_area(box) if width > 0 and height > 0 else 0.0
            if share >= best[0]:
                best = (share, block)
        matched[expected["key"]] = best[1]
    return matched


def _positions(keys, matched, by_id):
    """Reading position of each key: the order of its block. The later part of a joined paragraph has
    no order of its own (it is read with its beginning); it is placed where it stands on its page,
    directly before the next block read on that page, so that the expected order of a fixture can be
    the order on the pages. None for a block that is not in the reading order."""
    positions = {}
    for key in keys:
        block = matched[key]
        if block is not None and block["flow"] == FLOW_MERGED:
            head = by_id[block["merged_into"]]
            later = [b["order"] for b in by_id.values() if b["flow"] == FLOW_MAIN and b["order"] > head["order"]
                     and b["page_number"] == block["page_number"]]
            positions[key] = (min(later) if later else len(by_id)) - 0.5
        elif block is not None and block["flow"] == FLOW_MAIN:
            positions[key] = block["order"]
        else:
            positions[key] = None
    return positions


def _pairwise(keys, positions, matched):
    """(correct pairs, pairs) of `keys` (in the expected order). Two keys matched to one block are in
    order by definition: the block holds both."""
    pairs = correct = 0
    for i, first in enumerate(keys):
        for second in keys[i + 1:]:
            pairs += 1
            a, b = positions[first], positions[second]
            if a is None or b is None:
                continue
            correct += a < b or (a == b and matched[first] is matched[second])
    return correct, pairs


def evaluate_document(truth, document):
    """Compare a document.json object with the ground truth of a fixture ({"blocks": [...] in reading
    order, "sections": [...], "columns": n}).

    Returns {"rows": one per ground-truth block, and}
        blocks, found
        pairwise_order_accuracy          over all pairs of main-flow blocks of the document
        pages, pages_in_exact_order      pages whose main-flow blocks are read exactly in the expected order
        exact_order                      the whole document is
        type_accuracy                    canonical block type equal
        headings, headings_detected, false_headings, heading_detection_accuracy
                                         expected headings that are HEADING blocks; blocks that are HEADING
                                         without being one; detected / (expected + false)
        section_assignment_accuracy      blocks whose section is the expected one (same number, or same
                                         title and type when the section has no number)
        sections, section_hierarchy_accuracy
                                         expected sections found with the right level and the right parent
        attached, attached_correct       captions and equation numbers taken out of the reading order and
                                         attached to the right block
        furniture, furniture_correct     headers and page numbers kept out of the reading order
        continuations, continuations_joined
        layout_pages_correct             pages whose detected layout has the expected number of columns
    A rate without anything to count is None.
    """
    by_id = {block["block_id"]: block for block in document["blocks"]}
    sections = {section["section_id"]: section for section in document["sections"]}
    truth_sections = {section["key"]: section for section in truth["sections"]}
    matched = match_blocks(truth["blocks"], document)

    def same_section(expected, found):
        if found is None:
            return False
        if expected["number"]:
            return found["number"] == expected["number"]
        return found["section_type"] == expected["section_type"] and (
            expected["title"] is None or found["title"] is None or found["title"] == expected["title"])

    main = [b for b in truth["blocks"] if b["flow"] == FLOW_MAIN]
    keys = [b["key"] for b in main]
    positions = _positions(keys, matched, by_id)
    correct, pairs = _pairwise(keys, positions, matched)
    page_numbers = sorted({b["page_number"] for b in truth["blocks"]})
    exact_pages = 0
    for number in page_numbers:
        page_keys = [b["key"] for b in main if b["page_number"] == number]
        page_correct, page_pairs = _pairwise(page_keys, positions, matched)
        exact_pages += page_correct == page_pairs and all(positions[key] is not None for key in page_keys)

    rows = []
    for expected in truth["blocks"]:
        block = matched[expected["key"]]
        section = sections.get(block["section_id"]) if block else None
        owner = matched.get(expected.get("attached_to"))
        rows.append({
            "key": expected["key"], "page_number": expected["page_number"], "flow": expected["flow"],
            "block_id": block and block["block_id"], "expected_type": expected["type"],
            "type": block and block["type"], "order": positions.get(expected["key"]),
            "type_match": bool(block) and block["type"] == expected["type"],
            "section_match": (expected["flow"] != FLOW_FURNITURE and bool(block)
                              and same_section(truth_sections[expected["section"]], section)),
            "flow_match": bool(block) and (block["flow"] == expected["flow"] or (
                expected["flow"] == FLOW_MAIN and block["flow"] == FLOW_MERGED)),
            "attached_match": (expected["flow"] == FLOW_ATTACHED and bool(block) and bool(owner)
                               and block.get("attached_to") == owner["block_id"]),
        })

    headings = [row for row in rows if row["expected_type"] == TYPE_HEADING]
    detected = sum(1 for row in headings if row["type"] == TYPE_HEADING)
    heading_ids = {row["block_id"] for row in headings}
    false_headings = sum(1 for block in document["blocks"]
                         if block["type"] == TYPE_HEADING and block["block_id"] not in heading_ids)

    hierarchy = 0
    expected_sections = [s for s in truth["sections"] if s["section_type"] != "front_matter"]
    for expected in expected_sections:
        found = next((s for s in document["sections"] if same_section(expected, s)
                      and s["section_type"] != "front_matter"), None)
        parent = truth_sections.get(expected["parent"])
        found_parent = sections.get(found["parent_id"]) if found and found["parent_id"] else None
        hierarchy += bool(found) and found["level"] == expected["level"] and (
            (parent is None and found_parent is None) or (parent is not None and same_section(parent, found_parent)))

    continued = [b for b in truth["blocks"] if b.get("continues")]
    joined = sum(1 for b in continued if matched[b["key"]] is not None and matched[b["continues"]] is not None
                 and (matched[b["key"]].get("merged_into") == matched[b["continues"]]["block_id"]
                      or matched[b["key"]] is matched[b["continues"]]))
    expected_layout = {1: "single_column", 2: "two_column"}.get(truth.get("columns"), "multi_column")

    def rate(count, total):
        return count / total if total else None

    def select(flow):
        return [row for row in rows if row["flow"] == flow]

    sectioned = [row for row in rows if row["flow"] != FLOW_FURNITURE]
    return {
        "rows": rows,
        "blocks": len(rows),
        "found": sum(1 for row in rows if row["block_id"]),
        "pairwise_order_accuracy": rate(correct, pairs),
        "pages": len(page_numbers),
        "pages_in_exact_order": exact_pages,
        "exact_order": correct == pairs and all(positions[key] is not None for key in keys),
        "type_accuracy": rate(sum(row["type_match"] for row in rows), len(rows)),
        "headings": len(headings),
        "headings_detected": detected,
        "false_headings": false_headings,
        "heading_detection_accuracy": rate(detected, len(headings) + false_headings),
        "section_assignment_accuracy": rate(sum(row["section_match"] for row in sectioned), len(sectioned)),
        "sections": len(expected_sections),
        "section_hierarchy_accuracy": rate(hierarchy, len(expected_sections)),
        "attached": len(select(FLOW_ATTACHED)),
        "attached_correct": sum(row["attached_match"] for row in select(FLOW_ATTACHED)),
        "furniture": len(select(FLOW_FURNITURE)),
        "furniture_correct": sum(row["flow_match"] for row in select(FLOW_FURNITURE)),
        "continuations": len(continued),
        "continuations_joined": joined,
        "layout_pages_correct": sum(1 for page in document["pages"] if page["layout"] == expected_layout),
    }
