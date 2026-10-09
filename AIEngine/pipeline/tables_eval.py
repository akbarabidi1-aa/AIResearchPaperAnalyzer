"""Small evaluation helpers for extracted tables: row/column counts, cell text, structure.

Engineering checks on fixtures with known content. Not a benchmark.
"""
from .layout_eval import match_regions
from .ocr_eval import edit_counts, normalize_for_eval
from .tables import table_grid


def evaluate_table(expected, predicted):
    """Compare one extracted table (an entry of tables.json, or None when the table was not found)
    with its ground truth.

    `expected` has "grid" (all rows, header rows included; the positions covered by a merged cell hold
    ""), and optionally "merged_cells" ([{"row", "column", "row_span", "col_span"}]), "header_rows" and
    "caption". Cells are compared position by position after whitespace normalization; nothing else
    is normalized, so "94.7" does not match "94.70". Returns

        expected_rows, predicted_rows, rows_match, expected_columns, predicted_columns, columns_match
        total_cells          positions of the expected grid
        exact_cells          positions whose text is identical
        exact_cell_accuracy  exact_cells / total_cells
        cell_text_accuracy   mean over the positions of 1 - min(1, CER) (1 for two empty cells)
        structure_accuracy   cells with the same row, column, row_span and col_span in both tables,
                             divided by the cells that occur in either (text is ignored)
        header_rows_match, caption_match     None when the ground truth does not say
    """
    grid = [[normalize_for_eval(text) for text in row] for row in expected["grid"]]
    n_rows, n_columns = len(grid), len(grid[0]) if grid else 0
    found = predicted is not None and "error" not in predicted
    predicted_grid = [[normalize_for_eval(text) for text in row] for row in table_grid(predicted)] if found else []
    predicted_rows = len(predicted_grid)
    predicted_columns = len(predicted_grid[0]) if predicted_grid else 0

    exact, similarity = 0, 0.0
    for r, row in enumerate(grid):
        for c, text in enumerate(row):
            # A position missing from an extracted table is an empty cell; a missing table matches nothing.
            got = predicted_grid[r][c] if r < predicted_rows and c < predicted_columns else ("" if found else None)
            if got == text:
                exact, similarity = exact + 1, similarity + 1.0
            elif got and text:
                similarity += 1.0 - min(1.0, sum(edit_counts(text, got)) / len(text))
    total = n_rows * n_columns

    merged = {(m["row"], m["column"]): (m["row_span"], m["col_span"]) for m in expected.get("merged_cells", [])}
    covered = {(r + i, c + j) for (r, c), (rs, cs) in merged.items() for i in range(rs) for j in range(cs)}
    expected_cells = {(r, c, *merged.get((r, c), (1, 1))) for r in range(n_rows) for c in range(n_columns)
                      if (r, c) in merged or (r, c) not in covered}
    predicted_cells = ({(cell["row"], cell["column"], cell["row_span"], cell["col_span"]) for cell in predicted["cells"]}
                       if found else set())
    union = expected_cells | predicted_cells

    header_rows_match = caption_match = None
    if "header_rows" in expected:
        header_rows_match = found and predicted["header_rows"] == expected["header_rows"]
    if "caption" in expected:
        caption_match = found and predicted["caption"] == expected["caption"]
    return {
        "expected_rows": n_rows, "predicted_rows": predicted_rows, "rows_match": n_rows == predicted_rows,
        "expected_columns": n_columns, "predicted_columns": predicted_columns,
        "columns_match": n_columns == predicted_columns,
        "total_cells": total, "exact_cells": exact,
        "exact_cell_accuracy": exact / total if total else None,
        "cell_text_accuracy": similarity / total if total else None,
        "structure_accuracy": len(expected_cells & predicted_cells) / len(union) if union else None,
        "header_rows_match": header_rows_match,
        "caption_match": caption_match,
    }


def evaluate_tables(truth_tables, document, iou_threshold=0.5):
    """Compare a tables.json object with ground-truth tables ([{"page_number", "pdf_bbox", "grid", ...}]).

    Each ground-truth table is paired with the extracted table at the same place on the same page.
    Returns {"rows": [evaluate_table result + "page_number", "table_id", "source"], "tables", "found",
    "row_count_accuracy", "column_count_accuracy", "total_cells", "exact_cells", "exact_cell_accuracy",
    "cell_text_accuracy", "structure_accuracy", "header_accuracy", "caption_accuracy"}. The accuracies
    are over all ground-truth tables; a table that was not found counts with zero correct cells.
    """
    predicted = [{**table, "type": "TABLE"} for page in document["pages"] for table in page["tables"]]
    truth = [{**table, "type": "TABLE"} for table in truth_tables]
    by_box = {(table["page_number"], tuple(table["pdf_bbox"])): table for table in predicted}

    rows = []
    for expected, match in zip(truth, match_regions(truth, predicted, iou_threshold)):
        table = by_box.get((match["page_number"], tuple(match["predicted_bbox"] or ())))
        rows.append({"page_number": expected["page_number"], "table_id": table and table["table_id"],
                     "source": table and table["source"], **evaluate_table(expected, table)})

    def share(key):
        values = [row[key] for row in rows if row[key] is not None]
        return sum(values) / len(values) if values else None

    total, exact = sum(row["total_cells"] for row in rows), sum(row["exact_cells"] for row in rows)
    return {
        "rows": rows,
        "tables": len(rows),
        "found": sum(1 for row in rows if row["source"]),
        "row_count_accuracy": share("rows_match"),
        "column_count_accuracy": share("columns_match"),
        "total_cells": total,
        "exact_cells": exact,
        "exact_cell_accuracy": exact / total if total else None,
        "cell_text_accuracy": (sum(row["cell_text_accuracy"] * row["total_cells"] for row in rows) / total
                               if total else None),
        "structure_accuracy": share("structure_accuracy"),
        "header_accuracy": share("header_rows_match"),
        "caption_accuracy": share("caption_match"),
    }
