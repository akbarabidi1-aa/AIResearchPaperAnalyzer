import os
import json
import re
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams


def extract_pdf_content(file_path):
    """
    Extracts raw text, tables (heuristic), and logical flow steps from a PDF.
    Returns a dict with keys: raw_text, tables, flow_steps
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # --- 1. Extract raw text ---
    params = LAParams(line_margin=0.5, char_margin=2.0, detect_vertical=True)
    raw_text = extract_text(file_path, laparams=params)

    # Clean up whitespace
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    # --- 2. Heuristic table extraction ---
    # Look for lines with multiple tab/space-separated columns (≥3 tokens)
    tables = []
    current_table = []
    for line in lines:
        parts = re.split(r'\s{2,}|\t', line)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 3:
            current_table.append(parts)
        else:
            if len(current_table) >= 2:
                tables.append(current_table)
            current_table = []
    if len(current_table) >= 2:
        tables.append(current_table)

    # --- 3. Heuristic flow / section extraction ---
    # Pull out numbered or titled sections as "flow steps"
    flow_steps = []
    section_pattern = re.compile(
        r'^(\d+[\.\)]\s+[A-Z].{3,}|'      # "1. Introduction"
        r'[IVXLC]+\.\s+[A-Z].{3,}|'       # "II. Related Work"
        r'(Abstract|Introduction|Method|Results?|Discussion|Conclusion).{0,30})$',
        re.IGNORECASE
    )
    for line in lines:
        if section_pattern.match(line) and len(line) < 80:
            flow_steps.append(line.strip())

    return {
        "raw_text": clean_text,
        "tables": tables,
        "flow_steps": flow_steps[:10]  # cap to 10 steps
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "my_paper.pdf"
    result = extract_pdf_content(path)
    print(f"Flow Steps: {result['flow_steps']}")
    print(f"Tables found: {len(result['tables'])}")
    print(f"Text length: {len(result['raw_text'])} chars")
