import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extraction import extract_pdf_content
from preprocess import clean_text
from keyword_extractor import extract_keywords
from summarizer import summarize_large_text


def run_analysis(file_path):
    try:
        # 1. Extract
        extracted = extract_pdf_content(file_path)
        raw_text  = extracted["raw_text"]
        tables    = extracted["tables"]
        flow_steps = extracted["flow_steps"]

        if not raw_text.strip():
            print(json.dumps({"status": "error", "message": "No text found in PDF. Make sure it is not a scanned image."}))
            return

        # 2. Preprocess
        tokens = clean_text(raw_text)
        preprocessed = " ".join(tokens)

        # 3. Keywords
        keywords = extract_keywords(preprocessed, top_n=10)

        # 4. Summary (fast, no ML)
        summary = summarize_large_text(raw_text)

        # 5. Key points
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary) if len(s.strip()) > 30]
        important_points = sentences[:5]

        result = {
            "status":           "success",
            "file_name":        os.path.basename(file_path),
            "summary":          summary,
            "keywords":         keywords,
            "important_points": important_points,
            "flow":             flow_steps if flow_steps else ["Introduction", "Methodology", "Results", "Discussion", "Conclusion"],
            "tables":           tables[:3]
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Usage: python ai_engine.py <path>"}))
        sys.exit(1)
    run_analysis(sys.argv[1])