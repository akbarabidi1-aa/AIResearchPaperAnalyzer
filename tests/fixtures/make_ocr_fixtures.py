"""
Generates the SYNTHETIC fixtures of Phase 4 (OCR). Needs PyMuPDF. Run from anywhere:

    python tests/fixtures/make_ocr_fixtures.py

Writes:
    scanned/synthetic_scanned_text.pdf   digital/synthetic_digital.pdf with every page turned into one
                                         grayscale bitmap (150 DPI): same content, no text layer
    mixed/synthetic_mixed.pdf            1 page: native heading and paragraphs, and one paragraph that
                                         exists only as an embedded bitmap
    mixed/synthetic_mixed.json           the known text and position of that bitmap paragraph
"""
import json
import os

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))

HEADING = "1. Mixed Content Page"
NATIVE_ABOVE = ("This first paragraph is ordinary selectable text stored in the document. It describes a page "
                "that mixes two kinds of content, and it must be read from the text layer without any "
                "recognition step.")
IMAGE_TEXT = ("This second paragraph was pasted into the page as a picture. Its words are visible to a "
              "reader but absent from the text layer, so the pipeline has to recognize them from the "
              "pixels of the rendered page.")
NATIVE_BELOW = ("This third paragraph is selectable text again. It follows the picture and shows that the "
                "routing decision is taken for every region and not once for the whole page.")
IMAGE_RECT = (72.0, 190.0, 540.0, 240.0)      # PDF points, origin top-left


def scanned_text():
    source_path = os.path.join(HERE, "digital", "synthetic_digital.pdf")
    target = os.path.join(HERE, "scanned", "synthetic_scanned_text.pdf")
    with pymupdf.open(source_path) as source, pymupdf.open() as out:
        for page in source:
            pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
            out.new_page(width=page.rect.width, height=page.rect.height).insert_image(page.rect, pixmap=pix)
        save(out, target)


def mixed():
    target = os.path.join(HERE, "mixed", "synthetic_mixed.pdf")
    rect = pymupdf.Rect(IMAGE_RECT)
    # The paragraph is typeset on a scratch page and only its bitmap goes into the fixture.
    with pymupdf.open() as scratch:
        page = scratch.new_page()
        page.insert_textbox(rect, IMAGE_TEXT, fontsize=11)
        bitmap = page.get_pixmap(dpi=200, clip=rect, colorspace=pymupdf.csGRAY)

    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 90), HEADING, fontsize=14, fontname="hebo")
        page.insert_textbox(pymupdf.Rect(72, 110, 540, 165), NATIVE_ABOVE, fontsize=11)
        page.insert_image(rect, pixmap=bitmap)
        page.insert_textbox(pymupdf.Rect(72, 265, 540, 320), NATIVE_BELOW, fontsize=11)
        save(doc, target)

    truth = {"heading": HEADING, "native_above": NATIVE_ABOVE, "image_text": IMAGE_TEXT,
             "native_below": NATIVE_BELOW, "image_pdf_bbox": list(IMAGE_RECT)}
    with open(os.path.join(HERE, "mixed", "synthetic_mixed.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(truth, fh, indent=2)
        fh.write("\n")


def save(doc, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    doc.save(target, garbage=4, deflate=True, no_new_id=True)
    print(f"wrote {target} ({os.path.getsize(target)} bytes)")


if __name__ == "__main__":
    scanned_text()
    mixed()
