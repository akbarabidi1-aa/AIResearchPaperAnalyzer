"""
Generates tiny SYNTHETIC test PDFs (original text, no copyrighted material).
Standard library only. Run from anywhere:

    python tests/fixtures/make_synthetic_fixtures.py

Writes:
    digital/synthetic_digital.pdf          2-page single-column paper-like PDF with sections + a table
    multicolumn/synthetic_two_column.pdf   1-page, two text columns
    scanned/synthetic_scanned.pdf          1-page, image only (no text layer)
"""
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))


def esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_at(x, y, s, size=11, bold=False):
    font = "/F2" if bold else "/F1"
    return f"BT {font} {size} Tf {x} {y} Td ({esc(s)}) Tj ET\n"


def build_pdf(path, page_streams, image=None):
    """page_streams: list of content-stream strings. image: (w, h, hex_gray) drawn on page 1 if given."""
    objs = []  # 1-indexed object bodies (bytes)

    def add(body):
        objs.append(body if isinstance(body, bytes) else body.encode("latin-1"))
        return len(objs)

    add("<< /Type /Catalog /Pages 2 0 R >>")                    # 1
    n_pages = len(page_streams)
    kids = " ".join(f"{5 + 2 * i} 0 R" for i in range(n_pages))
    add(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>")   # 2
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")        # 3
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")   # 4
    img_obj = None
    for i, stream in enumerate(page_streams):
        page_id = 5 + 2 * i
        content_id = page_id + 1
        xobj = ""
        if image and i == 0:
            img_obj = 5 + 2 * n_pages
            xobj = f"/XObject << /Im1 {img_obj} 0 R >>"
        add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> {xobj} >> /Contents {content_id} 0 R >>")
        data = stream.encode("latin-1")
        add(b"<< /Length " + str(len(data)).encode() + b" >>\nstream\n" + data + b"\nendstream")
    if image:
        w, h, hexdata = image
        data = (hexdata + ">").encode()
        add(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "
            b"/BitsPerComponent 8 /Filter /ASCIIHexDecode /Length %d >>\nstream\n" % (w, h, len(data))
            + data + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(out)
    print(f"wrote {path} ({len(out)} bytes)")


def paragraph(x, y, text, width=88, size=11, leading=14):
    s, lines = "", textwrap.wrap(text, width)
    for line in lines:
        s += text_at(x, y, line, size)
        y -= leading
    return s, y


P_INTRO = ("Neural networks are widely used for image classification and language modelling. "
           "In this synthetic study we investigate how network depth affects classification accuracy. "
           "Deeper networks often improve accuracy but increase training cost and memory consumption. "
           "We evaluate several architectures on a small benchmark dataset and compare their accuracy.")
P_METHOD = ("Our method trains convolutional networks with different depths using stochastic gradient descent. "
            "Each network is trained for fifty epochs and evaluated on a held out validation set. "
            "We record accuracy, training time and memory usage for every architecture in the experiment.")
P_RESULTS = ("The deeper network achieved higher accuracy than the shallow network on the validation set. "
             "However training time increased substantially as network depth increased. "
             "These results suggest that moderate depth gives the best trade off between accuracy and cost.")
P_CONCL = ("We conclude that network depth improves accuracy with diminishing returns. "
           "Future work will study regularisation and data augmentation for deep networks. "
           "This synthetic document exists only to test the analysis pipeline.")


def digital():
    p1 = text_at(72, 740, "Effect of Network Depth on Accuracy", 16, True)
    p1 += text_at(72, 720, "Abstract", 12, True)
    s, y = paragraph(72, 704, P_INTRO)
    p1 += s
    p1 += text_at(72, y - 10, "1. Introduction", 12, True)
    s, y = paragraph(72, y - 26, P_INTRO + " " + P_INTRO)
    p1 += s
    p1 += text_at(72, y - 10, "2. Methodology", 12, True)
    s, y = paragraph(72, y - 26, P_METHOD)
    p1 += s

    p2 = text_at(72, 740, "3. Results", 12, True)
    s, y = paragraph(72, 724, P_RESULTS)
    p2 += s
    # Table: each cell positioned separately, as in a real typeset table
    y -= 12
    rows = [("Model", "Depth", "Accuracy", "Time"),
            ("Net-A", "8", "81.2", "12"),
            ("Net-B", "16", "86.5", "25"),
            ("Net-C", "32", "88.1", "61")]
    for r in rows:
        for cx, cell in zip((72, 180, 290, 400), r):
            p2 += text_at(cx, y, cell, 11, r is rows[0])
        y -= 16
    p2 += text_at(72, y - 8, "Table 1. Accuracy and training time by depth.", 10)
    p2 += text_at(72, y - 40, "4. Discussion", 12, True)
    s, y2 = paragraph(72, y - 56, P_RESULTS)
    p2 += s
    p2 += text_at(72, y2 - 10, "5. Conclusion", 12, True)
    s, _ = paragraph(72, y2 - 26, P_CONCL)
    p2 += s
    build_pdf(os.path.join(HERE, "digital", "synthetic_digital.pdf"), [p1, p2])


def multicolumn():
    s = text_at(72, 750, "Two Column Layout Test", 16, True)
    left = ("LEFT-1 The first column begins here and continues for several lines of ordinary prose text. "
            "LEFT-2 It describes the background of the problem and the motivation for the study. "
            "LEFT-3 This sentence is the last sentence of the left column in this test document.")
    right = ("RIGHT-1 The second column starts here and should be read after the whole left column. "
             "RIGHT-2 It describes the experimental setup and the evaluation procedure that was used. "
             "RIGHT-3 This sentence is the last sentence of the right column in this test document.")
    a, _ = paragraph(72, 710, left, width=38)
    b, _ = paragraph(320, 710, right, width=38)
    build_pdf(os.path.join(HERE, "multicolumn", "synthetic_two_column.pdf"), [s + a + b])


def scanned():
    w = h = 64
    px = []
    for yy in range(h):
        for xx in range(w):
            px.append(0 if (xx // 8 + yy // 8) % 2 == 0 else 220)
    hexdata = "".join(f"{v:02x}" for v in px)
    stream = "q 468 0 0 600 72 100 cm /Im1 Do Q\n"   # image only, no text objects
    build_pdf(os.path.join(HERE, "scanned", "synthetic_scanned.pdf"), [stream], image=(w, h, hexdata))


if __name__ == "__main__":
    digital()
    multicolumn()
    scanned()
