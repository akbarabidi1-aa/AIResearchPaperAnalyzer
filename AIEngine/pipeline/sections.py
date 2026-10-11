"""Phase 8 sections: the ordered blocks of a document -> headings -> section tree -> a section for every block.

A block is a heading on evidence from three sides, never from a list of names alone:

    label       the layout model called it a title ("paragraph_title")
    numbering   "3 Methodology", "3.1 Architecture", "IV. EXPERIMENTS", "A. Datasets", "Appendix B"
    form        one short line, no full stop at its end, not set smaller than the body text

The level of a section comes from its number ("3.1" is level 2 under "3"). Without a number a level
below 1 is assigned only when the heading is visibly set smaller than the top-level headings.
Nothing is renamed, reordered or summarized.
"""
import re
import statistics

TYPE_TITLE = "TITLE"
TYPE_HEADING = "HEADING"
TYPE_PARAGRAPH = "PARAGRAPH"

SECTION_FRONT = "front_matter"
SECTION_ABSTRACT = "abstract"
SECTION_BODY = "body"
SECTION_REFERENCES = "references"
SECTION_ACKNOWLEDGMENTS = "acknowledgments"
SECTION_APPENDIX = "appendix"
SECTION_TYPES = (SECTION_FRONT, SECTION_ABSTRACT, SECTION_BODY, SECTION_REFERENCES, SECTION_ACKNOWLEDGMENTS,
                 SECTION_APPENDIX)

FRONT_MATTER_ID = "sec_front_matter"
HEADING_LABELS = frozenset({"paragraph_title"})
ABSTRACT_LABELS = frozenset({"abstract"})
REFERENCE_LABELS = frozenset({"reference", "reference_content"})

MAX_HEADING_CHARACTERS = 120    # a "title" region longer than this is running text
MAX_PROMOTED_WORDS = 12         # an unlabelled line longer than this is not promoted to a heading
SMALLER_HEADING = 0.92          # an unnumbered heading this much smaller than the top-level ones is a subsection

_NUMERIC = re.compile(r"(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(\S.*)")
_ROMAN = re.compile(r"([IVX]{1,5})\.\s+(\S.*)")
_LETTER = re.compile(r"([A-Z])\.\s+(\S.*)")
_APPENDIX = re.compile(r"(?:Appendix|APPENDIX)\s+([A-Z])(?:\s*[:.–—-]\s*|\s+|$)(.*)")
_SPECIAL_NAMES = (
    (re.compile(r"abstract", re.I), SECTION_ABSTRACT),
    (re.compile(r"references?|bibliography|literature cited|works cited", re.I), SECTION_REFERENCES),
    (re.compile(r"acknowledge?ments?", re.I), SECTION_ACKNOWLEDGMENTS),
    (re.compile(r"appendi(?:x|ces)|supplementary materials?", re.I), SECTION_APPENDIX),
)


def parse_heading(text):
    """Number and title of a heading line: {"number", "title", "numbering"}.

        "3 Methodology"        -> number "3",   numbering "numeric"
        "3.1 Architecture"     -> number "3.1", numbering "numeric"
        "IV. EXPERIMENTS"      -> number "IV",  numbering "roman"
        "A. Datasets"          -> number "A",   numbering "letter"
        "Appendix B: Proofs"   -> number "B",   numbering "appendix"
        "Related Work"         -> number None,  numbering None
    A number counts only when a capitalized title follows it ("3 Methodology", not "3 of the 5 runs").
    """
    text = " ".join((text or "").split())
    match = _APPENDIX.fullmatch(text)
    if match:
        return {"number": match.group(1), "title": match.group(2).strip() or None, "numbering": "appendix"}
    for pattern, numbering in ((_NUMERIC, "numeric"), (_ROMAN, "roman"), (_LETTER, "letter")):
        match = pattern.fullmatch(text)
        if match and match.group(2)[0].isupper():
            return {"number": match.group(1), "title": match.group(2).strip(), "numbering": numbering}
    return {"number": None, "title": text or None, "numbering": None}


def special_section_type(title):
    """The special section a heading title names ("Abstract", "References", "Acknowledgments",
    "Appendix"), or None. The title must be that name and nothing else."""
    cleaned = (title or "").strip().rstrip(".:").strip()
    for pattern, section_type in _SPECIAL_NAMES:
        if pattern.fullmatch(cleaned):
            return section_type
    return None


def looks_like_heading(block, body_height=None):
    """True for a block the layout model did NOT label as a title but that is one by form and
    numbering: a single short line without a full stop at its end, not set smaller than the body
    text, that either carries a section number or is the name of a special section."""
    text = " ".join((block.get("text") or "").split())
    if not text or block.get("line_count", 1) != 1 or len(text.split()) > MAX_PROMOTED_WORDS:
        return False
    if text[-1] in ".,;:!?" and special_section_type(text) is None:
        return False
    height = block.get("line_height")
    if height and body_height and height < 0.98 * body_height:
        return False
    parsed = parse_heading(text)
    return parsed["numbering"] in ("numeric", "roman", "appendix") or special_section_type(text) is not None


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:40] or "untitled"


class _Tree:
    """The sections found so far, and where a new one hangs."""

    def __init__(self):
        self.sections, self.stack, self.ids = [], [], set()
        self.roman = self.in_appendix = False
        self.top_heights = []

    def add(self, section_id, title, number, level, section_type, block=None, implicit=False, parent_id=None):
        base, count = section_id, 1
        while section_id in self.ids:
            count += 1
            section_id = f"{base}_{count}"
        self.ids.add(section_id)
        section = {"section_id": section_id, "title": title, "number": number, "level": level,
                   "parent_id": parent_id, "section_type": section_type,
                   "heading_block_id": block["block_id"] if block else None, "implicit": implicit,
                   "page_start": None, "page_end": None, "blocks": []}
        self.sections.append(section)
        self.stack = [s for s in self.stack if s["level"] < level] + [section]
        return section

    def parent_of(self, level, number=None, numbering=None):
        """The section a new section of `level` hangs under, or None. A numbered subsection hangs only
        under the section whose number it extends ("3.1" under "3"); if that section was not found
        there is no parent, rather than a wrong one."""
        candidates = [s for s in self.stack if s["level"] < level]
        if not candidates or level == 1:
            return None
        if numbering == "numeric":
            prefix = number.rsplit(".", 1)[0]
            return next((s["section_id"] for s in reversed(candidates) if s["number"] == prefix), None)
        return next((s["section_id"] for s in reversed(candidates) if s["level"] == level - 1), None)

    def heading(self, block):
        parsed = parse_heading(block["text"])
        number, title, numbering = parsed["number"], parsed["title"], parsed["numbering"]
        special = special_section_type(title if numbering else block["text"])
        height = block.get("line_height")
        if numbering == "letter" and not (self.roman or self.in_appendix):
            number, title, numbering = None, " ".join(block["text"].split()), None     # "A. Smith ..." is no number
        if numbering == "roman":
            self.roman = True

        if numbering == "numeric":
            level = number.count(".") + 1
        elif numbering == "letter" and self.roman and not self.in_appendix:
            level = 2
        elif numbering is None and special is None and height and self.top_heights and self.stack \
                and height <= SMALLER_HEADING * statistics.median(self.top_heights):
            level = min(s["level"] for s in self.stack) + 1
        else:
            level = 1
        if level == 1 and height:
            self.top_heights.append(height)

        if special == SECTION_APPENDIX or numbering == "appendix":
            self.in_appendix = True
        if special:
            section_type = special
        elif self.in_appendix:
            section_type = SECTION_APPENDIX
        else:
            section_type = SECTION_BODY

        if numbering in ("numeric", "roman", "letter", "appendix"):
            prefix = "sec_appendix_" if numbering == "appendix" or (numbering == "letter" and level == 1) else "sec_"
            parent = self.parent_of(level, number, numbering)
            if numbering == "letter" and level == 2 and parent:
                section_id = f"{parent}_{number.lower()}"
            else:
                section_id = prefix + number.lower().replace(".", "_")
        else:
            parent = self.parent_of(level)
            section_id = "sec_" + (special if special else _slug(title))
        return self.add(section_id, title, number, level, section_type, block, parent_id=parent)


def build_sections(blocks):
    """Find the headings among `blocks` (the main-flow blocks of a document in reading order) and
    give every block a section. Returns (sections, document title).

    Each block is a dict with "block_id", "type", "text", "page_number", "source_label" and, where
    known, "line_height" and "line_count". The function sets "section_id" on every block and changes
    "type" in two cases: a PARAGRAPH that looks_like_heading becomes a HEADING, and a HEADING longer
    than MAX_HEADING_CHARACTERS becomes a PARAGRAPH.

    Sections, in document order: {"section_id", "title", "number", "level", "parent_id",
    "section_type", "heading_block_id", "implicit", "page_start", "page_end", "blocks": [block ids]}.
      - Everything before the first heading is the front matter (title, authors).
      - A heading opens a section; its level and parent come from its number (see parse_heading).
      - A block the layout model labelled "abstract", or as a reference entry, opens an implicit
        section of that type when no such section is open: "title" is null and "implicit" true,
        because no heading was read.
    The document title is the text of the first TITLE block, or None.
    """
    heights = [b["line_height"] for b in blocks if b["type"] == TYPE_PARAGRAPH and b.get("line_height")
               and b.get("line_count", 1) > 1]
    body_height = statistics.median(heights) if heights else None
    tree = _Tree()
    current = tree.add(FRONT_MATTER_ID, None, None, 1, SECTION_FRONT)
    title = None
    for block in blocks:
        if block["type"] == TYPE_HEADING and len(block.get("text") or "") > MAX_HEADING_CHARACTERS:
            block["type"] = TYPE_PARAGRAPH
        elif block["type"] == TYPE_PARAGRAPH and block.get("source_label") not in REFERENCE_LABELS | ABSTRACT_LABELS \
                and looks_like_heading(block, body_height):
            block["type"] = TYPE_HEADING

        label = block.get("source_label")
        if block["type"] == TYPE_TITLE and title is None:
            title = " ".join((block.get("text") or "").split()) or None
        if block["type"] == TYPE_HEADING and (block.get("text") or "").strip():
            current = tree.heading(block)
        elif label in ABSTRACT_LABELS and current["section_type"] == SECTION_FRONT:
            current = tree.add("sec_abstract", None, None, 1, SECTION_ABSTRACT, implicit=True)
        elif label in REFERENCE_LABELS and current["section_type"] != SECTION_REFERENCES:
            current = tree.add("sec_references", None, None, 1, SECTION_REFERENCES, implicit=True)
        block["section_id"] = current["section_id"]
        current["blocks"].append(block["block_id"])
        current["page_start"] = current["page_start"] or block["page_number"]
        current["page_end"] = block["page_number"]
    sections = [s for s in tree.sections if s["blocks"] or s["section_id"] != FRONT_MATTER_ID]
    return sections, title
