"""Small evaluation helpers for recognized equations: exact match and character error rate on
canonical LaTeX, recognition success rate, equation numbers.

Engineering checks on fixtures with known LaTeX. Not a benchmark.
"""
import re

from .layout_eval import match_regions
from .math_ocr import STATUS_SUCCESS, normalize_latex
from .ocr_eval import edit_counts

_TOKEN = re.compile(r"\\(?:begin|end)\s*\{[^{}]*\}|\\[A-Za-z]+\*?|\\.|\S", re.DOTALL)
# Tokens that only set spacing or size. They do not change which formula it is.
_LAYOUT_TOKENS = frozenset({
    r"\,", r"\;", r"\:", r"\!", r"\ ", "~", r"\quad", r"\qquad", r"\enspace", r"\thinspace",
    r"\displaystyle", r"\textstyle", r"\limits", r"\nolimits", r"\left", r"\right",
    r"\big", r"\Big", r"\bigg", r"\Bigg", r"\bigl", r"\bigr", r"\Bigl", r"\Bigr",
})
# \mathrm{log} and \operatorname{log} are the same operator as \log.
_OPERATORS = frozenset({"log", "ln", "exp", "sin", "cos", "tan", "max", "min", "lim", "det", "arg", "sup", "inf"})
_UPRIGHT = (r"\mathrm", r"\operatorname")
# Environments whose \begin is followed by an argument in braces.
_ARGUMENT_ENVIRONMENTS = ("array", "tabular", "alignedat", "subarray")
# Old-style switches: the braces around "{\bf x}" limit what they apply to.
_SWITCHES = frozenset({r"\bf", r"\rm", r"\it", r"\cal", r"\sf", r"\tt", r"\scriptstyle", r"\scriptscriptstyle",
                       r"\color", r"\small", r"\large"})
_MATRIX_DELIMITERS = {("[", "]"): "bmatrix", ("(", ")"): "pmatrix"}


def _takes_argument(token):
    """True when a brace group after `token` can be its argument."""
    if token.startswith(r"\begin"):
        return any(token.endswith("{" + name + "}") for name in _ARGUMENT_ENVIRONMENTS)
    return token in ("_", "^", "}") or (token.startswith("\\") and token[1:2].isalpha()
                                         and not token.startswith(r"\end"))


def _drop_grouping_braces(tokens):
    """Remove brace pairs that only group: not an argument of a command or script, not carrying a
    script themselves, not limiting a switch. "-{\\sum_{i}}y" -> "-\\sum_{i}y", "L{=}x" -> "L=x"."""
    changed = True
    while changed:
        changed = False
        stack, pairs = [], {}
        for index, token in enumerate(tokens):
            if token == "{":
                stack.append(index)
            elif token == "}" and stack:
                pairs[stack.pop()] = index
        for start in sorted(pairs):
            end = pairs[start]
            after = tokens[end + 1] if end + 1 < len(tokens) else None
            if ((start and _takes_argument(tokens[start - 1])) or after in ("_", "^")
                    or tokens[start + 1] in _SWITCHES):
                continue
            tokens = tokens[:start] + tokens[start + 1:end] + tokens[end + 1:]
            changed = True
            break
    return tokens


def _name_bracketed_matrices(tokens):
    """[ \\begin{matrix} ... \\end{matrix} ] -> \\begin{bmatrix} ... \\end{bmatrix}; ( ) -> pmatrix."""
    out, index = [], 0
    while index < len(tokens):
        if tokens[index] == r"\begin{matrix}" and out and r"\end{matrix}" in tokens[index:]:
            end = tokens.index(r"\end{matrix}", index)
            name = _MATRIX_DELIMITERS.get((out[-1], tokens[end + 1] if end + 1 < len(tokens) else None))
            if name and r"\begin{matrix}" not in tokens[index + 1:end]:
                out[-1:] = [rf"\begin{{{name}}}"] + tokens[index + 1:end] + [rf"\end{{{name}}}"]
                index = end + 2
                continue
        out.append(tokens[index])
        index += 1
    return out


def canonical_latex(latex):
    """One spelling for formulas that differ only in how they are written. Used for comparison only;
    equations.json never holds this form.

      - normalize_latex, then all whitespace removed (one space is kept after a command name that is
        followed by a letter: "\\alpha x");
      - spacing and size commands dropped: \\, \\; \\: \\! \\quad \\qquad ~ \\displaystyle \\left \\right \\big ...;
      - \\mathrm{log} / \\operatorname{log} -> \\log (for the usual operator names);
      - a one-token subscript or superscript gets braces: x_i^2 -> x_{i}^{2};
      - braces that only group are dropped: -{\\sum_{i}}y -> -\\sum_{i}y, L{=}x -> L=x, {a}&{b} -> a&b;
      - a row break directly before \\end{...} is dropped;
      - a matrix between [ ] or ( ) is written as bmatrix or pmatrix.

    Symbols and structure are never changed: "0" and "O", "1" and "l", "x" and "\\times", "a" and
    "\\alpha" stay different, and so do "&=" and "=&".
    """
    tokens = [re.sub(r"\s+", "", token) if token.startswith((r"\begin", r"\end")) else token
              for token in _TOKEN.findall(normalize_latex(latex))]
    tokens = [token for token in tokens if token not in _LAYOUT_TOKENS]

    merged, i = [], 0
    while i < len(tokens):      # \mathrm { l o g } -> \log
        if tokens[i] in _UPRIGHT and i + 1 < len(tokens) and tokens[i + 1] == "{" and "}" in tokens[i + 2:]:
            end = tokens.index("}", i + 2)
            word = "".join(tokens[i + 2:end])
            if word in _OPERATORS:
                merged.append("\\" + word)
                i = end + 1
                continue
        merged.append(tokens[i])
        i += 1

    braced, i = [], 0
    while i < len(merged):      # x_i -> x_{i}
        token = merged[i]
        following = merged[i + 1] if i + 1 < len(merged) else None
        if token in ("_", "^") and following is not None and following not in ("{", "}"):
            braced += [token, "{", following, "}"]
            i += 2
        else:
            braced.append(token)
            i += 1

    out = _drop_grouping_braces(braced)
    out = [token for index, token in enumerate(out)
           if not (token == r"\\" and index + 1 < len(out) and out[index + 1].startswith(r"\end"))]
    out = _name_bracketed_matrices(out)

    text = ""
    for token in out:
        if text and token[0].isalpha() and re.search(r"\\[A-Za-z]+\*?$", text):
            text += " "
        text += token
    return text


def strict_latex(latex):
    """The formula with its whitespace removed and nothing else changed."""
    return "".join(normalize_latex(latex).split())


def latex_cer(reference, hypothesis):
    """Character error rate between two formulas in canonical form: (S + D + I) / characters of the
    reference. Can exceed 1. Raises ValueError for an empty reference."""
    expected, found = canonical_latex(reference), canonical_latex(hypothesis or "")
    if not expected:
        raise ValueError("the reference formula is empty; an error rate is not defined")
    return sum(edit_counts(expected, found)) / len(expected)


def evaluate_equation(expected, predicted):
    """Compare one equation of equations.json (or None when the equation was not found) with its
    ground truth {"latex", optionally "equation_number"}.

    The predicted formula is `latex`, or `normalized_latex` when the validation rejected it: a rejected
    formula still has a character error rate but does not count as recognized. Returns

        status                     the equation's status, or None when it was not found
        recognized                 status == "success"
        exact_match                canonical forms are identical (canonical_latex)
        strict_match               identical once whitespace is removed, nothing else
        characters, edits, cer     on the canonical forms; cer = edits / characters
        normalized_latex_accuracy  1 - min(1, cer)
        number_match               equation numbers are equal (both null counts); None when the
                                   ground truth does not say
    """
    reference = canonical_latex(expected["latex"])
    if not reference:
        raise ValueError("the reference formula is empty")
    text = (predicted or {}).get("latex") or (predicted or {}).get("normalized_latex") or ""
    hypothesis = canonical_latex(text)
    edits = sum(edit_counts(reference, hypothesis))
    number_match = None
    if "equation_number" in expected:
        number_match = predicted is not None and predicted.get("equation_number") == expected["equation_number"]
    return {
        "status": predicted and predicted.get("status"),
        "recognized": bool(predicted) and predicted.get("status") == STATUS_SUCCESS,
        "exact_match": reference == hypothesis,
        "strict_match": strict_latex(expected["latex"]) == strict_latex(text),
        "characters": len(reference),
        "edits": edits,
        "cer": edits / len(reference),
        "normalized_latex_accuracy": 1.0 - min(1.0, edits / len(reference)),
        "number_match": number_match,
        "expected": reference,
        "predicted": hypothesis,
    }


def evaluate_equations(truth_equations, document, iou_threshold=0.3):
    """Compare an equations.json object with ground-truth equations ([{"page_number", "pdf_bbox",
    "latex", "equation_number"}]).

    Each ground-truth equation is paired with the extracted equation at the same place on the same
    page (boxes overlapping with IoU >= iou_threshold). Returns {"rows": [evaluate_equation result +
    "page_number", "equation_id", "kind"], "equations", "found", "recognized",
    "recognition_success_rate", "exact_matches", "exact_match_rate", "strict_matches", "characters", "edits", "cer",
    "normalized_latex_accuracy", "numbers_expected", "numbers_correct", "number_accuracy"}.

    All rates are over all ground-truth equations: one that was not found counts as not recognized,
    with every character wrong. `cer` is the edits summed over the characters summed;
    `normalized_latex_accuracy` is the mean of the per-equation values. `numbers_correct` counts the
    numbered ground-truth equations whose number was found; `number_accuracy` also counts an
    unnumbered equation that got no number as correct.
    """
    predicted = [{**equation, "type": "EQUATION"} for page in document["pages"] for equation in page["equations"]]
    truth = [{**equation, "type": "EQUATION"} for equation in truth_equations]
    by_box = {(equation["page_number"], tuple(equation["pdf_bbox"])): equation for equation in predicted}

    rows = []
    for expected, match in zip(truth, match_regions(truth, predicted, iou_threshold)):
        equation = by_box.get((match["page_number"], tuple(match["predicted_bbox"] or ())))
        rows.append({"page_number": expected["page_number"], "kind": expected.get("kind"),
                     "equation_id": equation and equation["equation_id"], **evaluate_equation(expected, equation)})

    count = len(rows)
    characters, edits = sum(row["characters"] for row in rows), sum(row["edits"] for row in rows)
    numbered = [row for row, expected in zip(rows, truth) if expected.get("equation_number")]
    judged = [row for row in rows if row["number_match"] is not None]
    return {
        "rows": rows,
        "equations": count,
        "found": sum(1 for row in rows if row["equation_id"]),
        "recognized": sum(1 for row in rows if row["recognized"]),
        "recognition_success_rate": sum(1 for row in rows if row["recognized"]) / count if count else None,
        "exact_matches": sum(1 for row in rows if row["exact_match"]),
        "exact_match_rate": sum(1 for row in rows if row["exact_match"]) / count if count else None,
        "strict_matches": sum(1 for row in rows if row["strict_match"]),
        "characters": characters,
        "edits": edits,
        "cer": edits / characters if characters else None,
        "normalized_latex_accuracy": sum(row["normalized_latex_accuracy"] for row in rows) / count if count else None,
        "numbers_expected": len(numbered),
        "numbers_correct": sum(1 for row in numbered if row["number_match"]),
        "number_accuracy": sum(1 for row in judged if row["number_match"]) / len(judged) if judged else None,
    }
