"""Small text helpers shared by BM25, reranking, citation checking and evaluation."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    """a an and are as at be but by can could did do does for from had has have how i if in into is it its
    may might must no not of on or our shall should so such than that the their them then there these they
    this those to under up was we were what when where which who whom why will with would you your""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
# curly quotes -> straight, en/em/minus dashes -> hyphen, non-breaking space -> space
_TRANSLATE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)


def _stem(tok: str) -> str:
    """Very light suffix stripping - enough to match 'policies'/'policy', 'days'/'day'."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and not tok[-2].isdigit():
        return tok[:-1]
    return tok


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE).lower()
    toks = _TOKEN.findall(text)
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_stem(t) for t in toks]


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lower-case, unify quotes/dashes, collapse whitespace.

    Returns the normalised string and, for each normalised character, the index of the
    original character it came from - so matches can be mapped back for highlighting.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        ch = unicodedata.normalize("NFKC", ch).translate(_TRANSLATE).lower()
        for c in ch:
            if c.isspace():
                if prev_space:
                    continue
                c = " "
                prev_space = True
            else:
                prev_space = False
            out.append(c)
            idx.append(i)
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def normalize(text: str) -> str:
    return normalize_with_map(text)[0]
