"""BGL log content variability per spec D17 v1.2 + D13.

Applied at test time to BGL anomaly paragraphs only (configs/bgl_*.yaml).
HDFS and Thunderbird disable variability (verified by test_configs.py).

Algorithm:
1. Identify top-K most-frequent verbs in TRAIN paragraphs (NLTK POS tagger).
2. Look up each verb's first-synset-first-lemma in WordNet.
3. Filter multi-word lemmas (D17 v1.2).
4. Apply whole-word substitutions to TEST anomaly paragraphs, preserving case.
5. Return modified paragraphs + audit report.

Reference: logfit-repro-spec-v1.3.md §1.5; decisions-v1.3.md D13, D17.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.types import Paragraph

DEFAULT_TOP_K_VERBS = 10  # D13

VERB_POS_TAGS = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"})
WORD_RE = re.compile(r"^[a-zA-Z]+$")
LEMMA_SEPARATOR_RE = re.compile(r"[_\s\-]")


@dataclass
class VariabilityReport:
    top_verbs: list[str] = field(default_factory=list)
    substitutions: dict[str, str] = field(default_factory=dict)
    skipped_no_lemma: list[str] = field(default_factory=list)
    skipped_multiword_lemma: dict[str, str] = field(default_factory=dict)
    skipped_identity_substitution: list[str] = field(default_factory=list)
    paragraphs_modified: int = 0
    total_substitutions_applied: int = 0


# Pure-logic helpers (no NLTK)

def is_single_word_lemma(lemma: str) -> bool:
    """D17 v1.2: single-word iff no underscore, whitespace, or hyphen."""
    if not lemma:
        return False
    return LEMMA_SEPARATOR_RE.search(lemma) is None


def _preserve_case(original: str, replacement: str) -> str:
    """Match the case of `original` onto `replacement`."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper() and original[1:].islower():
        return replacement.capitalize()
    return replacement.lower()


def substitute_in_lines(
    lines: list[str], substitutions: dict[str, str]
) -> tuple[list[str], int]:
    """Apply verb->lemma substitutions to each line.

    Whole-word, case-insensitive matching. Case preserved per `_preserve_case`.
    """
    if not substitutions:
        return list(lines), 0

    patterns = [
        (re.compile(rf"\b({re.escape(v)})\b", re.IGNORECASE), lemma)
        for v, lemma in substitutions.items()
    ]

    total = 0
    new_lines: list[str] = []
    for line in lines:
        modified = line
        for pattern, lemma in patterns:
            def _repl(match, _lemma=lemma):
                nonlocal total
                total += 1
                return _preserve_case(match.group(1), _lemma)
            modified = pattern.sub(_repl, modified)
        new_lines.append(modified)
    return new_lines, total


def build_substitutions_from_lemmas(
    verb_to_lemma: dict[str, str | None],
) -> tuple[dict[str, str], list[str], dict[str, str], list[str]]:
    """Apply D17 filters to a verb->lemma mapping. Returns
    (accepted, skipped_no_lemma, skipped_multiword, skipped_identity).
    """
    accepted: dict[str, str] = {}
    no_lemma: list[str] = []
    multiword: dict[str, str] = {}
    identity: list[str] = []

    for verb, lemma in verb_to_lemma.items():
        if lemma is None:
            no_lemma.append(verb)
            continue
        if not is_single_word_lemma(lemma):
            multiword[verb] = lemma
            continue
        if lemma.lower() == verb.lower():
            identity.append(verb)
            continue
        accepted[verb] = lemma

    return accepted, no_lemma, multiword, identity


def apply_substitutions_to_paragraphs(
    paragraphs: list[Paragraph], substitutions: dict[str, str]
) -> tuple[list[Paragraph], int, int]:
    """Apply substitutions. Returns (new_paragraphs, n_modified, n_total)."""
    new_paragraphs: list[Paragraph] = []
    n_modified = 0
    n_total = 0
    for p in paragraphs:
        new_lines, count = substitute_in_lines(p.lines, substitutions)
        if count > 0:
            n_modified += 1
            n_total += count
        new_paragraphs.append(Paragraph(
            paragraph_id=p.paragraph_id,
            lines=new_lines,
            label=p.label,
            source_window_id=p.source_window_id,
            start_timestamp=p.start_timestamp,
        ))
    return new_paragraphs, n_modified, n_total


# NLTK-dependent functions

def _import_nltk():
    """Lazy NLTK import with helpful error.

    On Narval pre-download corpora on a login node:
        python -m nltk.downloader wordnet averaged_perceptron_tagger punkt
    """
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        from nltk.tag import pos_tag
        from nltk.tokenize import word_tokenize
        return nltk, wn, pos_tag, word_tokenize
    except ImportError as e:
        raise ImportError(
            "NLTK is required for variability_bgl. Install: `pip install nltk` "
            "and download corpora: `python -m nltk.downloader wordnet "
            "averaged_perceptron_tagger punkt`."
        ) from e


def identify_top_k_verbs(
    paragraphs: list[Paragraph], k: int = DEFAULT_TOP_K_VERBS
) -> list[str]:
    """Top-K verbs across paragraphs (NLTK POS-tagged; alphabetic only)."""
    _, _, pos_tag, word_tokenize = _import_nltk()
    counter: Counter[str] = Counter()
    for p in paragraphs:
        for line in p.lines:
            tokens = word_tokenize(line)
            for word, tag in pos_tag(tokens):
                if tag in VERB_POS_TAGS and WORD_RE.match(word):
                    counter[word.lower()] += 1
    return [w for w, _ in counter.most_common(k)]


def get_first_synset_first_lemma(verb: str) -> str | None:
    """First-synset-first-lemma for `verb` (verb-sense only), or None."""
    _, wn, _, _ = _import_nltk()
    synsets = wn.synsets(verb, pos="v")
    if not synsets:
        return None
    lemmas = synsets[0].lemmas()
    if not lemmas:
        return None
    return lemmas[0].name().lower()


def lookup_lemmas_for_verbs(verbs: list[str]) -> dict[str, str | None]:
    """Map verbs to their first-synset-first-lemma (or None)."""
    return {v: get_first_synset_first_lemma(v) for v in verbs}


def apply_variability(
    train_paragraphs: list[Paragraph],
    test_paragraphs: list[Paragraph],
    k: int = DEFAULT_TOP_K_VERBS,
) -> tuple[list[Paragraph], VariabilityReport]:
    """Identify top verbs from train, substitute in test. Returns (modified, report)."""
    top_verbs = identify_top_k_verbs(train_paragraphs, k)
    verb_to_lemma = lookup_lemmas_for_verbs(top_verbs)
    accepted, no_lemma, multiword, identity = build_substitutions_from_lemmas(
        verb_to_lemma
    )
    modified, n_modified, n_total = apply_substitutions_to_paragraphs(
        test_paragraphs, accepted
    )
    return modified, VariabilityReport(
        top_verbs=top_verbs,
        substitutions=accepted,
        skipped_no_lemma=no_lemma,
        skipped_multiword_lemma=multiword,
        skipped_identity_substitution=identity,
        paragraphs_modified=n_modified,
        total_substitutions_applied=n_total,
    )


def main():
    import argparse
    import json
    import pickle
    from dataclasses import asdict

    parser = argparse.ArgumentParser(
        description="Apply D17 v1.2 BGL variability test."
    )
    parser.add_argument("--train-paragraphs-pkl", type=Path, required=True)
    parser.add_argument("--test-paragraphs-pkl", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--output-report-json", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K_VERBS)
    args = parser.parse_args()

    with args.train_paragraphs_pkl.open("rb") as f:
        train = pickle.load(f)
    with args.test_paragraphs_pkl.open("rb") as f:
        test = pickle.load(f)

    modified, report = apply_variability(train, test, k=args.top_k)

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as f:
        pickle.dump(modified, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.output_report_json:
        args.output_report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_report_json.open("w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)

    print(f"\nVariability applied (D17 v1.2, top-K={args.top_k}):")
    print(f"  Top-{args.top_k} verbs:")
    for i, v in enumerate(report.top_verbs):
        marker = "*" if v in report.substitutions else " "
        print(f"    {marker} {i + 1:2d}. {v}")
    print(f"\n  Accepted substitutions ({len(report.substitutions)}):")
    for v, lemma in report.substitutions.items():
        print(f"    {v!r} -> {lemma!r}")
    if report.skipped_no_lemma:
        print(f"  Skipped (no synset):    {report.skipped_no_lemma}")
    if report.skipped_multiword_lemma:
        print(f"  Skipped (multi-word):")
        for v, l in report.skipped_multiword_lemma.items():
            print(f"    {v!r} -> {l!r}")
    if report.skipped_identity_substitution:
        print(f"  Skipped (lemma==verb):  "
              f"{report.skipped_identity_substitution}")
    print(f"\n  Paragraphs modified: "
          f"{report.paragraphs_modified} / {len(test)}")
    print(f"  Total substitutions: {report.total_substitutions_applied}")
    print(f"  Wrote {args.output_pkl}")
    if args.output_report_json:
        print(f"  Wrote {args.output_report_json}")


if __name__ == "__main__":
    main()
