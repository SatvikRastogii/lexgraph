"""Locate, for every answerable question, which paragraphs carry the answer.

Ground truth in this gold set is document-level: a retriever scores full marks
for returning any chunk of the right judgment, including the one listing who
appeared for whom. That flatters every configuration equally, so the ablation
cannot see a retriever that finds the correct case but the wrong part of it.

These labels are derived, not hand-written, and that is deliberate. Each
question already carries ``must_contain`` terms that
``scripts/validate_goldset.py`` proves against the corpus text. Locating those
same terms inside the judgment's own numbered paragraphs extends an annotation
that is already verified rather than adding one that is merely asserted -- and
it stays correct if the corpus is re-scraped, which hand-written spans would
not.

The trade-off is honest but real: a term-bearing paragraph is where the answer
is *stated*, which is not always the whole of where it is *reasoned*. Treat
paragraph recall as a floor on passage quality, not a full account of it.

Four of the forty judgments carry no usable numbering and are skipped; queries
whose supporting documents are all unnumbered are excluded from the metric
rather than scored zero.

    python scripts/derive_paragraph_labels.py            # rewrite the gold set
    python scripts/derive_paragraph_labels.py --dry-run  # report only
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexgraph.chunking import split_numbered_paragraphs
from lexgraph.corpus import load_corpus

GOLDSET_PATH = os.path.join("data", "goldset.json")

# A term that appears in most of a judgment localises nothing -- "227" occurs
# 172 times in judgment_0128. Labelling 90% of the paragraphs would make the
# metric indistinguishable from document-level recall, so those are dropped.
MAX_PARAGRAPH_FRACTION = 0.5


def paragraphs_containing(body: str, terms: list[str]) -> tuple[list[int], int]:
    """Numbered paragraphs holding any term, and how many are numbered at all."""
    numbered = [
        (number, text) for number, text in split_numbered_paragraphs(body)
        if number is not None
    ]
    lowered = [t.lower() for t in terms]
    hits = [
        number for number, text in numbered
        if any(term in text.lower() for term in lowered)
    ]
    return sorted(set(hits)), len(numbered)


def derive(goldset: dict, documents: list) -> tuple[dict, list[str]]:
    """Return ``{question_id: {doc: [paragraphs]}}`` and a list of notes."""
    by_name = {document.filename: document for document in documents}
    labels, notes = {}, []

    for question in goldset["questions"]:
        if not question["answerable"] or not question.get("must_contain"):
            continue

        per_document = {}
        for name in question["relevant_docs"]:
            document = by_name.get(name)
            if document is None:
                continue
            hits, total = paragraphs_containing(document.body, question["must_contain"])
            if not hits:
                continue
            if total and len(hits) / total > MAX_PARAGRAPH_FRACTION:
                notes.append(
                    f"{question['id']}: {name} skipped -- term appears in "
                    f"{len(hits)}/{total} paragraphs, too diffuse to localise"
                )
                continue
            per_document[name] = hits

        if per_document:
            labels[question["id"]] = per_document
        else:
            notes.append(f"{question['id']}: no paragraph labels derived")

    return labels, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", default=GOLDSET_PATH)
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.goldset, encoding="utf-8") as handle:
        goldset = json.load(handle)
    documents = load_corpus(args.input_dir)

    labels, notes = derive(goldset, documents)

    answerable = [q for q in goldset["questions"] if q["answerable"]]
    spans = sum(len(p) for doc in labels.values() for p in doc.values())
    print(f"Labelled {len(labels)}/{len(answerable)} answerable questions")
    print(f"{spans} paragraph spans across "
          f"{len({d for doc in labels.values() for d in doc})} documents")

    if notes:
        print(f"\n{len(notes)} question(s) without usable labels:")
        for note in notes:
            print(f"  - {note}")

    if args.dry_run:
        return

    for question in goldset["questions"]:
        derived = labels.get(question["id"])
        if derived:
            question["relevant_paragraphs"] = derived
        else:
            question.pop("relevant_paragraphs", None)

    with open(args.goldset, "w", encoding="utf-8") as handle:
        json.dump(goldset, handle, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.goldset}")


if __name__ == "__main__":
    main()
