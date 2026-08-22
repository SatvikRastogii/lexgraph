"""Check every gold-set annotation against the corpus text.

A gold set is only worth the labels in it. This verifies, mechanically:

  * every ``relevant_docs`` entry names a file that exists in the corpus
  * each answerable question's supporting documents actually contain the terms
    the annotation claims they do
  * each ``answerable: false`` question is genuinely unsupported -- its
    ``must_not_contain`` terms appear in no document at all
  * the dissent question's ground truth matches the HAS DISSENT header

Run it in CI. A gold set that silently rots as the corpus changes is worse
than no gold set, because it keeps producing plausible numbers.

    python scripts/validate_goldset.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexgraph.corpus import load_corpus

GOLDSET_PATH = os.path.join("data", "goldset.json")


def load_goldset(path=GOLDSET_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def validate(goldset, documents):
    """Return a list of human-readable problems; empty means the gold set is sound."""
    by_name = {document.filename: document for document in documents}
    corpus_text = {name: doc.body.lower() for name, doc in by_name.items()}
    problems = []

    seen_ids = set()
    for question in goldset["questions"]:
        qid = question["id"]
        if qid in seen_ids:
            problems.append(f"{qid}: duplicate question id")
        seen_ids.add(qid)

        answerable = question["answerable"]
        relevant = question["relevant_docs"]

        if answerable and not relevant:
            problems.append(f"{qid}: answerable but lists no relevant documents")
        if not answerable and relevant:
            problems.append(f"{qid}: marked unanswerable but lists relevant documents")

        for name in relevant:
            if name not in by_name:
                problems.append(f"{qid}: relevant doc {name!r} is not in the corpus")

        terms = [t.lower() for t in question.get("must_contain", [])]
        if terms and answerable:
            present = [
                name
                for name in relevant
                if name in corpus_text
                and any(term in corpus_text[name] for term in terms)
            ]
            if question.get("must_contain_any"):
                # At least one supporting document must carry the term.
                if not present:
                    problems.append(
                        f"{qid}: none of {relevant} contain any of {terms}"
                    )
            else:
                missing = [n for n in relevant if n not in present]
                if missing:
                    problems.append(f"{qid}: {missing} do not contain any of {terms}")

        for term in question.get("must_not_contain", []):
            hits = [name for name, text in corpus_text.items() if term.lower() in text]
            if hits:
                problems.append(
                    f"{qid}: marked out-of-corpus but {term!r} appears in {hits[:3]}"
                    f"{' and more' if len(hits) > 3 else ''}"
                )

    problems.extend(_validate_dissent(goldset, documents))
    problems.extend(_validate_paragraphs(goldset, by_name))
    return problems


def _validate_paragraphs(goldset, by_name):
    """Every labelled paragraph must exist and carry one of the question's terms.

    These labels are derived by scripts/derive_paragraph_labels.py rather than
    written by hand, which makes them cheap to produce but easy to leave stale.
    Re-checking them here means a corpus change that moves the paragraphs
    fails CI instead of quietly scoring a retriever against the wrong spans.
    """
    from lexgraph.chunking import split_numbered_paragraphs

    problems = []
    for question in goldset["questions"]:
        labels = question.get("relevant_paragraphs")
        if not labels:
            continue
        qid = question["id"]
        terms = [t.lower() for t in question.get("must_contain", [])]

        for name, numbers in labels.items():
            document = by_name.get(name)
            if document is None:
                problems.append(f"{qid}: paragraph labels for unknown doc {name!r}")
                continue
            if name not in question["relevant_docs"]:
                problems.append(f"{qid}: {name} has paragraph labels but is not relevant")
                continue

            # A number can appear more than once: several judgments restart
            # numbering for a second opinion, so paragraph 3 exists twice. All
            # segments carrying that number are collected, and the label is
            # sound if any of them holds a term.
            texts_by_number = {}
            for number, text in split_numbered_paragraphs(document.body):
                if number is not None:
                    texts_by_number.setdefault(number, []).append(text.lower())

            for number in numbers:
                bodies = texts_by_number.get(number)
                if not bodies:
                    problems.append(f"{qid}: {name} has no paragraph {number}")
                elif terms and not any(
                    term in body for body in bodies for term in terms
                ):
                    problems.append(
                        f"{qid}: {name} para {number} carries none of {terms}"
                    )
    return problems


def _validate_dissent(goldset, documents):
    """The dissent question's labels must match the HAS DISSENT header exactly."""
    question = next((q for q in goldset["questions"] if q["id"] == "q24"), None)
    if question is None:
        return []
    actual = {d.filename for d in documents if d.has_dissent}
    claimed = set(question["relevant_docs"])
    problems = []
    if claimed - actual:
        problems.append(f"q24: claimed dissent but header says otherwise: {sorted(claimed - actual)}")
    if actual - claimed:
        problems.append(f"q24: header records a dissent not listed: {sorted(actual - claimed)}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", default=GOLDSET_PATH)
    parser.add_argument("--input-dir", default="input")
    args = parser.parse_args()

    goldset = load_goldset(args.goldset)
    documents = load_corpus(args.input_dir)
    questions = goldset["questions"]

    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    print(f"Corpus:    {len(documents)} documents from {args.input_dir}/")
    print(f"Gold set:  {len(questions)} questions "
          f"({len(answerable)} answerable, {len(unanswerable)} out-of-corpus)")

    problems = validate(goldset, documents)
    if problems:
        print(f"\n{len(problems)} problem(s) found:\n")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    print("\nAll annotations verified against the corpus text.")


if __name__ == "__main__":
    main()
