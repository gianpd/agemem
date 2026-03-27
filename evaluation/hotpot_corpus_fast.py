"""
Fast HotpotQA corpus builder - uses gliner NER for entity extraction.

Writes markdown files to corpus with proper entity metadata for corpus tools.

Usage:
    python evaluation/hotpot_corpus_fast.py
    python evaluation/hotpot_corpus_fast.py --gold-only
    python evaluation/hotpot_corpus_fast.py --limit 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


@dataclass
class Paragraph:
    """A unique paragraph from HotpotQA."""
    title: str
    text: str
    sentence_count: int

    def to_markdown(self) -> str:
        """Convert to markdown with frontmatter."""
        safe_title = self.title.replace('"', "'")
        return f"""---
title: "{safe_title}"
source: hotpotqa
type: wikipedia_paragraph
---

# {self.title}

{self.text}
"""


def load_unique_paragraphs(
    split: str = "validation",
    setting: str = "distractor",
    limit: int = 0,
    gold_only: bool = False,
) -> dict[str, "Paragraph"]:
    """Load all unique paragraphs from HotpotQA dataset."""
    from datasets import load_dataset

    print(f"[INFO] Loading HotpotQA {split} ({setting})...")
    dataset = load_dataset("hotpot_qa", setting, split=split)

    paragraphs: dict[str, Paragraph] = {}
    gold_titles_seen: set[str] = set()

    for idx, item in enumerate(dataset):
        if limit > 0 and idx >= limit:
            break

        if idx % 1000 == 0:
            print(f"[INFO] Processing item {idx}...")

        gold_titles = set(item["supporting_facts"]["title"])

        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            if gold_only and title not in gold_titles:
                continue

            if title in paragraphs:
                continue

            text = " ".join(sentences)
            paragraphs[title] = Paragraph(
                title=title,
                text=text,
                sentence_count=len(sentences),
            )

            if title in gold_titles:
                gold_titles_seen.add(title)

    print(f"[INFO] Unique paragraphs: {len(paragraphs)}")
    print(f"[INFO] Gold titles included: {len(gold_titles_seen)}")

    return paragraphs


def ingest_with_gliner(
    paragraphs: dict[str, "Paragraph"],
    corpus_dir: Path,
    batch_size: int = 100,
    labels: str = "generic",
) -> int:
    """
    Ingest paragraphs through gliner for entity extraction.

    Uses ingest_markdown which skips docling and directly extracts entities.
    """
    import tempfile
    from ingest.ingest import ingest_markdown

    corpus_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Ingesting {len(paragraphs)} paragraphs with gliner...")
    print(f"[INFO] Using labels: {labels}")

    success_count = 0
    failed = []

    for i, (title, para) in enumerate(paragraphs.items()):
        try:
            # Create temp markdown file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".md",
                delete=False,
                dir=corpus_dir,
                prefix=f"hotpot_{i}_"
            ) as f:
                f.write(para.to_markdown())
                temp_path = Path(f.name)

            # Ingest through gliner
            doc_id = ingest_markdown(temp_path, doc_type="hotpotqa", labels_arg=labels)
            success_count += 1

            # Clean up temp file
            temp_path.unlink(missing_ok=True)

            if (i + 1) % batch_size == 0:
                print(f"[INFO] Ingested {i + 1}/{len(paragraphs)} documents...")

        except Exception as e:
            failed.append((title, str(e)[:50]))
            logger.warning(f"Failed to ingest {title}: {e}")

    print(f"[INFO] Successfully ingested: {success_count}/{len(paragraphs)}")

    if failed:
        print(f"[WARN] Failed: {len(failed)}")
        for title, err in failed[:5]:
            print(f"  - {title}: {err}")

    return success_count


def main():
    parser = argparse.ArgumentParser(description="HotpotQA corpus builder with gliner NER")
    parser.add_argument(
        "--split", type=str, default="validation",
        help="Dataset split (default: validation)"
    )
    parser.add_argument(
        "--setting", type=str, default="distractor",
        choices=["distractor", "fullwiki"],
        help="Dataset setting (default: distractor)"
    )
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=Path("corpus"),
        help="Corpus directory (default: corpus/)"
    )
    parser.add_argument(
        "--gold-only", action="store_true",
        help="Only include gold paragraphs (smaller corpus)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of questions to process (0 = all)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Progress reporting batch size (default: 50)"
    )
    parser.add_argument(
        "--labels", type=str, default="generic",
        help="Gliner label set (default: generic)"
    )
    args = parser.parse_args()

    start = time.time()

    # Load unique paragraphs
    paragraphs = load_unique_paragraphs(
        split=args.split,
        setting=args.setting,
        limit=args.limit,
        gold_only=args.gold_only,
    )

    # Ingest with gliner
    count = ingest_with_gliner(
        paragraphs=paragraphs,
        corpus_dir=args.corpus_dir,
        batch_size=args.batch_size,
        labels=args.labels,
    )

    elapsed = time.time() - start
    print(f"\n[DONE] Ingested {count} documents in {elapsed:.1f}s")
    print(f"[INFO] Corpus location: {args.corpus_dir}")
    print(f"[INFO] Speed: {count/elapsed:.1f} docs/sec")


if __name__ == "__main__":
    main()