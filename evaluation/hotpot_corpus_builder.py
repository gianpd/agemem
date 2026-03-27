"""
Build a HotpotQA corpus for evaluation.

Ingests all unique Wikipedia paragraphs from HotpotQA dataset
into the document corpus. Run this ONCE before evaluation.

Usage:
    # Build full corpus (all paragraphs)
    python evaluation/hotpot_corpus_builder.py

    # Build gold-only corpus (smaller, oracle-like)
    python evaluation/hotpot_corpus_builder.py --gold-only

    # Limit for testing
    python evaluation/hotpot_corpus_builder.py --limit 100
"""

from __future__ import annotations

import argparse
import json
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
        return f"""---
title: {self.title}
source: hotpotqa
---

# {self.title}

{self.text}
"""


def load_unique_paragraphs(
    split: str = "validation",
    setting: str = "distractor",
    limit: int = 0,
    gold_only: bool = False,
) -> dict[str, Paragraph]:
    """
    Load all unique paragraphs from HotpotQA dataset.

    Returns:
        Dict mapping title -> Paragraph (deduplicated)
    """
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

        # Get gold titles for this question
        gold_titles = set(item["supporting_facts"]["title"])

        # Process each context paragraph
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            # Skip if gold_only and this isn't a gold paragraph
            if gold_only and title not in gold_titles:
                continue

            # Skip if already processed
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


def build_corpus(
    paragraphs: dict[str, Paragraph],
    output_dir: Path,
    batch_size: int = 100,
) -> int:
    """
    Ingest paragraphs into corpus.

    Returns:
        Number of documents successfully ingested
    """
    from ingest.ingest import ingest

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Ingesting {len(paragraphs)} paragraphs to {output_dir}...")

    # Write all markdown files first
    md_files: list[tuple[str, Path]] = []
    for title, para in paragraphs.items():
        # Sanitize title for filename
        safe_title = title.replace("/", "_").replace("\\", "_")[:100]
        md_path = output_dir / f"{safe_title}.md"
        md_path.write_text(para.to_markdown())
        md_files.append((title, md_path))

    print(f"[INFO] Written {len(md_files)} markdown files")

    # Ingest in batches
    success_count = 0
    failed = []

    for i, (title, md_path) in enumerate(md_files):
        try:
            doc_id = ingest(str(md_path))
            success_count += 1

            if (i + 1) % batch_size == 0:
                print(f"[INFO] Ingested {i + 1}/{len(md_files)} documents...")

        except Exception as e:
            failed.append((title, str(e)))
            logger.warning(f"Failed to ingest {title}: {e}")

    print(f"[INFO] Successfully ingested: {success_count}/{len(md_files)}")

    if failed:
        print(f"[WARN] Failed: {len(failed)}")
        for title, err in failed[:5]:
            print(f"  - {title}: {err[:50]}...")

    return success_count


def main():
    parser = argparse.ArgumentParser(description="Build HotpotQA corpus")
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
        "--output-dir", type=Path,
        default=Path("corpus/hotpotqa"),
        help="Output directory for corpus (default: corpus/hotpotqa)"
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
        "--batch-size", type=int, default=100,
        help="Batch size for progress reporting"
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

    # Build corpus
    count = build_corpus(
        paragraphs=paragraphs,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    elapsed = time.time() - start
    print(f"\n[DONE] Ingested {count} documents in {elapsed:.1f}s")
    print(f"[INFO] Corpus location: {args.output_dir}")


if __name__ == "__main__":
    main()