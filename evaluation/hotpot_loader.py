"""
HotpotQA dataset loader.

HotpotQA is a multi-hop question answering dataset requiring reasoning over multiple documents.
Dataset format: http://curtis.ml.cmu.edu/datasets/hotpot/
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HotpotParagraph:
    """A paragraph from HotpotQA context."""
    title: str
    sentences: list[str]

    @property
    def text(self) -> str:
        """Full paragraph text."""
        return " ".join(self.sentences)


@dataclass
class SupportingFact:
    """A supporting fact reference."""
    title: str
    sentence_idx: int


@dataclass
class HotpotItem:
    """A single HotpotQA question instance."""
    id: str
    question: str
    answer: str
    supporting_facts: list[SupportingFact]
    context: list[HotpotParagraph]
    type: str  # "comparison" or "bridge"
    level: str  # "easy", "medium", "hard"


class HotpotLoader:
    """Loader for HotpotQA dataset files."""

    def load(self, path: Path | str) -> list[HotpotItem]:
        """Load HotpotQA dataset from JSON file."""
        path = Path(path)
        logger.info(f"Loading HotpotQA from {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        items = [self._parse_item(item) for item in raw_data]
        logger.info(f"Loaded {len(items)} HotpotQA items")
        return items

    def _parse_item(self, raw: dict) -> HotpotItem:
        """Parse a raw dict into HotpotItem."""
        context = [
            HotpotParagraph(title=para[0], sentences=para[1])
            for para in raw["context"]
        ]

        supporting_facts = [
            SupportingFact(title=fact[0], sentence_idx=fact[1])
            for fact in raw["supporting_facts"]
        ]

        return HotpotItem(
            id=raw["_id"],
            question=raw["question"],
            answer=raw["answer"],
            supporting_facts=supporting_facts,
            context=context,
            type=raw["type"],
            level=raw["level"],
        )


def print_schema_stats(items: list[HotpotItem], sample_size: int = 3) -> None:
    """Print dataset statistics and sample items."""
    print(f"\n{'='*60}")
    print("HOTPOTQA DATASET SCHEMA")
    print(f"{'='*60}")
    print(f"\nTotal items: {len(items)}")

    # Count types and levels
    types = {}
    levels = {}
    for item in items:
        types[item.type] = types.get(item.type, 0) + 1
        levels[item.level] = levels.get(item.level, 0) + 1

    print(f"\nQuestion types: {types}")
    print(f"Difficulty levels: {levels}")

    # Context stats
    context_counts = [len(item.context) for item in items]
    sentence_counts = [len(p.sentences) for item in items for p in item.context]
    supporting_fact_counts = [len(item.supporting_facts) for item in items]

    print(f"\nContext paragraphs per question: min={min(context_counts)}, max={max(context_counts)}, avg={sum(context_counts)/len(context_counts):.1f}")
    print(f"Sentences per paragraph: min={min(sentence_counts)}, max={max(sentence_counts)}, avg={sum(sentence_counts)/len(sentence_counts):.1f}")
    print(f"Supporting facts per question: min={min(supporting_fact_counts)}, max={max(supporting_fact_counts)}, avg={sum(supporting_fact_counts)/len(supporting_fact_counts):.1f}")

    # Sample items
    print(f"\n{'='*60}")
    print(f"SAMPLE ITEMS (first {sample_size})")
    print(f"{'='*60}")

    for i, item in enumerate(items[:sample_size]):
        print(f"\n--- Item {i+1} ---")
        print(f"ID: {item.id}")
        print(f"Question: {item.question}")
        print(f"Answer: {item.answer}")
        print(f"Type: {item.type} | Level: {item.level}")
        print(f"Supporting facts: {[(f.title, f.sentence_idx) for f in item.supporting_facts]}")
        print(f"Context titles: {[p.title for p in item.context]}")

        # Show supporting fact sentences
        print("Supporting sentences:")
        for fact in item.supporting_facts:
            for para in item.context:
                if para.title == fact.title:
                    if fact.sentence_idx < len(para.sentences):
                        sent = para.sentences[fact.sentence_idx]
                        print(f"  [{fact.title}:{fact.sentence_idx}] {sent[:100]}...")
                    break


if __name__ == "__main__":
    # Test loader with local dataset
    loader = HotpotLoader()
    items = loader.load("documents/hotpot_dev_fullwiki_v1.json")
    print_schema_stats(items)