#!/usr/bin/env python3
"""
Demonstration of entity extraction improvements.

This script demonstrates the new features:
1. Generic label set for unknown documents
2. Post-processing pipeline with configurable filters
3. Document type auto-detection
4. Multi-scale extraction
"""

from ingest.gliner_labels.gliner_labels import (
    get_builtin_labels,
    list_builtin_labels,
    BUILTIN_LABEL_SETS,
)
from ingest.entity_post_processor import (
    EntityPostProcessor,
    create_processor,
    PostProcessorConfig,
    DEFAULT_CONFIG,
    CONSERVATIVE_CONFIG,
    AGGRESSIVE_CONFIG,
)


def demo_generic_labels():
    """Demonstrate the generic label set."""
    print("=" * 60)
    print("DEMO 1: Generic Label Set")
    print("=" * 60)

    generic = get_builtin_labels("generic")
    print(f"Description: {generic['description']}")
    print(f"Number of labels: {len(generic['labels'])}")
    print(f"Number of buckets: {len(generic['buckets'])}")
    print()
    print("Entity labels:")
    for label in generic['labels']:
        bucket = generic['label_map'].get(label, 'N/A')
        print(f"  - {label:20} → {bucket}")
    print()


def demo_post_processing():
    """Demonstrate post-processing pipeline."""
    print("=" * 60)
    print("DEMO 2: Post-Processing Pipeline")
    print("=" * 60)

    # Simulate raw GLiNER output with noise
    raw_entities = {
        'people': [
            {'text': 'John Smith', 'label': 'person', 'score': 0.85},
            {'text': 'john smith', 'label': 'person', 'score': 0.75},  # duplicate
            {'text': 'A', 'label': 'person', 'score': 0.45},  # too short
            {'text': 'Jane Doe', 'label': 'person', 'score': 0.90},
        ],
        'organizations': [
            {'text': 'Acme Corp', 'label': 'organization', 'score': 0.88},
            {'text': 'the', 'label': 'organization', 'score': 0.35},  # stopword
            {'text': 'Tech Inc', 'label': 'organization', 'score': 0.92},
        ],
        'dates': [
            {'text': 'January 15, 2024', 'label': 'date', 'score': 0.95},
        ]
    }

    label_map = {
        'person': 'people',
        'organization': 'organizations',
        'date': 'dates',
    }

    print(f"Raw entities: {sum(len(v) for v in raw_entities.values())} total")
    for bucket, entities in raw_entities.items():
        print(f"  {bucket}: {len(entities)} entities")
    print()

    # Test different configs
    for config_name in ['default', 'conservative', 'aggressive']:
        processor = create_processor(config_name)
        result = processor.process(raw_entities, label_map)
        total = sum(len(v) for v in result.values())
        print(f"{config_name.capitalize():12} config: {total} entities after filtering")
    print()


def demo_document_detection():
    """Demonstrate document type detection."""
    print("=" * 60)
    print("DEMO 3: Document Type Auto-Detection")
    print("=" * 60)

    # Import just the function to avoid docling dependency
    import re

    def detect_document_type(text: str, sample_size: int = 5000) -> str:
        sample = text[:sample_size].lower()

        signals = {
            'edilizia': [
                'cig', 'cup', 'appalto', 'gara', 'committente', 'affidamento',
                'procedura negoziata', 'scadenza', 'ingegnere', 'architetto',
            ],
            'research': [
                'abstract', 'methodology', 'results', 'conclusion', 'et al',
                'experiment', 'dataset', 'neural', 'accuracy', 'benchmark',
            ],
            'legal': [
                'contract', 'agreement', 'clause', 'jurisdiction',
                'plaintiff', 'defendant', 'court', 'liable', 'indemnify',
            ],
            'finance': [
                'revenue', 'earnings', 'fiscal year', 'quarterly',
                'balance sheet', 'cash flow', 'stock', 'dividend',
            ],
        }

        scores = {}
        for doc_type, patterns in signals.items():
            score = sum(1 for pattern in patterns if pattern in sample)
            scores[doc_type] = score

        best_type = max(scores, key=scores.get)
        if scores[best_type] >= 2:
            return best_type
        return 'generic'

    test_docs = [
        ("The CUP is 12345 and the appalto is open. Ingegnere required.", "edilizia"),
        ("This paper presents methodology and results. Accuracy improved.", "research"),
        ("This contract shall be governed by law. Plaintiff agrees.", "legal"),
        ("Q3 revenue increased. Fiscal year results show growth.", "finance"),
        ("Just some random text about various topics and things.", "generic"),
    ]

    print("Document samples and detected types:")
    for text, expected in test_docs:
        detected = detect_document_type(text)
        status = "✓" if detected == expected else "✗"
        print(f"  {status} Expected: {expected:12} Detected: {detected:12}")
        print(f"     Sample: {text[:50]}...")
    print()


def demo_config_comparison():
    """Compare the three pre-configured settings."""
    print("=" * 60)
    print("DEMO 4: Config Comparison")
    print("=" * 60)

    configs = {
        'default': DEFAULT_CONFIG,
        'conservative': CONSERVATIVE_CONFIG,
        'aggressive': AGGRESSIVE_CONFIG,
    }

    print(f"{'Config':<12} {'Min Conf':<10} {'Min Len':<8} {'Stopwords':<12} {'Multi-scale'}")
    print("-" * 60)
    for name, config in configs.items():
        print(f"{name:<12} {config.min_confidence:<10.2f} {config.min_entity_length:<8} "
              f"{config.stopword_ratio_threshold:<12.1%} {config.enable_multiscale}")
    print()

    print("Recommendations:")
    print("  - default:     Balanced precision/recall for most documents")
    print("  - conservative: High precision, use when accuracy is critical")
    print("  - aggressive:   High recall, use for exploratory analysis")
    print()


def main():
    """Run all demonstrations."""
    print("\n")
    print("█" * 60)
    print("  ENTITY EXTRACTION IMPROVEMENTS - DEMONSTRATION")
    print("█" * 60)
    print()

    demo_generic_labels()
    demo_post_processing()
    demo_document_detection()
    demo_config_comparison()

    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("Usage examples:")
    print("  python ingest.py document.pdf --labels generic")
    print("  python ingest.py document.pdf --labels generic --post-process-config aggressive")
    print("  python ingest.py document.pdf --labels generic --multiscale")
    print()


if __name__ == "__main__":
    main()
