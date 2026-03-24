"""
Entity Post-Processing Pipeline for GLiNER NER results.

Provides configurable filters and multi-scale extraction strategies
to improve entity extraction accuracy on generic documents.
"""

import re
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from collections import Counter


# ═══════════════════════════════════════════════════════════════════
# Common Stopwords and False Positive Patterns
# ═══════════════════════════════════════════════════════════════════

COMMON_STOPWORDS: Set[str] = {
    'the', 'and', 'or', 'but', 'a', 'an', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of',
    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
    'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'then', 'once',
    # Italian stopwords
    'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'e',
    'ed', 'o', 'od', 'ma', 'per', 'con', 'su', 'tra', 'fra',
    'in', 'di', 'da', 'a', 'che', 'chi', 'cui', 'come', 'quale',
}

# Pattern validators for common entity types
ENTITY_PATTERNS: Dict[str, re.Pattern] = {
    'date': re.compile(
        r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|'
        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|'
        r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\b',
        re.IGNORECASE
    ),
    'email': re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ),
    'phone': re.compile(
        r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
    ),
    'url': re.compile(
        r'\b(?:https?://|www\.)[^\s<>\"{}|\[\]^`]+'
    ),
    'monetary': re.compile(
        r'\b(?:\$|€|£|¥)?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY)?\b|\b\d+(?:\.\d{2})?\s*(?:dollars?|euros?|pounds?)\b',
        re.IGNORECASE
    ),
    'percentage': re.compile(
        r'\b\d{1,3}(?:\.\d{1,2})?\s*%\b'
    ),
    'number': re.compile(
        r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
    ),
}


# ═══════════════════════════════════════════════════════════════════
# Configuration Dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PostProcessorConfig:
    """Configuration for entity post-processing pipeline."""

    # Length filters
    min_entity_length: int = 2
    max_entity_length: int = 500

    # Confidence/score thresholds
    min_confidence: float = 0.3
    high_confidence_threshold: float = 0.7

    # Stopword filtering
    filter_stopwords: bool = True
    stopword_ratio_threshold: float = 0.5  # Remove if >50% stopwords

    # Pattern validation
    validate_patterns: bool = True
    pattern_strictness: str = "relaxed"  # "strict", "relaxed", "off"

    # Cross-reference boosting
    enable_coreference_boost: bool = True
    coreference_boost_factor: float = 0.15

    # Multi-scale extraction
    enable_multiscale: bool = True
    primary_threshold: float = 0.4
    secondary_threshold: float = 0.25
    secondary_labels: Optional[List[str]] = field(default_factory=list)

    # Deduplication
    deduplicate: bool = True
    dedup_strategy: str = "highest_confidence"  # "highest_confidence", "longest", "first"

    # Bucket size limits
    max_entities_per_bucket: int = 15

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'min_entity_length': self.min_entity_length,
            'max_entity_length': self.max_entity_length,
            'min_confidence': self.min_confidence,
            'high_confidence_threshold': self.high_confidence_threshold,
            'filter_stopwords': self.filter_stopwords,
            'stopword_ratio_threshold': self.stopword_ratio_threshold,
            'validate_patterns': self.validate_patterns,
            'pattern_strictness': self.pattern_strictness,
            'enable_coreference_boost': self.enable_coreference_boost,
            'coreference_boost_factor': self.coreference_boost_factor,
            'enable_multiscale': self.enable_multiscale,
            'primary_threshold': self.primary_threshold,
            'secondary_threshold': self.secondary_threshold,
            'secondary_labels': self.secondary_labels,
            'deduplicate': self.deduplicate,
            'dedup_strategy': self.dedup_strategy,
            'max_entities_per_bucket': self.max_entities_per_bucket,
        }


# ═══════════════════════════════════════════════════════════════════
# Pre-configured Settings
# ═══════════════════════════════════════════════════════════════════

# Conservative settings for high-precision extraction
CONSERVATIVE_CONFIG = PostProcessorConfig(
    min_entity_length=3,
    min_confidence=0.5,
    high_confidence_threshold=0.8,
    filter_stopwords=True,
    stopword_ratio_threshold=0.3,
    validate_patterns=True,
    pattern_strictness="strict",
    enable_coreference_boost=True,
    coreference_boost_factor=0.1,
    enable_multiscale=False,
    primary_threshold=0.5,
    secondary_threshold=0.35,
    deduplicate=True,
    max_entities_per_bucket=10,
)

# Aggressive settings for high-recall extraction
AGGRESSIVE_CONFIG = PostProcessorConfig(
    min_entity_length=2,
    min_confidence=0.25,
    high_confidence_threshold=0.6,
    filter_stopwords=True,
    stopword_ratio_threshold=0.7,
    validate_patterns=True,
    pattern_strictness="relaxed",
    enable_coreference_boost=True,
    coreference_boost_factor=0.2,
    enable_multiscale=True,
    primary_threshold=0.4,
    secondary_threshold=0.25,
    deduplicate=True,
    max_entities_per_bucket=20,
)

# Balanced settings (default)
DEFAULT_CONFIG = PostProcessorConfig()


# ═══════════════════════════════════════════════════════════════════
# Core Post-Processing Functions
# ═══════════════════════════════════════════════════════════════════

def filter_by_length(
    entities: List[Dict[str, Any]],
    min_length: int = 2,
    max_length: int = 500
) -> List[Dict[str, Any]]:
    """Filter entities by text length."""
    return [
        e for e in entities
        if min_length <= len(e.get('text', '')) <= max_length
    ]


def filter_by_stopwords(
    entities: List[Dict[str, Any]],
    stopwords: Set[str] = COMMON_STOPWORDS,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Filter out entities that are mostly stopwords."""
    filtered = []
    for entity in entities:
        text = entity.get('text', '').lower()
        words = text.split()
        if not words:
            continue

        stopword_count = sum(1 for w in words if w in stopwords)
        stopword_ratio = stopword_count / len(words)

        if stopword_ratio <= threshold:
            filtered.append(entity)

    return filtered


def validate_entity_patterns(
    entities: List[Dict[str, Any]],
    label_map: Dict[str, str],
    strictness: str = "relaxed"
) -> List[Dict[str, Any]]:
    """
    Validate entities against expected patterns for their type.

    Args:
        entities: List of entity dictionaries
        label_map: Mapping from GLiNER labels to bucket names
        strictness: "strict" (must match), "relaxed" (boost if matches), "off"
    """
    if strictness == "off":
        return entities

    validated = []
    for entity in entities:
        text = entity.get('text', '')
        label = entity.get('label', '')
        score = entity.get('score', 0.0)

        # Check if label maps to a pattern-validated bucket
        bucket = label_map.get(label, label)

        # Determine which patterns to check
        patterns_to_check = []
        if bucket in ['dates', 'date'] or 'date' in label.lower():
            patterns_to_check.append(('date', ENTITY_PATTERNS['date']))
        if bucket in ['emails', 'email'] or 'email' in label.lower():
            patterns_to_check.append(('email', ENTITY_PATTERNS['email']))
        if bucket in ['phones', 'phone'] or 'phone' in label.lower():
            patterns_to_check.append(('phone', ENTITY_PATTERNS['phone']))
        if bucket in ['urls', 'url', 'links', 'link'] or 'url' in label.lower():
            patterns_to_check.append(('url', ENTITY_PATTERNS['url']))
        if bucket in ['values', 'monetary'] or 'financial' in label.lower():
            patterns_to_check.append(('monetary', ENTITY_PATTERNS['monetary']))
        if bucket in ['percentages', 'percentage'] or 'percent' in label.lower():
            patterns_to_check.append(('percentage', ENTITY_PATTERNS['percentage']))

        if patterns_to_check:
            matches = any(
                pattern.match(text) for _, pattern in patterns_to_check
            )

            if strictness == "strict" and not matches:
                continue  # Skip non-matching entities
            elif matches:
                # Boost confidence for matching entities
                entity = entity.copy()
                entity['score'] = min(score + 0.1, 1.0)

        validated.append(entity)

    return validated


def boost_by_coreference(
    entities: List[Dict[str, Any]],
    boost_factor: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Boost confidence scores for entities appearing multiple times.

    Entities that appear multiple times in a document are more likely
    to be important and correctly identified.
    """
    # Count normalized entity occurrences
    normalized_counts: Dict[str, int] = Counter()
    for entity in entities:
        text = entity.get('text', '').lower().strip()
        normalized_counts[text] += 1

    # Apply boost based on frequency
    boosted = []
    for entity in entities:
        text = entity.get('text', '').lower().strip()
        count = normalized_counts.get(text, 1)

        entity_copy = entity.copy()
        original_score = entity.get('score', 0.0)

        # Boost: +boost_factor for each additional occurrence, max +0.3
        boost = min((count - 1) * boost_factor, 0.3)
        entity_copy['score'] = min(original_score + boost, 1.0)

        boosted.append(entity_copy)

    return boosted


def deduplicate_entities(
    entities: List[Dict[str, Any]],
    strategy: str = "highest_confidence"
) -> List[Dict[str, Any]]:
    """
    Remove duplicate entities based on normalized text.

    Args:
        entities: List of entity dictionaries
        strategy: How to handle duplicates:
            - "highest_confidence": Keep the highest scoring variant
            - "longest": Keep the longest text variant
            - "first": Keep the first occurrence
    """
    seen: Dict[str, Dict[str, Any]] = {}

    for entity in entities:
        text = entity.get('text', '').lower().strip()
        if not text:
            continue

        if text not in seen:
            seen[text] = entity
        else:
            existing = seen[text]

            if strategy == "highest_confidence":
                if entity.get('score', 0) > existing.get('score', 0):
                    seen[text] = entity
            elif strategy == "longest":
                if len(entity.get('text', '')) > len(existing.get('text', '')):
                    seen[text] = entity
            # "first" strategy keeps existing

    return list(seen.values())


def filter_by_confidence(
    entities: List[Dict[str, Any]],
    min_confidence: float
) -> List[Dict[str, Any]]:
    """Filter entities below confidence threshold."""
    return [
        e for e in entities
        if e.get('score', 0) >= min_confidence
    ]


def sort_by_confidence(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort entities by confidence score descending."""
    return sorted(entities, key=lambda x: x.get('score', 0), reverse=True)


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline Class
# ═══════════════════════════════════════════════════════════════════

class EntityPostProcessor:
    """
    Configurable post-processing pipeline for GLiNER NER results.

    Provides filtering, validation, and confidence boosting to improve
    entity extraction accuracy, especially for generic documents.
    """

    def __init__(self, config: Optional[PostProcessorConfig] = None):
        """
        Initialize the post-processor.

        Args:
            config: PostProcessorConfig instance, or None for defaults
        """
        self.config = config or DEFAULT_CONFIG

    def process(
        self,
        entities: Dict[str, List[Dict[str, Any]]],
        label_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process entities through the full pipeline.

        Args:
            entities: Dictionary mapping bucket names to entity lists
            label_map: Mapping from GLiNER labels to bucket names

        Returns:
            Filtered and processed entities
        """
        result = {}
        label_map = label_map or {}

        for bucket, bucket_entities in entities.items():
            processed = self._process_bucket(bucket_entities, label_map)
            if processed:
                result[bucket] = processed

        return result

    def _process_bucket(
        self,
        entities: List[Dict[str, Any]],
        label_map: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Process a single bucket's entities."""
        if not entities:
            return []

        # Work with copies to avoid modifying originals
        working = [e.copy() for e in entities]

        # 1. Filter by confidence
        working = filter_by_confidence(working, self.config.min_confidence)

        # 2. Filter by length
        working = filter_by_length(
            working,
            self.config.min_entity_length,
            self.config.max_entity_length
        )

        # 3. Filter stopwords
        if self.config.filter_stopwords:
            working = filter_by_stopwords(
                working,
                COMMON_STOPWORDS,
                self.config.stopword_ratio_threshold
            )

        # 4. Validate patterns
        if self.config.validate_patterns:
            working = validate_entity_patterns(
                working,
                label_map,
                self.config.pattern_strictness
            )

        # 5. Deduplicate
        if self.config.deduplicate:
            working = deduplicate_entities(working, self.config.dedup_strategy)

        # 6. Boost by coreference
        if self.config.enable_coreference_boost:
            working = boost_by_coreference(working, self.config.coreference_boost_factor)

        # 7. Final confidence filter after boosts
        working = filter_by_confidence(working, self.config.min_confidence)

        # 8. Sort by confidence
        working = sort_by_confidence(working)

        # 9. Limit per bucket
        working = working[:self.config.max_entities_per_bucket]

        return working

    def process_multiscale(
        self,
        primary_entities: Dict[str, List[Dict[str, Any]]],
        secondary_entities: Dict[str, List[Dict[str, Any]]],
        label_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Merge and process results from multi-scale extraction.

        Combines high-confidence primary extraction with lower-confidence
        secondary extraction, deduplicating and ranking results.

        Args:
            primary_entities: High-confidence extraction results
            secondary_entities: Lower-confidence secondary pass results
            label_map: Mapping from GLiNER labels to bucket names

        Returns:
            Merged and processed entities
        """
        label_map = label_map or {}

        # Process primary with strict config
        primary_processed = self.process(primary_entities, label_map)

        # Temporarily relax threshold for secondary
        original_threshold = self.config.min_confidence
        self.config.min_confidence = self.config.secondary_threshold

        # Process secondary
        secondary_processed = self.process(secondary_entities, label_map)

        # Restore threshold
        self.config.min_confidence = original_threshold

        # Merge buckets
        merged: Dict[str, List[Dict[str, Any]]] = {}
        all_buckets = set(primary_processed.keys()) | set(secondary_processed.keys())

        for bucket in all_buckets:
            primary = primary_processed.get(bucket, [])
            secondary = secondary_processed.get(bucket, [])

            # Mark source for debugging
            for e in primary:
                e['source'] = 'primary'
            for e in secondary:
                e['source'] = 'secondary'

            # Combine and deduplicate
            combined = primary + secondary
            combined = deduplicate_entities(combined, self.config.dedup_strategy)
            combined = sort_by_confidence(combined)
            combined = combined[:self.config.max_entities_per_bucket]

            if combined:
                merged[bucket] = combined

        return merged


# ═══════════════════════════════════════════════════════════════════
# Helper Functions for Ingest Integration
# ═══════════════════════════════════════════════════════════════════

def create_processor(config_name: str = "default") -> EntityPostProcessor:
    """
    Create a post-processor with pre-configured settings.

    Args:
        config_name: One of "default", "conservative", "aggressive"

    Returns:
        Configured EntityPostProcessor instance
    """
    configs = {
        "default": DEFAULT_CONFIG,
        "conservative": CONSERVATIVE_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config '{config_name}'. Use: {list(configs.keys())}")

    return EntityPostProcessor(configs[config_name])


def apply_post_processing(
    entities: Dict[str, List[Dict[str, Any]]],
    label_map: Dict[str, str],
    config: Optional[PostProcessorConfig] = None
) -> Dict[str, List[str]]:
    """
    Convenience function for basic use case.

    Processes entities and returns simple text lists (backwards compatible).

    Args:
        entities: Raw entity extraction results
        label_map: Label to bucket mapping
        config: Optional custom config

    Returns:
        Dictionary mapping buckets to sorted unique entity texts
    """
    processor = EntityPostProcessor(config)
    processed = processor.process(entities, label_map)

    # Convert to simple text lists for backwards compatibility
    return {
        bucket: sorted(set(e['text'] for e in bucket_entities))
        for bucket, bucket_entities in processed.items()
        if bucket_entities
    }
