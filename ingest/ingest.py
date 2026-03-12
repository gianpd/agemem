# ingest.py
# pip install docling pyyaml gliner
#
# Usage:
#   python ingest.py report.pdf [doc_type]
#   python ingest.py contracts/acme.pdf contract
#   python ingest.py bandi/gara_2024.pdf bando
#
# With custom labels:
#   python ingest.py paper.pdf research --labels research
#   python ingest.py bando.pdf bando --labels edilizia
#   python ingest.py doc.pdf custom --labels /path/to/config.yaml:medical

import os
import gc
import yaml
import hashlib
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set cache paths BEFORE any docling import to ensure models are stored locally
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("DOCLING_ARTIFACTS_PATH", os.path.expanduser("~/.cache/docling/models"))

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
    TableFormerMode,
    TableStructureOptions
)
from docling.datamodel.base_models import InputFormat

try:
    from .gliner_labels import get_builtin_labels, list_builtin_labels, BUILTIN_LABEL_SETS
except ImportError:
    # When running as script directly
    from gliner_labels import get_builtin_labels, list_builtin_labels, BUILTIN_LABEL_SETS

# Try to import torch for GPU memory management
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore

CORPUS = Path("corpus")

# ═══════════════════════════════════════════════════════════════
# Docling Converter — Optimized singleton with local model caching
# ═══════════════════════════════════════════════════════════════

_CONVERTER: DocumentConverter | None = None  # module-level singleton
_CONVERTER_PARAMS: dict = {}  # track params used to build converter


def _ensure_models_downloaded() -> None:
    """Pre-download docling models to ensure they're available offline."""
    try:
        from docling.utils.model_downloader import download_models
        download_models()
        print("      Models verified (cached locally)")
    except Exception as e:
        print(f"      [warn] Could not verify model download: {e}")
        print("      Models will be downloaded on first use if needed.")


def _build_optimized_converter(
    do_ocr: bool = False,
    enable_table_structure: bool = True,
    fast_mode: bool = False,
) -> DocumentConverter:
    """
    Build an optimized DocumentConverter for local execution.

    Optimized for: CPU + optional GPU, with models cached locally.

    Args:
        do_ocr: Enable OCR for scanned PDFs
        enable_table_structure: Enable table structure recognition
        fast_mode: Use faster but less accurate table mode
    """
    accelerator = AcceleratorOptions(
        num_threads=min(8, os.cpu_count() or 4),
        device=AcceleratorDevice.AUTO,  # uses CUDA if available, else CPU
    )

    # Use FAST mode for table structure by default (ACCURATE is too slow for large docs)
    table_mode = TableFormerMode.FAST if fast_mode else TableFormerMode.ACCURATE

    pipeline_options = PdfPipelineOptions(
        accelerator_options=accelerator,
        do_ocr=do_ocr,
        do_table_structure=enable_table_structure,
        table_structure_options=TableStructureOptions(mode=table_mode),
        images_scale=1.5,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter


def _get_converter(
    do_ocr: bool = False,
    enable_table_structure: bool = True,
    fast_mode: bool = False,
) -> DocumentConverter:
    """
    Lazy singleton with param caching — rebuild only if params change.

    This ensures the converter is reused when possible but correctly
    handles parameter changes (e.g., OCR on/off).
    """
    global _CONVERTER, _CONVERTER_PARAMS

    current_params = {
        "do_ocr": do_ocr,
        "enable_table_structure": enable_table_structure,
        "fast_mode": fast_mode,
    }

    if _CONVERTER is None or _CONVERTER_PARAMS != current_params:
        if _CONVERTER is not None:
            print("      [info] Rebuilding converter with new parameters...")
        _CONVERTER = _build_optimized_converter(**current_params)
        _CONVERTER_PARAMS = current_params

    return _CONVERTER


def _clear_gpu_cache() -> None:
    """Clear GPU memory cache after conversion."""
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════
# Model Cache Diagnostics
# ═══════════════════════════════════════════════════════════════

def audit_model_cache() -> None:
    """
    Audit the docling model cache and print diagnostic information.

    Shows which models are present, their sizes, and relevant environment variables.
    """
    import pathlib

    CACHE_DIRS = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/docling/models"),
        os.path.expanduser("~/.cache/docling"),
        os.environ.get("HF_HOME", ""),
    ]

    DOCLING_MODEL_REPOS = [
        "ds4sd--docling-models",       # Layout + TableFormer
        "ds4sd--DocLayNet",            # Layout detection weights
        "ds4sd--TableFormer",          # Table structure
        "easyocr",                     # OCR (only if scanned PDF)
    ]

    print("=== Docling Model Cache Audit ===\n")
    total_size = 0
    for base in CACHE_DIRS:
        if not base or not os.path.exists(base):
            print(f"  [MISSING]  {base}")
            continue
        p = pathlib.Path(base)
        dirs = [d.name for d in p.iterdir() if d.is_dir()]
        matched = [d for d in dirs if any(m in d for m in DOCLING_MODEL_REPOS)]
        print(f"  [FOUND]    {base}")
        if matched:
            for m in matched:
                size = sum(f.stat().st_size for f in (p/m).rglob("*") if f.is_file())
                total_size += size
                print(f"             ✓ {m}  ({size/1e9:.2f} GB)")
        else:
            print("             — no docling models found here")

    print(f"\n  Total cached: {total_size/1e9:.2f} GB")

    print("\n=== Relevant Environment Variables ===")
    for k in ["HF_HOME", "TRANSFORMERS_CACHE", "DOCLING_ARTIFACTS_PATH",
              "CUDA_VISIBLE_DEVICES", "TORCH_HOME"]:
        print(f"  {k} = {os.environ.get(k, '(not set)')}")

    # Check GPU availability
    print("\n=== GPU Status ===")
    if _HAS_TORCH:
        if torch.cuda.is_available():
            print(f"  CUDA available: Yes")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  CUDA available: No (will use CPU)")
    else:
        print("  PyTorch not installed (CPU-only mode)")


# ═══════════════════════════════════════════════════════════════
# OCR Detection — Automatic detection of scanned vs native PDFs
# ═══════════════════════════════════════════════════════════════

def _needs_ocr(pdf_path: Path, sample_pages: int = 5) -> bool:
    """
    Detect if PDF appears to be scanned (no text layer).

    Returns True if average text per page is very low.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        text_chars = 0
        pages_checked = min(sample_pages, doc.page_count)
        for i in range(pages_checked):
            text_chars += len(doc[i].get_text())
        doc.close()
        avg_chars_per_page = text_chars / pages_checked if pages_checked > 0 else 0
        return avg_chars_per_page < 100  # threshold: scanned if very few chars
    except ImportError:
        print("      [info] PyMuPDF not available for OCR detection, assuming native PDF")
        return False
    except Exception as e:
        print(f"      [warn] OCR detection failed: {e}, assuming native PDF")
        return False

# ── GLiNER entity extractor (zero-shot, no training needed) ───
try:
    from gliner import GLiNER
    # Initialize GLiNER with explicit tokenizer max_length to avoid truncation warnings
    _ner = GLiNER.from_pretrained(
        "urchade/gliner_medium-v2.1",
        max_length=384  # Explicitly set tokenizer max_length to match model limit
    )
    NER_BACKEND = "gliner"
except ImportError:
    _ner = None
    NER_BACKEND = "none"
    print("[warn] gliner not installed — entity extraction disabled.")
    print("       pip install gliner")
except Exception as e:
    _ner = None
    NER_BACKEND = "none"
    print(f"[warn] GLiNER initialization failed: {e}")
    print("       Entity extraction disabled.")


# ═══════════════════════════════════════════════════════════════
# Label Configuration Loading
# ═══════════════════════════════════════════════════════════════

def load_labels_from_yaml(config_path: Path, key: str) -> Dict[str, Any]:
    """Load a label set from a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if key not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Label set '{key}' not found in {config_path}. Available: {available}")

    label_config = config[key]

    return {
        "labels": label_config["labels"],
        "label_map": label_config["label_map"],
        "buckets": {bucket: [] for bucket in label_config["buckets"]},
        "description": label_config.get("description", "Custom label set"),
    }


def load_labels(labels_arg: Optional[str]) -> Dict[str, Any]:
    """
    Load label configuration from argument.

    Args:
        labels_arg: Can be:
            - None: use default 'edilizia' built-in
            - 'edilizia', 'research', 'legal': use built-in
            - 'path/to/config.yaml:key': load from YAML file

    Returns:
        Dictionary with 'labels', 'label_map', 'buckets', 'description'
    """

    if labels_arg is None:
        # Default to edilizia
        return get_builtin_labels("edilizia")

    # Check if it's a built-in label set
    if labels_arg in BUILTIN_LABEL_SETS:
        return get_builtin_labels(labels_arg)

    # Check if it's a path:key format
    if ":" in labels_arg:
        parts = labels_arg.rsplit(":", 1)
        config_path = Path(parts[0]).expanduser().resolve()
        key = parts[1]
        return load_labels_from_yaml(config_path, key)

    # Check if it's a YAML file without key (assume 'default' key)
    potential_path = Path(labels_arg).expanduser()
    if potential_path.exists() and potential_path.suffix in ('.yaml', '.yml'):
        return load_labels_from_yaml(potential_path, "default")

    # Unknown format
    raise ValueError(
        f"Invalid --labels argument: {labels_arg}\n"
        f"Use a built-in label set ({', '.join(BUILTIN_LABEL_SETS.keys())}), "
        f"or 'path/to/config.yaml:key' for custom labels."
    )


# ═══════════════════════════════════════════════════════════════
# Global label configuration (set during ingest)
# ═══════════════════════════════════════════════════════════════
_current_label_config: Optional[Dict[str, Any]] = None


def get_current_labels() -> Dict[str, Any]:
    """Get the current label configuration."""
    if _current_label_config is None:
        raise RuntimeError("Label configuration not set. Call load_labels() first.")
    return _current_label_config


# GLiNER's underlying transformer has a 384 token limit.
# Long sentences get truncated, missing entities at the tail.
# We split at natural boundaries to preserve recall.
GLINER_MAX_TOKENS = 384
# Approximate chars per token (conservative for Italian legal text)
CHARS_PER_TOKEN_ESTIMATE = 4
# Safe sentence length in characters before splitting
MAX_SENTENCE_CHARS = GLINER_MAX_TOKENS * CHARS_PER_TOKEN_ESTIMATE  # ~1536

# Split boundaries in order of preference (less semantic disruption)
SENTENCE_SPLIT_BOUNDARIES = [
    ';',      # Italian legal text uses semicolons heavily
    ',',      # Secondary break point
    ' e ',    # 'and' conjunction (Italian)
    ' ed ',   # 'and' before vowels (Italian)
    ' oppure ',  # 'or'
]


def _split_long_sentence(
    sentence: str, 
    max_chars: int = MAX_SENTENCE_CHARS,
    overlap_chars: int = 100
) -> List[str]:
    """
    Split a long sentence at natural boundaries for GLiNER processing.

    GLiNER's tokenizer truncates at 384 tokens, so we proactively split
    long sentences at semicolons, commas, or conjunctions to preserve entity recall.
    
    Uses sliding window with overlap for very long sentences to ensure entities
    at segment boundaries are not missed.

    Args:
        sentence: A single sentence (no newlines)
        max_chars: Maximum characters per segment (~4 chars/token)
        overlap_chars: Number of characters to overlap between segments

    Returns:
        List of sentence segments, each within token limit
    """
    if len(sentence) <= max_chars:
        return [sentence]

    segments = []
    remaining = sentence
    start_pos = 0

    while start_pos < len(sentence):
        # Calculate end position for this segment
        end_pos = min(start_pos + max_chars, len(sentence))
        
        # If we're not at the end, look for a good split point
        if end_pos < len(sentence):
            # Look for split point within the safe zone (from start_pos to end_pos)
            search_region = sentence[start_pos:end_pos]
            
            best_split = -1
            for boundary in SENTENCE_SPLIT_BOUNDARIES:
                # Find last occurrence of boundary in search region
                idx = search_region.rfind(boundary)
                if idx > max_chars * 0.3:  # Don't split too early (at least 30% of max)
                    split_at = start_pos + idx + len(boundary)
                    # Prefer semicolons/conjunctions — keep delimiter with left segment
                    if boundary in (';', ','):
                        best_split = split_at
                        break  # semicolon/comma are ideal, stop here
                    elif best_split == -1:
                        best_split = split_at
            
            if best_split == -1:
                # No good boundary found; hard split at max_chars
                best_split = end_pos
            
            # Add segment
            segment = sentence[start_pos:best_split].strip()
            if segment:
                segments.append(segment)
            
            # Move start position with overlap
            start_pos = max(start_pos + 1, best_split - overlap_chars)
        else:
            # Last segment
            segment = sentence[start_pos:].strip()
            if segment:
                segments.append(segment)
            break

    return [s for s in segments if s.strip()]


def _split_text_for_gliner(text: str, max_chars: int = MAX_SENTENCE_CHARS) -> List[str]:
    """
    Split text into GLiNER-safe segments, respecting sentence boundaries.

    First splits on sentences, then further splits long sentences at
    semicolons/commas/conjunctions to stay within token limits.

    Args:
        text: Full text to process
        max_chars: Maximum characters per segment

    Returns:
        List of text segments safe for GLiNER inference
    """
    # Split into sentences (simple but effective for this use case)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    segments = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            segments.append(sentence)
        else:
            # Split long sentence at natural boundaries
            segments.extend(_split_long_sentence(sentence, max_chars))

    return [s for s in segments if s.strip()]


def _find_segment_offset(text: str, segment: str, start_search: int = 0) -> int:
    """
    Find the offset of a segment in the original text.
    
    Args:
        text: Original text
        segment: Segment to find
        start_search: Position to start searching from
        
    Returns:
        Offset of segment in original text, or -1 if not found
    """
    idx = text.find(segment, start_search)
    return idx if idx != -1 else start_search


def extract_entities(
    text: str, 
    label_config: Optional[Dict[str, Any]] = None,
    return_positions: bool = False
) -> Dict[str, List[str]] | Dict[str, List[Dict[str, Any]]]:
    """
    Extract named entities from text using GLiNER with the configured labels.

    Uses sentence-aware splitting to avoid truncation at GLiNER's 384 token limit.
    Entity spans are re-mapped to original text offsets when return_positions=True.

    Args:
        text: The text to analyze
        label_config: Optional label configuration (uses current if not provided)
        return_positions: If True, return entities with their positions in original text

    Returns:
        Dictionary of entity buckets with extracted values (or entities with positions)
    """
    config = label_config or get_current_labels()
    labels = config["labels"]
    label_map = config["label_map"]
    buckets = {k: set() for k in config["buckets"].keys()}
    
    # For position tracking
    entities_with_positions: Dict[str, List[Dict[str, Any]]] = {k: [] for k in config["buckets"].keys()}

    if NER_BACKEND == "gliner" and _ner is not None:
        segments = _split_text_for_gliner(text)
        
        # Track position in original text for re-mapping
        current_offset = 0

        for i, segment in enumerate(segments):
            if len(segments) > 1:
                print(f"      Processing segment {i + 1}/{len(segments)} ({len(segment):,} chars)...")

            # Find segment position in original text for offset re-mapping
            segment_offset = _find_segment_offset(text, segment, current_offset)
            current_offset = segment_offset + len(segment)

            hits = _ner.predict_entities(segment, labels, threshold=0.4)
            for h in hits:
                bucket = label_map.get(h["label"])
                if bucket and len(h["text"].strip()) > 2:
                    entity_text = h["text"].strip()
                    buckets[bucket].add(entity_text)
                    
                    # Re-map entity position to original text
                    if return_positions:
                        # Adjust start/end positions to original text offset
                        original_start = segment_offset + h.get("start", 0)
                        original_end = segment_offset + h.get("end", len(entity_text))
                        
                        entities_with_positions[bucket].append({
                            "text": entity_text,
                            "label": h["label"],
                            "start": original_start,
                            "end": original_end,
                            "score": h.get("score", 0.0)
                        })

    # Convert sets to sorted lists and cap each bucket
    if return_positions:
        # Return entities with positions, deduplicated by text
        result = {}
        for bucket, entities in entities_with_positions.items():
            if entities:
                # Deduplicate by text, keeping highest score
                seen = {}
                for entity in entities:
                    text_key = entity["text"]
                    if text_key not in seen or entity["score"] > seen[text_key]["score"]:
                        seen[text_key] = entity
                result[bucket] = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:15]
        return result
    else:
        return {k: sorted(v)[:15] for k, v in buckets.items() if v}


def detect_doc_date(text: str, entities: Dict[str, List[str]]) -> Optional[str]:
    """Detect document date from text or extracted entities."""
    iso = re.search(r'\b(20\d{2}-\d{2}-\d{2})\b', text)
    if iso:
        return iso.group(1)
    if entities.get("dates"):
        return entities["dates"][0]
    return None


def _guess_title(markdown: str, fallback: str) -> str:
    """Extract title from markdown or use fallback."""
    m = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
    return m.group(1).strip() if m else fallback.replace('_', ' ').title()


# ═══════════════════════════════════════════════════════════════
# 1. PARSE — Docling → full markdown + section list
# ═══════════════════════════════════════════════════════════════
def parse_pdf(
    pdf_path: Path,
    auto_detect_ocr: bool = True,
    force_ocr: bool = False,
    fast_mode: bool = True,  # Default to FAST for reasonable performance
    disable_tables: bool = False,
) -> tuple[str, List[str]]:
    """
    Returns (full_markdown, list_of_section_titles).

    Args:
        pdf_path: Path to the PDF file
        auto_detect_ocr: Automatically detect if OCR is needed (default: True)
        force_ocr: Force OCR on for scanned PDFs (overrides auto_detect)
        fast_mode: Use fast table mode (default: True). ACCURATE mode is very slow.
        disable_tables: Disable table structure recognition entirely (fastest)
    """
    # Determine if OCR is needed
    do_ocr = force_ocr
    if auto_detect_ocr and not force_ocr:
        print("      Auto-detecting PDF type...")
        do_ocr = _needs_ocr(pdf_path)
        print(f"      OCR: {'enabled' if do_ocr else 'disabled'} (native PDF detected)")

    # Ensure models are downloaded before first conversion
    _ensure_models_downloaded()

    # Get optimized converter (creates singleton on first call)
    print("      Initializing converter...")
    converter = _get_converter(
        do_ocr=do_ocr,
        enable_table_structure=not disable_tables,
        fast_mode=fast_mode,
    )

    print(f"      Converting PDF (this may take several minutes for large documents)...")
    print(f"      Table mode: {'FAST' if fast_mode else 'ACCURATE'}, Tables: {'enabled' if not disable_tables else 'disabled'}")

    result = converter.convert(str(pdf_path))

    print("      Exporting to markdown...")
    markdown = result.document.export_to_markdown()
    sections = re.findall(r'^#{1,2}\s+(.+)$', markdown, re.MULTILINE)

    # Clear GPU cache after conversion
    _clear_gpu_cache()

    return markdown, sections


# ═══════════════════════════════════════════════════════════════
# 2. WRITE — YAML frontmatter + full markdown body
# ═══════════════════════════════════════════════════════════════
def write_document(
    pdf: Path,
    markdown: str,
    sections: List[str],
    entities: Dict[str, List[str]],
    doc_type: str,
    label_config: Dict[str, Any],
) -> Path:
    """Write document with frontmatter to corpus directory."""
    # Hash-safe doc_id: stem + 6-char md5 to prevent collisions
    raw_bytes = pdf.read_bytes()
    safe_stem = re.sub(r'\W+', '_', pdf.stem.lower()).strip('_')
    short_hash = hashlib.md5(raw_bytes).hexdigest()[:6]
    doc_id = f"{safe_stem}_{short_hash}"

    src_hash = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()[:16]
    doc_date = detect_doc_date(markdown, entities)
    page_count = len(re.findall(r'<!-- page \d+ -->', markdown)) or None

    frontmatter = {
        # ── identity ──────────────────────────────────────────
        "doc_id": doc_id,
        "doc_title": _guess_title(markdown, pdf.stem),
        "doc_type": doc_type,
        "source_file": str(pdf),
        "source_hash": src_hash,
        "doc_date": doc_date,
        "ingested_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        # ── NER configuration ─────────────────────────────────
        "ner_config": label_config.get("description", "custom"),
        # ── named entities (primary search surface) ───────────
        "entities": entities,
        # ── structure ─────────────────────────────────────────
        "page_count": page_count,
        "has_tables": bool(re.search(r'\|.+\|.+\|', markdown)),
        "has_figures": bool(re.search(r'(figure|fig\.)\s*\d+', markdown, re.I)),
        "has_code": "```" in markdown,
        "language": "it",
        "sections": sections[:25],
    }

    CORPUS.mkdir(parents=True, exist_ok=True)
    out_path = CORPUS / f"{doc_id}.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(yaml.dump(frontmatter, allow_unicode=True,
                          sort_keys=False, default_flow_style=False))
        f.write("---\n\n")
        f.write(markdown.strip())
        f.write("\n")

    return out_path


# ── index ──────────────────────────────────────────────────────
def update_index(doc_id: str, title: str, doc_type: str,
                 doc_date: Optional[str], filepath: Path):
    """Update the corpus index with document metadata."""
    idx_path = CORPUS / "_index.yaml"
    index: Dict = {}
    if idx_path.exists():
        with open(idx_path) as f:
            index = yaml.safe_load(f) or {}

    index[doc_id] = {
        "title": title,
        "type": doc_type,
        "date": doc_date,
        "file": str(filepath),
        "added_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(idx_path, "w") as f:
        yaml.dump(index, f, allow_unicode=True, sort_keys=False)


# ═══════════════════════════════════════════════════════════════
# 4. INGEST — orchestrate
# ═══════════════════════════════════════════════════════════════
def ingest(
    pdf_path: str,
    doc_type: str = "document",
    labels_arg: Optional[str] = None,
    auto_detect_ocr: bool = True,
    force_ocr: bool = False,
    fast_mode: bool = True,
    disable_tables: bool = False,
) -> str:
    """
    Ingest a PDF document into the corpus.

    Args:
        pdf_path: Path to the PDF file
        doc_type: Document type/category
        labels_arg: Label configuration (built-in name or path:key)
        auto_detect_ocr: Automatically detect if PDF needs OCR
        force_ocr: Force OCR on (for known scanned PDFs)
        fast_mode: Use fast table mode (default: True)
        disable_tables: Disable table structure recognition (fastest)

    Returns:
        Document ID string
    """
    global _current_label_config

    pdf = Path(pdf_path)
    if not pdf.exists():
        print(f"[error] File not found: {pdf_path}")
        sys.exit(1)

    # Load label configuration
    print(f"[0/4] Loading labels configuration...")
    _current_label_config = load_labels(labels_arg)
    print(f"      Using: {_current_label_config['description']}")
    print(f"      Labels: {len(_current_label_config['labels'])} entity types")

    print(f"[1/4] Parsing    {pdf.name}  (docling) ...")
    markdown, sections = parse_pdf(
        pdf,
        auto_detect_ocr=auto_detect_ocr,
        force_ocr=force_ocr,
        fast_mode=fast_mode,
        disable_tables=disable_tables,
    )

    print(f"[2/4] Extracting entities  ({NER_BACKEND}) ...")
    entities = extract_entities(markdown)

    print(f"[3/4] Writing markdown ...")
    out_path = write_document(pdf, markdown, sections, entities, doc_type, _current_label_config)

    doc_id = out_path.stem
    title = _guess_title(markdown, pdf.stem)
    doc_date = detect_doc_date(markdown, entities)

    print(f"[4/4] Updating   _index.yaml ...")
    update_index(doc_id, title, doc_type, doc_date, out_path)

    print(f"\n✓  {out_path}  ({len(markdown):,} chars, {len(sections)} sections)")
    print(f"   doc_id : {doc_id}")
    print(f"   entities found : { {k: len(v) for k, v in entities.items()} }")

    return doc_id


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the corpus with NER extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s report.pdf
  %(prog)s contracts/acme.pdf contract --labels legal
  %(prog)s papers/ml_paper.pdf research --labels research
  %(prog)s bandi/gara.pdf bando --labels edilizia
  %(prog)s doc.pdf custom --labels /path/to/my_labels.yaml:medical
  %(prog)s scanned.pdf document --ocr          # force OCR for scanned PDFs

Built-in label sets:
  edilizia  - Italian construction and public tenders
  research  - Scientific papers and academic publications
  legal     - Legal documents and contracts

For custom labels, create a YAML file with the same structure as
ingest/gliner_config.yaml and reference it as 'path/to/file.yaml:key'.
        """
    )
    parser.add_argument("pdf_path", nargs="?", default=None, help="Path to the PDF file to ingest")
    parser.add_argument(
        "doc_type",
        nargs="?",
        default="document",
        help="Document type/category (default: document)"
    )
    parser.add_argument(
        "--labels",
        dest="labels_arg",
        default=None,
        help=(
            "Label configuration to use. Can be: "
            "(1) a built-in name (edilizia, research, legal), "
            "(2) 'path/to/config.yaml:key' for custom labels"
        )
    )
    parser.add_argument(
        "--ocr",
        dest="force_ocr",
        action="store_true",
        default=False,
        help="Force OCR on (for scanned/image-based PDFs)"
    )
    parser.add_argument(
        "--no-auto-ocr",
        dest="auto_detect_ocr",
        action="store_false",
        default=True,
        help="Disable automatic OCR detection (assume native PDF)"
    )
    parser.add_argument(
        "--verify-models",
        dest="verify_models",
        action="store_true",
        default=False,
        help="Verify docling models are downloaded and exit"
    )
    parser.add_argument(
        "--audit",
        dest="audit_cache",
        action="store_true",
        default=False,
        help="Audit model cache and show diagnostic information"
    )
    parser.add_argument(
        "--accurate-tables",
        dest="accurate_tables",
        action="store_true",
        default=False,
        help="Use ACCURATE table mode (slower but better tables). Default: FAST mode"
    )
    parser.add_argument(
        "--disable-tables",
        dest="disable_tables",
        action="store_true",
        default=False,
        help="Disable table structure recognition entirely (fastest option)"
    )

    args = parser.parse_args()

    if args.audit_cache:
        audit_model_cache()
        return

    if args.verify_models:
        print("Verifying docling models are cached locally...")
        _ensure_models_downloaded()
        return
    
    if args.pdf_path is None:
        parser.error("pdf_path is required unless using --verify-models or --audit")

    ingest(
        args.pdf_path,
        args.doc_type,
        args.labels_arg,
        auto_detect_ocr=args.auto_detect_ocr,
        force_ocr=args.force_ocr,
        fast_mode=not args.accurate_tables,  # FAST by default
        disable_tables=args.disable_tables,
    )


if __name__ == "__main__":
    main()
