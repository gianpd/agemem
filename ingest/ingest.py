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
)
from docling.datamodel.base_models import InputFormat

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


def _ensure_models_downloaded() -> None:
    """Pre-download docling models to ensure they're available offline."""
    try:
        from docling.utils.model_downloader import download_models_hf
        download_models_hf(force=False)  # skip if already present
        print("      Models verified (cached locally)")
    except Exception as e:
        print(f"      [warn] Could not verify model download: {e}")
        print("      Models will be downloaded on first use if needed.")


def _build_optimized_converter(
    do_ocr: bool = False,
    enable_table_structure: bool = True,
) -> DocumentConverter:
    """
    Build an optimized DocumentConverter for local execution.

    Optimized for: CPU + optional GPU, with models cached locally.
    """
    accelerator = AcceleratorOptions(
        num_threads=min(8, os.cpu_count() or 4),
        device=AcceleratorDevice.AUTO,  # uses CUDA if available, else CPU
    )

    pipeline_options = PdfPipelineOptions(
        accelerator_options=accelerator,
        do_ocr=do_ocr,
        do_table_structure=enable_table_structure,
        table_structure_options={"mode": TableFormerMode.ACCURATE},
        images_scale=1.5,  # good balance: 108 DPI (1.5 * 72)
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
) -> DocumentConverter:
    """Lazy singleton — build once, reuse across calls."""
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = _build_optimized_converter(do_ocr, enable_table_structure)
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
    _ner = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    NER_BACKEND = "gliner"
except ImportError:
    _ner = None
    NER_BACKEND = "none"
    print("[warn] gliner not installed — entity extraction disabled.")
    print("       pip install gliner")


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
    # Import built-in labels
    from gliner_labels.gliner_labels import get_builtin_labels, BUILTIN_LABEL_SETS

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


# GLiNER has a ~12k character limit per batch
GLINER_CHUNK_SIZE = 12_000
GLINER_CHUNK_OVERLAP = 200


def _chunk_text(text: str, chunk_size: int = GLINER_CHUNK_SIZE, overlap: int = GLINER_CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks for batch processing."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            break_point = text.rfind(' ', start + chunk_size - 100, end)
            if break_point > start:
                end = break_point

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else end
        if start <= 0 or (chunks and start <= len(chunks[-1])):
            start = end

    return chunks


def extract_entities(text: str, label_config: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """
    Extract named entities from text using GLiNER with the configured labels.

    Args:
        text: The text to analyze
        label_config: Optional label configuration (uses current if not provided)

    Returns:
        Dictionary of entity buckets with extracted values
    """
    config = label_config or get_current_labels()
    labels = config["labels"]
    label_map = config["label_map"]
    buckets = {k: set() for k in config["buckets"].keys()}

    if NER_BACKEND == "gliner" and _ner is not None:
        chunks = _chunk_text(text, GLINER_CHUNK_SIZE, GLINER_CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"      Processing chunk {i + 1}/{len(chunks)} ({len(chunk):,} chars)...")

            hits = _ner.predict_entities(chunk, labels, threshold=0.4)
            for h in hits:
                bucket = label_map.get(h["label"])
                if bucket and len(h["text"].strip()) > 2:
                    buckets[bucket].add(h["text"].strip())

    # Convert sets to sorted lists and cap each bucket
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
) -> tuple[str, List[str]]:
    """
    Returns (full_markdown, list_of_section_titles).

    Args:
        pdf_path: Path to the PDF file
        auto_detect_ocr: Automatically detect if OCR is needed (default: True)
        force_ocr: Force OCR on for scanned PDFs (overrides auto_detect)
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
    converter = _get_converter(do_ocr=do_ocr)

    result = converter.convert(str(pdf_path))
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
) -> str:
    """
    Ingest a PDF document into the corpus.

    Args:
        pdf_path: Path to the PDF file
        doc_type: Document type/category
        labels_arg: Label configuration (built-in name or path:key)
        auto_detect_ocr: Automatically detect if PDF needs OCR
        force_ocr: Force OCR on (for known scanned PDFs)

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
    markdown, sections = parse_pdf(pdf, auto_detect_ocr=auto_detect_ocr, force_ocr=force_ocr)

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
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest")
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

    args = parser.parse_args()

    if args.audit_cache:
        audit_model_cache()
        return

    if args.verify_models:
        print("Verifying docling models are cached locally...")
        _ensure_models_downloaded()
        return

    ingest(
        args.pdf_path,
        args.doc_type,
        args.labels_arg,
        auto_detect_ocr=args.auto_detect_ocr,
        force_ocr=args.force_ocr,
    )


if __name__ == "__main__":
    main()
