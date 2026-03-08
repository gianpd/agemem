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

import yaml
import hashlib
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from docling.document_converter import DocumentConverter

CORPUS = Path("corpus")

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
def parse_pdf(pdf_path: Path) -> tuple[str, List[str]]:
    """Returns (full_markdown, list_of_section_titles)."""
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()
    sections = re.findall(r'^#{1,2}\s+(.+)$', markdown, re.MULTILINE)
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
) -> str:
    """
    Ingest a PDF document into the corpus.

    Args:
        pdf_path: Path to the PDF file
        doc_type: Document type/category
        labels_arg: Label configuration (built-in name or path:key)

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
    markdown, sections = parse_pdf(pdf)

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

    args = parser.parse_args()
    ingest(args.pdf_path, args.doc_type, args.labels_arg)


if __name__ == "__main__":
    main()
