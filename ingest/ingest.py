# ingest.py
# pip install docling pyyaml gliner
#
# Usage:
#   python ingest.py report.pdf [doc_type]
#   python ingest.py contracts/acme.pdf contract
#   python ingest.py bandi/gara_2024.pdf bando

import yaml, hashlib, re, sys
from pathlib import Path
from datetime import datetime
from docling.document_converter import DocumentConverter

CORPUS = Path("corpus")

# ── GLiNER entity extractor (zero-shot, no training needed) ───
# Replaces spacy en_core_web_sm — handles niche industry entities,
# non-English names, and custom label sets out of the box.
try:
    from gliner import GLiNER
    _ner = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    NER_BACKEND = "gliner"
except ImportError:
    _ner = None
    NER_BACKEND = "none"
    print("[warn] gliner not installed — entity extraction disabled.")
    print("       pip install gliner")

GLINER_LABELS = [
    # ── Base entities ─────────────────────────────────────────
    "person",
    "organization",
    "date",
    "financial value",
    "location",
    "articles",

    # ── Gara e Appalto (Tender/Contract) ──────────────────────
    "tender code",           # CIG, CUP
    "tender type",           # asta pubblica, trattativa privata
    "contract type",         # appalto, concessione
    "award criterion",       # offerta più vantaggiosa, prezzo più basso

    # ── Soggetti e Ruoli (Subjects/Roles) ─────────────────────
    "public administration", # Comune, Regione, ASL
    "professional role",     # Direttore Lavori, RUP
    "professional title",    # Ingegnere, Architetto, Geometra
    "contractor",            # Impresa esecutrice

    # ── Opere e Interventi (Works) ───────────────────────────
    "work type",             # ristrutturazione, nuova costruzione
    "building type",         # scuola, ospedale, ponte
    "construction material", # cemento, acciaio
    "construction phase",    # progettazione, cantiere, collaudo

    # ── Tecnico-Normativo (Technical/Regulatory) ─────────────
    "technical standard",    # UNI, ISO, Eurocodice
    "law or regulation",     # D.Lgs., D.M., Legge
    "safety standard",       # D.Lgs. 81/2008
    "permit type",           # concessione, SCIA
    "classification code",   # CPV, ATECO

    # ── Finanziario (Financial) ──────────────────────────────
    "budget category",       # base d'asta, oneri sicurezza
    "funding source",        # PNRR, fondi europei
    "guarantee",             # cauzione, polizza fideiussoria

    # ── Temporal ─────────────────────────────────────────────
    "deadline",              # termine presentazione
    "duration",              # giorni, mesi
]

LABEL_MAP = {
    # ── Base entities ─────────────────────────────────────────
    "person":                 "people",
    "organization":           "orgs",
    "date":                   "dates",
    "financial value":        "values",
    "location":               "locations",
    "articles":               "articles",

    # ── Gara e Appalto ───────────────────────────────────────
    "tender code":            "codes",
    "tender type":            "gara",
    "contract type":          "gara",
    "award criterion":        "gara",

    # ── Soggetti e Ruoli ─────────────────────────────────────
    "public administration":  "orgs",
    "professional role":      "roles",
    "professional title":     "roles",
    "contractor":             "orgs",

    # ── Opere e Interventi ───────────────────────────────────
    "work type":              "opere",
    "building type":          "opere",
    "construction material":  "materiali",
    "construction phase":     "fasi",

    # ── Tecnico-Normativo ────────────────────────────────────
    "technical standard":     "norme",
    "law or regulation":      "norme",
    "safety standard":        "norme",
    "permit type":            "autorizzazioni",
    "classification code":    "codes",

    # ── Finanziario ──────────────────────────────────────────
    "budget category":        "valori",
    "funding source":         "finanziamenti",
    "guarantee":              "garanzie",

    # ── Temporal ─────────────────────────────────────────────
    "deadline":               "scadenze",
    "duration":               "durate",

    # ── Legacy/compatibility ─────────────────────────────────
    "software vulnerability": "values",
    "product":                "orgs",
}


# ══════════════════════════════════════════════════════════════
# 1. PARSE — Docling → full markdown + section list
# ══════════════════════════════════════════════════════════════
def parse_pdf(pdf_path: Path) -> tuple[str, list[str]]:
    """Returns (full_markdown, list_of_section_titles)."""
    converter = DocumentConverter()
    result    = converter.convert(str(pdf_path))
    markdown  = result.document.export_to_markdown()
    sections  = re.findall(r'^#{1,2}\s+(.+)$', markdown, re.MULTILINE)
    return markdown, sections


# ══════════════════════════════════════════════════════════════
# 2. ENTITIES — GLiNER zero-shot extraction
# ══════════════════════════════════════════════════════════════

# GLiNER has a ~12k character limit per batch
GLINER_CHUNK_SIZE = 12_000
# Overlap between chunks to avoid missing entities at boundaries
GLINER_CHUNK_OVERLAP = 200


def _chunk_text(text: str, chunk_size: int = GLINER_CHUNK_SIZE, overlap: int = GLINER_CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks for batch processing.
    
    Args:
        text: The input text to chunk
        chunk_size: Maximum size of each chunk (default 12,000 chars)
        overlap: Number of characters to overlap between chunks (default 200)
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If not the last chunk, try to break at a word boundary
        if end < len(text):
            # Look for a good break point (space, newline, punctuation)
            break_point = text.rfind(' ', start + chunk_size - 100, end)
            if break_point > start:
                end = break_point
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start forward, accounting for overlap
        start = end - overlap if end < len(text) else end
        
        # Prevent infinite loop if overlap is too large
        if start <= len(chunks[-1]) if chunks else 0:
            start = end
    
    return chunks


def extract_entities(text: str) -> dict:
    """
    Extract named entities from text using GLiNER with automatic batching.
    
    Handles texts longer than 12k characters by splitting into overlapping chunks,
    processing each chunk, and merging results with deduplication.
    """
    buckets: dict[str, set] = {
        # Base entities
        "people": set(),
        "orgs": set(),
        "dates": set(),
        "values": set(),
        "locations": set(),
        "articles": set(),
        # Edilizia e Lavori Pubblici specific
        "codes": set(),          # CIG, CUP, CPV
        "gara": set(),           # Tipo gara, criteri
        "roles": set(),          # Ruoli professionali
        "opere": set(),          # Tipo opere
        "materiali": set(),      # Materiali
        "fasi": set(),           # Fasi lavori
        "norme": set(),          # Norme e regolamenti
        "autorizzazioni": set(), # Permessi
        "finanziamenti": set(),  # Fonti finanziamento
        "garanzie": set(),       # Garanzie
        "scadenze": set(),       # Deadline
        "durate": set(),         # Durate
    }

    if NER_BACKEND == "gliner" and _ner is not None:
        # Chunk text if it exceeds GLiNER's limit
        chunks = _chunk_text(text, GLINER_CHUNK_SIZE, GLINER_CHUNK_OVERLAP)
        
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"      Processing chunk {i + 1}/{len(chunks)} ({len(chunk):,} chars)...")
            
            hits = _ner.predict_entities(chunk, GLINER_LABELS, threshold=0.4)
            for h in hits:
                bucket = LABEL_MAP.get(h["label"])
                if bucket and len(h["text"].strip()) > 2:
                    buckets[bucket].add(h["text"].strip())

    # Deduplicate + cap each bucket
    return {k: sorted(v)[:15] for k, v in buckets.items() if v}


def detect_doc_date(text: str, entities: dict) -> str | None:
    iso = re.search(r'\b(20\d{2}-\d{2}-\d{2})\b', text)
    if iso:
        return iso.group(1)
    if entities.get("dates"):
        return entities["dates"][0]
    return None


# ══════════════════════════════════════════════════════════════
# 3. WRITE — YAML frontmatter + full markdown body
# ══════════════════════════════════════════════════════════════
def write_document(
    pdf: Path,
    markdown: str,
    sections: list[str],
    entities: dict,
    doc_type: str,
) -> Path:
    # Hash-safe doc_id: stem + 6-char md5 to prevent collisions
    # e.g. contracts/acme.pdf → acme_3f9a1c.md
    raw_bytes  = pdf.read_bytes()
    safe_stem  = re.sub(r'\W+', '_', pdf.stem.lower()).strip('_')
    short_hash = hashlib.md5(raw_bytes).hexdigest()[:6]
    doc_id     = f"{safe_stem}_{short_hash}"

    src_hash   = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()[:16]
    doc_date   = detect_doc_date(markdown, entities)
    page_count = len(re.findall(r'<!-- page \d+ -->', markdown)) or None

    frontmatter = {
        # ── identity ──────────────────────────────────────────
        "doc_id":      doc_id,
        "doc_title":   _guess_title(markdown, pdf.stem),
        "doc_type":    doc_type,
        "source_file": str(pdf),
        "source_hash": src_hash,
        "doc_date":    doc_date,
        "ingested_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        # ── named entities (primary search surface) ───────────
        "entities":    entities,
        # ── structure ─────────────────────────────────────────
        "page_count":  page_count,
        "has_tables":  bool(re.search(r'\|.+\|.+\|', markdown)),
        "has_figures": bool(re.search(r'(figure|fig\.)\s*\d+', markdown, re.I)),
        "has_code":    "```" in markdown,
        "language":    "it",
        "sections":    sections[:25],
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
                 doc_date: str | None, filepath: Path):
    idx_path = CORPUS / "_index.yaml"
    index: dict = {}
    if idx_path.exists():
        with open(idx_path) as f:
            index = yaml.safe_load(f) or {}

    index[doc_id] = {
        "title":    title,
        "type":     doc_type,
        "date":     doc_date,
        "file":     str(filepath),
        "added_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(idx_path, "w") as f:
        yaml.dump(index, f, allow_unicode=True, sort_keys=False)


def _guess_title(markdown: str, fallback: str) -> str:
    m = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
    return m.group(1).strip() if m else fallback.replace('_', ' ').title()


# ══════════════════════════════════════════════════════════════
# 4. INGEST — orchestrate
# ══════════════════════════════════════════════════════════════
def ingest(pdf_path: str, doc_type: str = "document"):
    pdf = Path(pdf_path)
    if not pdf.exists():
        print(f"[error] File not found: {pdf_path}")
        sys.exit(1)

    print(f"[1/4] Parsing    {pdf.name}  (docling) ...")
    markdown, sections = parse_pdf(pdf)

    print(f"[2/4] Extracting entities  ({NER_BACKEND}) ...")
    entities = extract_entities(markdown)

    print(f"[3/4] Writing markdown ...")
    out_path = write_document(pdf, markdown, sections, entities, doc_type)

    doc_id   = out_path.stem
    title    = _guess_title(markdown, pdf.stem)
    doc_date = detect_doc_date(markdown, entities)

    print(f"[4/4] Updating   _index.yaml ...")
    update_index(doc_id, title, doc_type, doc_date, out_path)

    print(f"\n✓  {out_path}  ({len(markdown):,} chars, {len(sections)} sections)")
    print(f"   doc_id : {doc_id}")
    print(f"   entities found : { {k: len(v) for k, v in entities.items()} }")


if __name__ == "__main__":
    ingest(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else "document",
    )