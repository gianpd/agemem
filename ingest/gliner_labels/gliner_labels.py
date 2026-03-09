"""
GLiNER label definitions for different document domains.

This module provides pre-defined label sets for various document types.
Users can also define custom labels via YAML configuration files.
"""

from typing import Dict, List, Any


# ═══════════════════════════════════════════════════════════════
# Domain: Italian Construction & Public Tenders (Edilizia/Lavori Pubblici)
# ═══════════════════════════════════════════════════════════════
EDILIZIA_LABELS: List[str] = [
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

EDILIZIA_LABEL_MAP: Dict[str, str] = {
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
}

EDILIZIA_BUCKETS: Dict[str, List[str]] = {
    # Base entities
    "people":       [],
    "orgs":         [],
    "dates":        [],
    "values":       [],
    "locations":    [],
    "articles":     [],
    # Edilizia specific
    "codes":        [],
    "gara":         [],
    "roles":        [],
    "opere":        [],
    "materiali":    [],
    "fasi":         [],
    "norme":        [],
    "autorizzazioni": [],
    "finanziamenti": [],
    "garanzie":     [],
    "valori":       [],
    "scadenze":     [],
    "durate":       [],
}


# ═══════════════════════════════════════════════════════════════
# Domain: Research & Science Papers
# ═══════════════════════════════════════════════════════════════
RESEARCH_LABELS: List[str] = [
    # ── Base entities ─────────────────────────────────────────
    "person",                   # Authors, researchers
    "organization",             # Universities, research institutes
    "location",                 # Countries, cities
    "date",                     # Publication dates

    # ── Academic/Research ────────────────────────────────────
    "research field",           # Machine Learning, Quantum Physics
    "research method",          # Experimental, Theoretical, Simulation
    "dataset",                  # ImageNet, COCO, PubMed
    "metric",                   # Accuracy, F1-score, BLEU
    "algorithm",                # Transformer, CNN, LSTM
    "model architecture",       # BERT, GPT, ResNet
    "hyperparameter",           # learning rate, batch size

    # ── Scientific ────────────────────────────────────────────
    "chemical compound",        # H2O, CO2, proteins
    "biological entity",        # Gene names, proteins, organisms
    "physical quantity",        # Temperature, pressure, velocity
    "unit of measurement",      # kg, meters, seconds
    "mathematical concept",     # Theorem, lemma, equation

    # ── Publication ───────────────────────────────────────────
    "journal name",             # Nature, Science, ICML
    "conference name",          # NeurIPS, CVPR, ACL
    "publication type",         # Paper, Preprint, Review
    "citation",                 # Cited works
    "funding source",           # NSF, NIH, ERC
    "grant number",             # Grant identifiers

    # ── Technical ─────────────────────────────────────────────
    "software",                 # Python, TensorFlow, PyTorch
    "hardware",                 # GPU, TPU, CPU
    "programming language",     # Python, C++, Julia
    "file format",              # PDF, JSON, CSV
]

RESEARCH_LABEL_MAP: Dict[str, str] = {
    # Base entities
    "person":               "people",
    "organization":         "orgs",
    "location":             "locations",
    "date":                 "dates",

    # Academic/Research
    "research field":       "fields",
    "research method":      "methods",
    "dataset":              "datasets",
    "metric":               "metrics",
    "algorithm":            "algorithms",
    "model architecture":   "models",
    "hyperparameter":       "hyperparams",

    # Scientific
    "chemical compound":    "compounds",
    "biological entity":    "bio_entities",
    "physical quantity":    "quantities",
    "unit of measurement":  "units",
    "mathematical concept": "math_concepts",

    # Publication
    "journal name":         "venues",
    "conference name":      "venues",
    "publication type":     "pub_types",
    "citation":             "citations",
    "funding source":       "funding",
    "grant number":         "grants",

    # Technical
    "software":             "software",
    "hardware":             "hardware",
    "programming language": "languages",
    "file format":          "formats",
}

RESEARCH_BUCKETS: Dict[str, List[str]] = {
    # Base entities
    "people":         [],
    "orgs":           [],
    "locations":      [],
    "dates":          [],
    # Academic
    "fields":         [],
    "methods":        [],
    "datasets":       [],
    "metrics":        [],
    "algorithms":     [],
    "models":         [],
    "hyperparams":    [],
    # Scientific
    "compounds":      [],
    "bio_entities":   [],
    "quantities":     [],
    "units":          [],
    "math_concepts":  [],
    # Publication
    "venues":         [],
    "pub_types":      [],
    "citations":      [],
    "funding":        [],
    "grants":         [],
    # Technical
    "software":       [],
    "hardware":       [],
    "languages":      [],
    "formats":        [],
}


# ═══════════════════════════════════════════════════════════════
# Domain: Legal & Contracts (General)
# ═══════════════════════════════════════════════════════════════
LEGAL_LABELS: List[str] = [
    # ── Base entities ─────────────────────────────────────────
    "person",
    "organization",
    "date",
    "financial value",
    "location",

    # ── Legal Parties ────────────────────────────────────────
    "plaintiff",                # Complainant
    "defendant",                # Accused/Respondent
    "judge",                    # Judicial authority
    "attorney",                 # Lawyer
    "witness",                  # Testimony provider
    "expert",                   # Expert witness

    # ── Legal Documents ──────────────────────────────────────
    "contract type",            # NDA, SLA, Employment
    "clause type",              # Termination, Confidentiality
    "legal reference",          # Case law, precedent
    "statute",                  # Law code reference
    "regulation",               # Administrative rules

    # ── Legal Terms ──────────────────────────────────────────
    "jurisdiction",             # Legal authority area
    "court",                    # Specific court name
    "case number",              # Docket/case ID
    "liability type",           # Limited, Joint, Several
    "remedy",                   # Damages, Injunction
    "obligation",               # Duty/Responsibility
]

LEGAL_LABEL_MAP: Dict[str, str] = {
    "person":           "people",
    "organization":     "orgs",
    "date":             "dates",
    "financial value":  "values",
    "location":         "locations",
    "plaintiff":        "parties",
    "defendant":        "parties",
    "judge":            "judicial",
    "attorney":         "legal_repr",
    "witness":          "witnesses",
    "expert":           "experts",
    "contract type":    "doc_types",
    "clause type":      "clauses",
    "legal reference":  "references",
    "statute":          "statutes",
    "regulation":       "regulations",
    "jurisdiction":     "jurisdictions",
    "court":            "courts",
    "case number":      "case_ids",
    "liability type":   "liabilities",
    "remedy":           "remedies",
    "obligation":       "obligations",
}

LEGAL_BUCKETS: Dict[str, List[str]] = {
    "people":        [],
    "orgs":          [],
    "dates":         [],
    "values":        [],
    "locations":     [],
    "parties":       [],
    "judicial":      [],
    "legal_repr":    [],
    "witnesses":     [],
    "experts":       [],
    "doc_types":     [],
    "clauses":       [],
    "references":    [],
    "statutes":      [],
    "regulations":   [],
    "jurisdictions": [],
    "courts":        [],
    "case_ids":      [],
    "liabilities":   [],
    "remedies":      [],
    "obligations":   [],
}


# ═══════════════════════════════════════════════════════════════
# Registry of built-in label sets
# ═══════════════════════════════════════════════════════════════
BUILTIN_LABEL_SETS: Dict[str, Dict[str, Any]] = {
    "edilizia": {
        "labels": EDILIZIA_LABELS,
        "label_map": EDILIZIA_LABEL_MAP,
        "buckets": EDILIZIA_BUCKETS,
        "description": "Italian construction and public tenders (CIG, CUP, appalti)",
    },
    "research": {
        "labels": RESEARCH_LABELS,
        "label_map": RESEARCH_LABEL_MAP,
        "buckets": RESEARCH_BUCKETS,
        "description": "Research papers and scientific publications",
    },
    "legal": {
        "labels": LEGAL_LABELS,
        "label_map": LEGAL_LABEL_MAP,
        "buckets": LEGAL_BUCKETS,
        "description": "Legal documents and contracts",
    },
}


def get_builtin_labels(name: str) -> Dict[str, Any]:
    """
    Get a built-in label set by name.

    Args:
        name: One of 'edilizia', 'research', 'legal'

    Returns:
        Dictionary with 'labels', 'label_map', 'buckets', 'description'

    Raises:
        ValueError: If name is not a recognized built-in label set
    """
    if name not in BUILTIN_LABEL_SETS:
        available = ", ".join(BUILTIN_LABEL_SETS.keys())
        raise ValueError(f"Unknown label set '{name}'. Available: {available}")
    return BUILTIN_LABEL_SETS[name].copy()


def list_builtin_labels() -> Dict[str, str]:
    """List all available built-in label sets with descriptions."""
    return {name: info["description"] for name, info in BUILTIN_LABEL_SETS.items()}
