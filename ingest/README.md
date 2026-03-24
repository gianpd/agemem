  What It Is

  The /ingest mechanism is a document ingestion pipeline that converts PDFs into structured, searchable markdown with intelligent metadata extraction. It's essentially a
  lightweight RAG (Retrieval-Augmented Generation) system that allows the agent to read and reason over documents you provide.

  How It Works (4-Stage Pipeline)

  Stage 1: Parse (ingest/ingest.py)
  - Uses Docling (IBM's document conversion library) to convert PDFs to markdown
  - Auto-detects scanned vs. native PDFs using PyMuPDF
  - Optional OCR for scanned documents
  - Configurable table structure recognition (FAST vs. ACCURATE mode)
  - Models are cached locally (~/.cache/docling, ~/.cache/huggingface)

  Stage 2: Extract (ingest/ingest.py)
  - Uses GLiNER (zero-shot NER) to extract named entities without training
  - Handles GLiNER's 384-token limit via sentence-aware splitting
  - Domain-specific entity labels via configurable "label sets"
  - **NEW: Multi-scale extraction** — secondary pass at lower threshold for better recall
  - **NEW: Post-processing pipeline** — configurable filters for improved accuracy

  Stage 3: Write (ingest/ingest.py)
  - Creates markdown files with YAML frontmatter containing:
    - Document identity (doc_id, title, type, source hash)
    - Extracted entities organized into buckets
    - Structure metadata (page count, has_tables, has_figures, sections)
  - Outputs to corpus/{doc_id}.md

  Stage 4: Index (ingest/ingest.py)
  - Updates corpus/_index.yaml for quick metadata lookups

  Domain-Specific Label Sets

  Four built-in configurations in ingest/gliner_labels/:

  ┌──────────┬──────────────────────────────┬───────────────────────────────────────────┐
  │   Set    │           Use Case           │             Example Entities              │
  ├──────────┼──────────────────────────────┼───────────────────────────────────────────┤
  │ edilizia │ Italian construction/tenders │ CIG, CUP, appalti, RUP, Direttore Lavori  │
  ├──────────┼──────────────────────────────┼───────────────────────────────────────────┤
  │ research │ Scientific papers            │ datasets, algorithms, citations, models   │
  ├──────────┼──────────────────────────────┼───────────────────────────────────────────┤
  │ legal    │ Contracts/legal              │ parties, clauses, jurisdictions, statutes │
  ├──────────┼──────────────────────────────┼───────────────────────────────────────────┤
  │ generic  │ Unknown/mixed documents      │ people, orgs, dates, emails, products     │
  └──────────┴──────────────────────────────┴───────────────────────────────────────────┘

  You can also define custom labels via YAML: --labels /path/to/config.yaml:my_domain

  Generic Label Set (NEW)

  The `generic` label set is designed for documents of unknown type or mixed content.
  It extracts 22 universal entity types organized into intuitive buckets:

  Core Entities:
    - person → people
    - organization → organizations
    - location → locations
    - date → dates
    - email → emails
    - phone number → phones
    - url → urls
    - address → addresses

  Numeric/Financial:
    - number → numbers
    - monetary value → values
    - percentage → percentages
    - quantity → quantities
    - unit → units

  Document Structure:
    - section → sections
    - heading → headings
    - reference number → references
    - version → versions

  Content Types:
    - product → products
    - service → services
    - event → events
    - technology → technologies
    - file format → formats

  Post-Processing Pipeline (NEW)

  A configurable post-processing pipeline improves extraction accuracy by filtering
  and validating entities before they are stored. Located in `ingest/entity_post_processor.py`.

  Features:
    - Length filtering: Remove entities that are too short or too long
    - Stopword filtering: Filter out entities that are mostly common stopwords
    - Pattern validation: Validate entities against regex patterns (dates, emails, phones, etc.)
    - Confidence boosting: Boost scores for entities appearing multiple times (coreference)
    - Deduplication: Remove duplicates keeping highest confidence or longest variant
    - Multi-scale extraction: Run secondary extraction at lower threshold for better recall

  Pre-configured Settings:

  ┌─────────────┬────────────────────────────────────────────────────────────┐
  │   Config    │                    Characteristics                         │
  ├─────────────┼────────────────────────────────────────────────────────────┤
  │ default     │ Balanced precision/recall — good for most documents        │
  ├─────────────┼────────────────────────────────────────────────────────────┤
  │ conservative│ High precision, strict filtering — when accuracy matters   │
  ├─────────────┼────────────────────────────────────────────────────────────┤
  │ aggressive  │ High recall, permissive — for exploratory analysis         │
  └─────────────┴────────────────────────────────────────────────────────────┘

  Document Type Auto-Detection (NEW)

  The system can analyze document content to suggest the appropriate label set:

  ```python
  from ingest.ingest import detect_document_type

  detected = detect_document_type(text)
  # Returns: 'edilizia', 'research', 'legal', 'finance', 'medical', or 'generic'
  ```

  Detection is based on keyword signals (requires 3+ matches for specificity).

  CLI Usage

  Basic ingestion:
    uv run ingest/ingest.py document.pdf [doc_type] [--labels <label_set>]

  Examples:
    # Use generic labels for unknown document type
    uv run ingest/ingest.py report.pdf document --labels generic

    # Use aggressive post-processing for better recall
    uv run ingest/ingest.py report.pdf --labels generic --post-process-config aggressive

    # Enable multi-scale extraction explicitly
    uv run ingest/ingest.py report.pdf --labels generic --multiscale

    # Use conservative settings for high precision
    uv run ingest/ingest.py contract.pdf --labels legal --post-process-config conservative

    # Disable post-processing (raw GLiNER output)
    uv run ingest/ingest.py report.pdf --labels research --no-post-process

  CLI Arguments:
    --labels <set>              Label set: edilizia, research, legal, generic, or path:key
    --post-process-config <cfg> Post-processing: default, conservative, aggressive
    --no-post-process           Disable post-processing pipeline
    --multiscale                Enable multi-scale extraction
    --no-multiscale             Disable multi-scale extraction
    --ocr                       Force OCR for scanned PDFs
    --accurate-tables           Use ACCURATE table mode (slower)
    --disable-tables            Disable table recognition (fastest)

  How the Agent Uses It

  The tools/corpus.py module provides 5 tools the agent can call:

  1. list_documents — See what's available
  2. search_metadata — Find by title/type/tags
  3. grep_corpus — Full-text search across all documents (uses ripgrep)
  4. read_document — Read full content (truncated if >8KB)
  5. read_lines — Read specific line ranges for large docs

  The "Real Deal"

  It's a local, privacy-preserving document intelligence system:

  - No API calls for ingestion — everything runs locally (Docling + GLiNER)
  - Zero-shot NER — no training required; GLiNER understands natural language labels
  - Structured search surface — instead of just embedding search, you get explicit entity buckets (codes, dates, orgs, etc.) that can be matched precisely
  - Post-processing pipeline — configurable filters remove false positives and improve accuracy
  - Generic label set — works across document types when domain is unknown
  - Persistent corpus — documents survive across sessions; the agent builds up a library
  - Tool-integrated — the agent decides when to search documents based on your queries

  Typical workflow:
  # Ingest documents
  uv run ingest/ingest.py bando.pdf bando --labels edilizia
  uv run ingest/ingest.py paper.pdf research --labels research
  uv run ingest/ingest.py unknown.pdf document --labels generic --multiscale

  # Agent can now search them during conversation
  # "What does the document say about CIG codes?" → triggers grep_corpus
  # "List my construction documents" → triggers list_documents

  The corpus lives in /home/jaco/develops/agemem/corpus/ (defined in core/config.py), and the tools are registered in the main orchestrator so the agent can invoke them
  automatically based on your questions.

  Python API

  For programmatic use:

  ```python
  from ingest.ingest import ingest, extract_entities
  from ingest.entity_post_processor import create_processor
  from ingest.gliner_labels import get_builtin_labels

  # Get label configuration
  config = get_builtin_labels('generic')

  # Create post-processor
  processor = create_processor('default')  # or 'conservative', 'aggressive'

  # Ingest with options
  doc_id = ingest(
      'document.pdf',
      doc_type='document',
      labels_arg='generic',
      post_process=True,
      post_process_config='default',
      enable_multiscale=True,
  )
  ```
