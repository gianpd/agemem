  What It Is

  The /ingest mechanism is a document ingestion pipeline that converts PDFs into structured, searchable markdown with intelligent metadata extraction. It's essentially a
  lightweight RAG (Retrieval-Augmented Generation) system that allows the agent to read and reason over documents you provide.

  How It Works (4-Stage Pipeline)

  Stage 1: Parse (ingest/ingest.py:483-530)
  - Uses Docling (IBM's document conversion library) to convert PDFs to markdown
  - Auto-detects scanned vs. native PDFs using PyMuPDF
  - Optional OCR for scanned documents
  - Configurable table structure recognition (FAST vs. ACCURATE mode)
  - Models are cached locally (~/.cache/docling, ~/.cache/huggingface)

  Stage 2: Extract (ingest/ingest.py:429-461)
  - Uses GLiNER (zero-shot NER) to extract named entities without training
  - Handles GLiNER's 384-token limit via sentence-aware splitting
  - Domain-specific entity labels via configurable "label sets"

  Stage 3: Write (ingest/ingest.py:536-588)
  - Creates markdown files with YAML frontmatter containing:
    - Document identity (doc_id, title, type, source hash)
    - Extracted entities organized into buckets
    - Structure metadata (page count, has_tables, has_figures, sections)
  - Outputs to corpus/{doc_id}.md

  Stage 4: Index (ingest/ingest.py:592-610)
  - Updates corpus/_index.yaml for quick metadata lookups

  Domain-Specific Label Sets

  Three built-in configurations in ingest/gliner_labels/:

  ┌──────────┬──────────────────────────────┬───────────────────────────────────────────┐
  │   Set    │           Use Case           │             Example Entities              │
  ├──────────┼──────────────────────────────┼───────────────────────────────────────────┤
  │ edilizia │ Italian construction/tenders │ CIG, CUP, appalti, RUP, Direttore Lavori  │
  ├──────────┼──────────────────────────────┼───────────────────────────────────────────┤
  │ research │ Scientific papers            │ datasets, algorithms, citations, models   │
  ├──────────┼──────────────────────────────┼───────────────────────────────────────────┤
  │ legal    │ Contracts/legal              │ parties, clauses, jurisdictions, statutes │
  └──────────┴──────────────────────────────┴───────────────────────────────────────────┘

  You can also define custom labels via YAML: --labels /path/to/config.yaml:my_domain

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
  - Persistent corpus — documents survive across sessions; the agent builds up a library
  - Tool-integrated — the agent decides when to search documents based on your queries

  Typical workflow:
  # Ingest documents
  uv run ingest/ingest.py bando.pdf bando --labels edilizia
  uv run ingest/ingest.py paper.pdf research --labels research

  # Agent can now search them during conversation
  # "What does the document say about CIG codes?" → triggers grep_corpus
  # "List my construction documents" → triggers list_documents

  The corpus lives in /home/jaco/develops/agemem/corpus/ (defined in core/config.py:32), and the tools are registered in the main orchestrator so the agent can invoke them
  automatically based on your questions.