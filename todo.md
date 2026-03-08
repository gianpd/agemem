# Docling Optimization Guide
### Target: 350-page PDF → Markdown | CPU + 8 GB GPU + 32 GB RAM

---

## 1. How Docling Works Internally

Docling uses a pipeline of AI models to parse PDFs. Understanding what runs under the hood is essential before optimizing.

| Stage | Model / Tool | Purpose |
|---|---|---|
| **Layout Detection** | `DocLayNet` (RT-DETR / DINO-based) | Detects page regions: text, table, figure, title |
| **Table Structure** | `TableFormer` | Reconstructs rows/columns from detected table regions |
| **OCR (optional)** | `EasyOCR` or `Tesseract` | Used only on scanned/image-based PDFs |
| **Reading Order** | Rule-based heuristic | Orders detected blocks into logical flow |
| **Markdown export** | Docling serializer | Converts internal DocItem tree → Markdown |

The default `DocumentConverter()` **auto-downloads all models** on first run into the HuggingFace cache. No models = no conversion.

---

## 2. Where to Find Installed Models

Docling stores models in the standard HuggingFace Hub cache and its own artifact directory. Check these locations in order:

### 2.1 Primary Cache Locations

```bash
# HuggingFace Hub cache (main location)
~/.cache/huggingface/hub/

# Docling's dedicated artifact path (newer versions)
~/.cache/docling/models/

# Alternative if HF_HOME env var is set
$HF_HOME/hub/

# Root user (Docker/server environments)
/root/.cache/huggingface/hub/
/root/.cache/docling/models/
```

### 2.2 Quick Diagnostic Script

Run this to confirm which models are present:

```python
import os
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
            print(f"             ✓ {m}  ({size/1e9:.2f} GB)")
    else:
        print("             — no docling models found here")

# Also check environment
print("\n=== Relevant Environment Variables ===")
for k in ["HF_HOME", "TRANSFORMERS_CACHE", "DOCLING_ARTIFACTS_PATH",
          "CUDA_VISIBLE_DEVICES", "TORCH_HOME"]:
    print(f"  {k} = {os.environ.get(k, '(not set)')}")
```

### 2.3 Force a Specific Cache Path

```python
import os
os.environ["HF_HOME"] = "/data/models/hf_cache"          # custom HF path
os.environ["DOCLING_ARTIFACTS_PATH"] = "/data/models/docling"  # custom docling path
```

Set these **before** importing docling.

### 2.4 Pre-download Models (Offline / Air-gapped systems)

```bash
# Download docling models explicitly before first use
python -c "
from docling.utils.model_downloader import download_models_hf
download_models_hf(force=False)  # skip if already present
"

# Or use HF CLI
pip install huggingface_hub
huggingface-cli download ds4sd/docling-models
huggingface-cli download ds4sd/DocLayNet-base
```

---

## 3. Optimized Converter for Your Hardware

### Hardware Profile
- **CPU**: Available (main processing)
- **GPU**: 8 GB VRAM (enough for layout + TableFormer inference)
- **RAM**: 32 GB (ample for 350-page batch)

### 3.1 Full Optimized Implementation

```python
import re
import gc
import torch
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
    TableFormerMode,
)
from docling.datamodel.base_models import InputFormat


def build_optimized_converter() -> DocumentConverter:
    """
    Optimized converter for: CPU + 8GB GPU + 32GB RAM.
    Targets 350-page PDF → Markdown with maximum throughput.
    """
    accelerator = AcceleratorOptions(
        num_threads=8,                          # CPU threads (tune to your core count)
        device=AcceleratorDevice.AUTO,          # uses CUDA if available, else CPU
    )

    pipeline_options = PdfPipelineOptions(
        # --- Model acceleration ---
        accelerator_options=accelerator,

        # --- Table detection ---
        # ACCURATE = full TableFormer (best quality, moderate GPU use)
        # FAST = lighter mode (faster, slightly lower accuracy)
        table_structure_options={"mode": TableFormerMode.ACCURATE},

        # --- OCR: disable for native PDFs (major speed gain) ---
        # Set to True ONLY if your PDF is a scanned image
        do_ocr=False,

        # --- Page image resolution ---
        # 144 DPI is the docling default; lower = faster, higher = better tables
        images_scale=1.0,                       # 1.0 = 72 DPI, 2.0 = 144 DPI (default)

        # --- Generation options ---
        generate_page_images=False,             # skip page image export (saves RAM)
        generate_picture_images=False,          # skip figure extraction
        generate_table_images=False,            # skip table image export
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter


def parse_pdf(pdf_path: Path) -> Tuple[str, List[str]]:
    """
    Returns (full_markdown, list_of_section_titles).
    Drop-in replacement for the base method with optimized settings.
    """
    converter = build_optimized_converter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()
    sections = re.findall(r'^#{1,2}\s+(.+)$', markdown, re.MULTILINE)
    return markdown, sections


def parse_pdf_chunked(
    pdf_path: Path,
    chunk_size: int = 50,
) -> Tuple[str, List[str]]:
    """
    Process large PDFs in page chunks to avoid OOM errors.
    Recommended for 350+ page documents on 8 GB GPU.
    """
    converter = build_optimized_converter()

    # Get total page count first
    import fitz  # PyMuPDF — lightweight, no GPU needed
    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    doc.close()

    all_markdown_parts = []
    all_sections = []

    print(f"Processing {total_pages} pages in chunks of {chunk_size}...")

    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        print(f"  Pages {start+1}–{end} / {total_pages}")

        result = converter.convert(
            str(pdf_path),
            page_range=(start, end - 1),   # 0-indexed, inclusive
        )
        md_chunk = result.document.export_to_markdown()
        all_markdown_parts.append(md_chunk)

        chunk_sections = re.findall(r'^#{1,2}\s+(.+)$', md_chunk, re.MULTILINE)
        all_sections.extend(chunk_sections)

        # Free GPU memory between chunks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_markdown = "\n\n---\n\n".join(all_markdown_parts)
    return full_markdown, all_sections
```

---

## 4. Key Optimization Levers

### 4.1 OCR — The Biggest Decision

```python
# Native/digital PDF (text layer present) → OCR OFF
do_ocr=False   # ← saves ~60% of processing time

# Scanned PDF (image only) → OCR ON
do_ocr=True
ocr_options=EasyOcrOptions(lang=["en"], use_gpu=True)
```

**How to detect automatically:**

```python
import fitz  # PyMuPDF

def needs_ocr(pdf_path: Path, sample_pages: int = 5) -> bool:
    """Returns True if PDF appears to be scanned (no text layer)."""
    doc = fitz.open(str(pdf_path))
    text_chars = 0
    pages_checked = min(sample_pages, doc.page_count)
    for i in range(pages_checked):
        text_chars += len(doc[i].get_text())
    doc.close()
    avg_chars_per_page = text_chars / pages_checked
    return avg_chars_per_page < 100   # threshold: scanned if very few chars
```

### 4.2 GPU Memory Management (8 GB VRAM)

The two GPU-resident models are:
- **Layout model** (DocLayNet): ~1.5–2 GB VRAM
- **TableFormer**: ~1–2 GB VRAM

Both fit comfortably in 8 GB. However, large batches can spike usage. Mitigation:

```python
# Option A: Reduce image resolution (less GPU memory per page)
images_scale=1.5   # instead of 2.0 — good balance of quality vs. memory

# Option B: Disable table reconstruction for non-tabular PDFs
from docling.datamodel.pipeline_options import TableStructureOptions
table_structure_options=TableStructureOptions(do_cell_matching=False)

# Option C: Force CPU-only if GPU keeps OOMing
device=AcceleratorDevice.CPU
```

### 4.3 Threading

```python
# Set num_threads to physical core count (not hyperthreads)
import os
num_threads = os.cpu_count() // 2   # conservative; use full count if CPU is idle
accelerator_options=AcceleratorOptions(num_threads=num_threads, device=AcceleratorDevice.AUTO)
```

### 4.4 What to Disable When You Don't Need It

```python
PdfPipelineOptions(
    generate_page_images=False,      # -30% memory if you only need text
    generate_picture_images=False,   # skips figure crops
    generate_table_images=False,     # skips table image export
    do_table_structure=False,        # skip TableFormer entirely (only if no tables)
)
```

---

## 5. Expected Performance on Your Hardware

| Configuration | Est. Time (350 pages) | VRAM Usage |
|---|---|---|
| Default (GPU + OCR off) | ~8–15 min | ~3–4 GB |
| Chunked (50 pages/batch) | ~10–18 min | ~2–3 GB peak |
| CPU only | ~35–60 min | 0 GB VRAM |
| OCR enabled (scanned) | ~40–80 min | ~4–5 GB |

Estimates assume a modern CPU (8+ cores) and CUDA 11.8+ with PyTorch.

---

## 6. Complete Drop-in Replacement for Base Method

```python
import re
import gc
import os
import torch
from pathlib import Path
from typing import List, Tuple

# Set cache paths before any docling import
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("DOCLING_ARTIFACTS_PATH", os.path.expanduser("~/.cache/docling/models"))

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.datamodel.base_models import InputFormat


_CONVERTER: DocumentConverter | None = None  # module-level singleton


def _get_converter() -> DocumentConverter:
    """Lazy singleton — build once, reuse across calls."""
    global _CONVERTER
    if _CONVERTER is None:
        opts = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
                num_threads=min(8, os.cpu_count() or 4),
                device=AcceleratorDevice.AUTO,
            ),
            do_ocr=False,
            generate_page_images=False,
            generate_picture_images=False,
            generate_table_images=False,
        )
        _CONVERTER = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _CONVERTER


def parse_pdf(pdf_path: Path) -> Tuple[str, List[str]]:
    """
    Returns (full_markdown, list_of_section_titles).
    Optimized for CPU + 8 GB GPU + 32 GB RAM, 350-page PDFs.
    """
    converter = _get_converter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()
    sections = re.findall(r'^#{1,2}\s+(.+)$', markdown, re.MULTILINE)

    # Release GPU cache after conversion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return markdown, sections
```

---

## 7. Troubleshooting Checklist

| Problem | Likely Cause | Fix |
|---|---|---|
| Slow first run | Models downloading | Pre-download with `download_models_hf()` |
| `CUDA out of memory` | VRAM spike | Use `images_scale=1.5`, chunked processing |
| Empty markdown output | Scanned PDF | Enable `do_ocr=True` |
| Missing tables | Tables rendered as images | Ensure `do_table_structure=True` (default) |
| Models not found offline | Cache path wrong | Set `DOCLING_ARTIFACTS_PATH` explicitly |
| Slow on CPU despite GPU available | CUDA not installed | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

---

## 8. Install Command Reference

```bash
# Base install
pip install docling

# With GPU support (CUDA 11.8)
pip install docling torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Pre-download all models
python -c "from docling.utils.model_downloader import download_models_hf; download_models_hf()"

# Check GPU availability
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```