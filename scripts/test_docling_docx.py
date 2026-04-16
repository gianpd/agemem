"""Test Docling's DOCX conversion capabilities."""
from pathlib import Path
from docling.document_converter import DocumentConverter


def test_docx_conversion(docx_path: str | Path) -> str:
    """Convert a DOCX file to markdown using Docling."""
    converter = DocumentConverter()
    result = converter.convert(docx_path)
    return result.document.export_to_markdown()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_docling_docx.py <path_to_docx>")
        print("\nExample: python test_docling_docx.py sample.docx")
        sys.exit(1)

    docx_path = Path(sys.argv[1])

    if not docx_path.exists():
        print(f"File not found: {docx_path}")
        sys.exit(1)

    if docx_path.suffix.lower() != ".docx":
        print(f"Expected .docx file, got: {docx_path.suffix}")
        sys.exit(1)

    print(f"Converting: {docx_path.name}")
    print("-" * 40)

    markdown = test_docx_conversion(docx_path)

    print(markdown)
    print("-" * 40)
    print(f"Output length: {len(markdown)} characters")

    # Optionally save to file
    output_path = docx_path.with_suffix(".md")
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Saved to: {output_path}")