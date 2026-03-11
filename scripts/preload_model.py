#!/usr/bin/env python3
"""
scripts/preload_model.py
------------------------
One-shot script to download and cache the embedding model.
"""
import os
import sys
from pathlib import Path

# Set cache paths BEFORE importing sentence_transformers
cache_dir = Path.home() / ".cache" / "agemem" / "models"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(cache_dir)
os.environ["HF_HUB_CACHE"] = str(cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

from sentence_transformers import SentenceTransformer


def main():
    model_name = "Qwen/Qwen3-Embedding-0.6B"

    print(f"Downloading {model_name}...")
    print(f"Cache directory: {cache_dir}")

    try:
        model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_dir),
            trust_remote_code=True,
        )
        print(f"\nSuccess! Model cached at: {cache_dir}")
        print(f"Model: {model_name}")
        return 0
    except Exception as e:
        print(f"\nError downloading model: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())