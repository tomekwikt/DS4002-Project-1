"""Encode career profiles with sentence-transformers/all-MiniLM-L6-v2.

Install dependencies: python -m pip install numpy sentence-transformers
Run: python SCRIPTS/02_create_career_embeddings.py
The first run downloads the pretrained model; subsequent runs use its cache.

Load and compare later (use the SAME model for resume + interests):
    careers = np.load("DATA/career_embeddings.npy", allow_pickle=False)
    user = model.encode(resume_and_interests, normalize_embeddings=True)
    similarities = careers @ user
Because both vectors are normalized, their dot product is cosine similarity.
Metadata row i identifies embedding row i (the CSV header is not a data row).

Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
Script 01 selects complete source text within 250 tokens, including special
tokens. This script rejects overlength inputs instead of silently truncating.
"""

import argparse
import csv
import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "DATA"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CODE = "O*NET-SOC Code"


def load_profiles(path):
    """Validate inputs without sorting or dropping rows, preserving alignment."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        # Script 01 calls its combined text `profile_text`; also accept `text`.
        text_column = "profile_text" if "profile_text" in columns else "text"
        missing = {CODE, "Title", text_column} - set(columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        texts, metadata, seen = [], [], set()
        for row_number, row in enumerate(reader, start=2):
            code = (row[CODE] or "").strip()
            title = (row["Title"] or "").strip()
            text = " ".join((row[text_column] or "").split())
            if not code or not title or not text:
                raise ValueError(f"Row {row_number}: code, title, and profile text must be nonempty")
            if code in seen:
                raise ValueError(f"Row {row_number}: duplicate occupation code {code}")
            seen.add(code)
            texts.append(text)
            metadata.append({CODE: code, "Title": title})
    if not texts:
        raise ValueError(f"{path}: no career profiles found")
    return texts, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate aligned career embeddings and metadata.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    texts, metadata = load_profiles(args.data_dir / "processed_career_profiles.csv")

    # Lazy imports allow input validation and --help without ML dependencies.
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit(
            "Install dependencies with: python -m pip install numpy sentence-transformers"
        ) from error

    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
    # Measure truncation using the actual tokenizer, including special tokens.
    token_counts = np.array([
        len(ids) for ids in model.tokenizer(
            texts, add_special_tokens=True, truncation=False, padding=False, verbose=False
        )["input_ids"]
    ])
    limit = min(256, model.max_seq_length)
    truncated = int((token_counts > limit).sum())
    stats = {
        "model": MODEL_NAME, "revision": MODEL_REVISION,
        "profiles": len(texts), "token_limit_including_special_tokens": limit,
        "average_tokens": float(token_counts.mean()),
        "median_tokens": float(np.median(token_counts)),
        "minimum_tokens": int(token_counts.min()), "maximum_tokens": int(token_counts.max()),
        "percentage_over_limit": 100 * truncated / len(texts),
        "profiles_in_245_to_250_range": int(((token_counts >= 245) & (token_counts <= 250)).sum()),
    }
    print(
        f"Token audit: average {token_counts.mean():.2f}, median {np.median(token_counts):.0f}, "
        f"minimum {token_counts.min()}, maximum {token_counts.max()}; "
        f"over limit {truncated / len(texts):.2%}."
    )
    if truncated:
        raise ValueError("Profiles exceed the model token limit; rerun script 01. No embeddings written.")
    stats["percentage_truncated"] = 0.0
    print(f"Encoding {len(texts):,} profiles with 0.00% truncated.")
    # encode returns results in input order, even when batching internally.
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    if embeddings.shape != (len(metadata), 384) or not np.isfinite(embeddings).all():
        raise ValueError("Expected one finite, 384-dimensional embedding per career")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise ValueError("Career embeddings must have unit length for cosine comparisons")

    # Numeric .npy files preserve precision and can be loaded without pickle.
    np.save(args.data_dir / "career_embeddings.npy", embeddings, allow_pickle=False)
    with (args.data_dir / "career_metadata.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=[CODE, "Title"])
        writer.writeheader()
        writer.writerows(metadata)
    (args.data_dir / "career_embedding_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Saved embeddings {embeddings.shape} and matching metadata to {args.data_dir}")


if __name__ == "__main__":
    main()
