"""Encode processed user profiles with the same MiniLM model as script 02.

Run: .venv/Scripts/python.exe SCRIPTS/04_create_user_embeddings.py
Install dependencies with: python -m pip install -r requirements.txt

Load later with np.load("DATA/user_embeddings.npy", allow_pickle=False).
Row i corresponds to row i of user_metadata.csv (excluding the header).
Both users and careers are unit-normalized float32 vectors, so
user_embeddings @ career_embeddings.T gives their cosine similarities.
"""

import argparse
import csv
import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "DATA"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def load_profiles(path):
    """Reject invalid records without sorting, dropping, or reassigning IDs."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = {"user_id", "profile_text"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        texts, metadata, seen = [], [], set()
        for row_number, row in enumerate(reader, 2):
            user_id = (row["user_id"] or "").strip()
            text = " ".join((row["profile_text"] or "").split())
            if not user_id or not text:
                raise ValueError(f"Row {row_number}: user ID and profile text must be nonempty")
            if user_id in seen:
                raise ValueError(f"Row {row_number}: duplicate user ID {user_id}")
            seen.add(user_id)
            texts.append(text)
            metadata.append({"user_id": user_id})
    if not texts:
        raise ValueError(f"{path}: no user profiles found")
    return texts, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate aligned user embeddings and metadata.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-users", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size < 1 or args.expected_users < 1:
        parser.error("--batch-size and --expected-users must be positive")
    texts, metadata = load_profiles(args.data_dir / "processed_user_profiles.csv")
    if len(texts) != args.expected_users:
        raise ValueError(f"Expected {args.expected_users} users; found {len(texts)}")

    # Lazy imports keep --help and input validation usable without ML packages.
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit("Install dependencies with: python -m pip install -r requirements.txt") from error

    # Validate compatibility against the artifacts created by script 02.
    career_stats = json.loads((args.data_dir / "career_embedding_stats.json").read_text(encoding="utf-8"))
    if career_stats.get("model") != MODEL_NAME or career_stats.get("revision") != MODEL_REVISION:
        raise ValueError("Career model/revision differs; regenerate both sets with the same model")
    careers = np.load(args.data_dir / "career_embeddings.npy", allow_pickle=False)
    if (careers.ndim != 2 or careers.shape[1] != 384 or not len(careers)
            or not np.isfinite(careers).all()
            or not np.allclose(np.linalg.norm(careers, axis=1), 1.0, atol=1e-5)):
        raise ValueError("Expected finite, unit-normalized 384-dimensional career embeddings")

    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
    token_counts = np.array([len(ids) for ids in model.tokenizer(
        texts, add_special_tokens=True, truncation=False, padding=False, verbose=False
    )["input_ids"]])
    limit = min(256, model.max_seq_length)
    over_limit = int((token_counts > limit).sum())
    print(f"Token audit: mean {token_counts.mean():.2f}, median {np.median(token_counts):.1f}, "
          f"minimum {token_counts.min()}, maximum {token_counts.max()}; "
          f"over limit {over_limit / len(texts):.2%}.")
    if over_limit:
        raise ValueError("User profiles exceed the token limit; rerun script 03. No outputs written.")

    # These settings match script 02. encode preserves input order across batches.
    embeddings = model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    if embeddings.shape != (len(metadata), 384) or not np.isfinite(embeddings).all():
        raise ValueError("Expected one finite, 384-dimensional embedding per user")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise ValueError("User embeddings must have unit length for cosine comparisons")

    embedding_path = args.data_dir / "user_embeddings.npy"
    metadata_path = args.data_dir / "user_metadata.csv"
    np.save(embedding_path, embeddings, allow_pickle=False)
    with metadata_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["user_id"])
        writer.writeheader()
        writer.writerows(metadata)

    # Round-trip verification checks saved arrays and every metadata row.
    saved = np.load(embedding_path, allow_pickle=False)
    with metadata_path.open(encoding="utf-8", newline="") as source:
        saved_metadata = list(csv.DictReader(source))
    if not np.array_equal(saved, embeddings) or saved_metadata != metadata:
        raise ValueError("Saved embeddings or metadata failed round-trip verification")
    stats = {
        "model": MODEL_NAME, "revision": MODEL_REVISION,
        "users": len(texts), "dimensions": 384, "dtype": "float32",
        "normalize_embeddings": True, "career_compatibility_verified": True,
        "metadata_order_verified": True, "token_limit_including_special_tokens": limit,
        "average_tokens": float(token_counts.mean()), "median_tokens": float(np.median(token_counts)),
        "minimum_tokens": int(token_counts.min()), "maximum_tokens": int(token_counts.max()),
        "percentage_truncated": 0.0,
    }
    (args.data_dir / "user_embedding_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Saved and verified {embeddings.shape} user embeddings and matching metadata in {args.data_dir}")
    print("Career model/revision and unit normalization match; 0% of profiles truncated.")


if __name__ == "__main__":
    main()
