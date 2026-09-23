"""Rank existing normalized career embeddings for each user; no model loading.

Run: .venv/Scripts/python.exe SCRIPTS/05_match_careers.py
Requires only NumPy and the artifacts from preparation/embedding scripts.
Equal similarity scores are ordered by career code for reproducible ranks.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = "O*NET-SOC Code"
FIELDS = ["user_id", "rank", "career_code", "career_title", "cosine_similarity"]


def career_file(data_dir, name):
    """Support the original layout and the reorganized career-data folder."""
    candidates = [data_dir / name, data_dir / "Career Profile Files" / name]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {name} in DATA or Career Profile Files; found {len(found)}")
    return found[0]


def read_metadata(path, fields):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = [{key: (row[key] or "").strip() for key in fields} for row in reader]
    if not rows or any(not value for row in rows for value in row.values()):
        raise ValueError(f"{path}: empty metadata or required values")
    ids = [row[fields[0]] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate IDs")
    return rows


def validate_embeddings(embeddings, metadata, expected_count, label):
    """Check dimensions, row counts, finite values, and existing normalization."""
    if embeddings.shape != (expected_count, 384) or len(metadata) != expected_count:
        raise ValueError(f"{label}: expected {expected_count} aligned rows with 384 dimensions; "
                         f"found embeddings {embeddings.shape}, metadata {len(metadata)}")
    if not np.issubdtype(embeddings.dtype, np.floating) or not np.isfinite(embeddings).all():
        raise ValueError(f"{label}: embeddings must contain finite floating-point values")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError(f"{label}: embeddings are not unit normalized")


def rank_matches(users, careers, user_metadata, career_metadata, top_k=5):
    if not 1 <= top_k <= len(career_metadata):
        raise ValueError("top_k must be between one and the number of careers")
    # Cosine similarity equals the dot product for the validated unit vectors.
    # Float64 arithmetic reduces rounding noise; saved embeddings are untouched.
    similarities = users.astype(np.float64) @ careers.astype(np.float64).T
    if not np.isfinite(similarities).all() or np.any(np.abs(similarities) > 1 + 1e-5):
        raise ValueError("Invalid cosine similarity values")
    similarities = np.clip(similarities, -1.0, 1.0)
    codes = np.array([row[CODE] for row in career_metadata])
    results = []
    for user_index, user in enumerate(user_metadata):
        # lexsort's last key is primary: descending score, then ascending code.
        best = np.lexsort((codes, -similarities[user_index]))[:top_k]
        for rank, career_index in enumerate(best, 1):
            career = career_metadata[career_index]
            results.append({"user_id": user["user_id"], "rank": rank,
                            "career_code": career[CODE], "career_title": career["Title"],
                            "cosine_similarity": float(similarities[user_index, career_index])})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "DATA")
    parser.add_argument("--output", type=Path, default=ROOT / "OUTPUT/career_matches.csv")
    args = parser.parse_args()
    data_dir = args.data_dir
    users = np.load(data_dir / "user_embeddings.npy", allow_pickle=False)
    careers = np.load(data_dir / "career_embeddings.npy", allow_pickle=False)
    user_metadata = read_metadata(data_dir / "user_metadata.csv", ["user_id"])
    career_metadata = read_metadata(career_file(data_dir, "career_metadata.csv"), [CODE, "Title"])
    validate_embeddings(users, user_metadata, 20, "Users")
    validate_embeddings(careers, career_metadata, 1016, "Careers")

    # Counts alone cannot detect reordered metadata. Compare every ID/title
    # against the source profile order used by scripts 02 and 04.
    if user_metadata != read_metadata(data_dir / "processed_user_profiles.csv", ["user_id"]):
        raise ValueError("User metadata order differs from processed user profiles")
    if career_metadata != read_metadata(career_file(data_dir, "processed_career_profiles.csv"), [CODE, "Title"]):
        raise ValueError("Career metadata order differs from processed career profiles")
    user_stats = json.loads((data_dir / "user_embedding_stats.json").read_text(encoding="utf-8"))
    career_stats = json.loads((data_dir / "career_embedding_stats.json").read_text(encoding="utf-8"))
    for key in ("model", "revision"):
        if not user_stats.get(key) or user_stats[key] != career_stats.get(key):
            raise ValueError(f"User and career embeddings have incompatible {key}")

    results = rank_matches(users, careers, user_metadata, career_metadata)
    if len(results) != 100:
        raise ValueError("Expected 100 recommendations")
    for user in user_metadata:
        matches = [row for row in results if row["user_id"] == user["user_id"]]
        if ([row["rank"] for row in matches] != [1, 2, 3, 4, 5]
                or len({row["career_code"] for row in matches}) != 5
                or any(a["cosine_similarity"] < b["cosine_similarity"] for a, b in zip(matches, matches[1:]))):
            raise ValueError(f"Invalid recommendations for {user['user_id']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print("Verified 20 users x 1,016 careers; model, dimensions, normalization, and metadata order match.")
    for index in range(0, len(results), 5):
        print(f"\n{results[index]['user_id']}")
        for row in results[index:index + 5]:
            print(f"  {row['rank']}. {row['career_title']} ({row['career_code']}) "
                  f"- cosine similarity {row['cosine_similarity']:.4f}")
    print(f"\nSaved {len(results)} recommendations (5 per user) to {args.output}")


if __name__ == "__main__":
    main()
