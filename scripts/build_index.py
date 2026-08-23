"""
Minimal, reusable RAG index builder: embed a JSONL corpus of passages
with a local sentence-transformers model and build a FAISS index.

Unlike the earlier anime-Wikipedia experiment, this script contains no
franchise-specific content or data-collection logic (Wikidata/MediaWiki
scraping) — the corpus is supplied externally as a plain JSONL file.

Corpus format (JSONL, one passage per line):
    {"title": "<source title>", "text": "<passage text>"}

`title` is used only for display/citation; `text` is what gets embedded
and retrieved. Chunk long documents into passages yourself before
building the index (a few hundred characters each works well).

Usage:
    conda activate lora
    python build_index.py --data ../data/sample_corpus.jsonl --output_dir ../out
"""

import argparse
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args():
    p = argparse.ArgumentParser(description="Build a local FAISS RAG index from a JSONL corpus.")
    p.add_argument("--data", required=True, help="Path to a JSONL file with {'title','text'} per line.")
    p.add_argument("--output_dir", required=True, help="Where to save the FAISS index and metadata.")
    p.add_argument(
        "--embed_model", default="intfloat/multilingual-e5-large",
        help="sentence-transformers model used to embed passages (must match query.py's --embed_model).",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'.")
    return p.parse_args()


def load_corpus(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append({"title": row.get("title", ""), "text": row["text"]})
    if not rows:
        raise ValueError(f"No passages found in {path}")
    return rows


def main():
    args = parse_args()

    rows = load_corpus(args.data)
    print(f"Loaded {len(rows)} passages from {args.data}")

    print(f"Loading embedding model: {args.embed_model}")
    model = SentenceTransformer(args.embed_model, device=args.device)

    # e5 models expect a "passage: " / "query: " prefix convention.
    passages = ["passage: " + r["text"] for r in rows]
    embeddings = model.encode(
        passages, batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, "faiss.index")
    meta_path = os.path.join(args.output_dir, "metadata.jsonl")

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nBuilt index with {len(rows)} passages (dim={dim}).")
    print(f"Saved index to:    {index_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
