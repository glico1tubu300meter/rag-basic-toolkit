"""
Minimal RAG query script: retrieve passages from a FAISS index built by
build_index.py, and (optionally) ask an LLM to answer a question grounded
in the retrieved passages.

Usage:
    conda activate lora

    # retrieval only (no GPU-heavy LLM needed)
    python query.py --index ../out --question "質問文をここに" --no_generate

    # retrieval + grounded generation
    python query.py --index ../out --question "質問文をここに"
"""

import argparse
import json
import os
import sys

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

GROUNDED_SYSTEM_PROMPT = (
    "あなたは検索結果に基づいて質問に答えるアシスタントです。"
    "以下の参考情報だけを根拠に回答してください。"
    "参考情報に答えが含まれていない場合は、推測で補わず「資料からは分かりません」と述べてください。"
)


def parse_args():
    p = argparse.ArgumentParser(description="Query a FAISS RAG index, optionally with grounded generation.")
    p.add_argument("--index", required=True, help="Directory containing faiss.index and metadata.jsonl.")
    p.add_argument("--question", required=True, help="Question to ask.")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument(
        "--embed_model", default="intfloat/multilingual-e5-large",
        help="Must match the model used in build_index.py.",
    )
    p.add_argument("--gen_model", default=GEN_MODEL_NAME, help="LLM used for grounded generation.")
    p.add_argument("--no_generate", action="store_true", help="Only show retrieved passages, skip LLM generation.")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'.")
    return p.parse_args()


def load_index(index_dir):
    index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    meta = []
    with open(os.path.join(index_dir, "metadata.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return index, meta


def retrieve(index, meta, embed_model, query, top_k):
    query_vec = embed_model.encode(["query: " + query], normalize_embeddings=True, convert_to_numpy=True)
    scores, indices = index.search(query_vec, top_k)
    return [
        {**meta[idx], "score": float(score)}
        for score, idx in zip(scores[0], indices[0])
        if idx != -1
    ]


def format_passages(passages):
    return "\n\n".join(f"[{i + 1}] {p['title']}: {p['text']}" for i, p in enumerate(passages))


def generate_answer(question, passages, gen_model_name, max_new_tokens):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.float16,
        disable_mmap=True,
    )

    user_content = f"### 参考情報\n{format_passages(passages)}\n\n### 質問\n{question}"
    messages = [
        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

    generated = output[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    args = parse_args()

    index, meta = load_index(args.index)
    embed_model = SentenceTransformer(args.embed_model, device=args.device)

    passages = retrieve(index, meta, embed_model, args.question, args.top_k)

    print(f"【検索結果】(query: {args.question})\n{format_passages(passages)}\n")

    if not args.no_generate:
        answer = generate_answer(args.question, passages, args.gen_model, args.max_new_tokens)
        print(f"【回答】\n{answer}\n")


if __name__ == "__main__":
    main()
