"""
Agentic RAG query script, implemented as a LangGraph graph: instead of a
single retrieve-then-generate pass (see query.py), three roles cooperate:

  1. search  - retrieves passages from the FAISS index for the current query
  2. critic  - reads the question + retrieved passages, judges whether they
               are sufficient, and proposes a follow-up search query if not
  3. judge   - writes the final answer once the critic is satisfied (or the
               round limit is reached), grounded in all retrieved passages

Unlike a plain if-chain, LangGraph lets this be a real cycle
(search -> critic -> search -> critic -> ... -> judge), so the critic can
trigger up to MAX_ROUNDS re-searches instead of just one.

Usage:
    conda activate lora
    python agentic_query.py --index ../out --question "質問文をここに"
"""

import argparse
import sys

import faiss
import torch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from typing_extensions import TypedDict

from query import load_index  # reuse the index loader from query.py

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TOP_K = 3
MAX_ROUNDS = 2

CRITIC_SYSTEM_PROMPT = (
    "あなたは回答の質をチェックする批評担当です。質問と検索結果の一覧を読み、"
    "検索結果だけで質問に十分答えられるか評価してください。"
    "質問に対する直接的な答えが検索結果の中に(たとえ1文だけでも)含まれている場合は、"
    "背景や理由などの周辺情報が薄くても『十分』と判定し、追加検索クエリは『追加検索クエリ: なし』としてください。"
    "追加検索が必要なのは、質問への直接的な答えが検索結果のどこにも見当たらない場合だけです。"
    "情報が本当に不足・矛盾している場合のみ具体的に指摘し、"
    "最後の行に『追加検索クエリ: <検索語>』の形式で検索クエリを1つ提案してください。"
)
JUDGE_SYSTEM_PROMPT = (
    "あなたは最終回答をまとめる判定担当です。検索結果と批評担当の指摘を踏まえ、"
    "検索結果に基づいた正確な回答を日本語で作成してください。"
    "検索結果の中に質問へ直接答えている記述があれば、批評担当が『情報不足』と述べていても、"
    "その記述を優先してそのまま採用してください。"
    "検索結果のどこにも該当する記述が見当たらない場合に限り、"
    "推測で補わず『資料からは分かりません』と述べてください。"
)


class GraphState(TypedDict):
    question: str
    query: str
    passages: list[dict]
    critic_text: str
    round: int
    answer: str


def parse_args():
    p = argparse.ArgumentParser(description="Agentic (searcher/critic/judge) RAG query via LangGraph.")
    p.add_argument("--index", required=True, help="Directory containing faiss.index and metadata.jsonl.")
    p.add_argument("--question", required=True)
    p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--max_rounds", type=int, default=MAX_ROUNDS)
    p.add_argument("--embed_model", default="intfloat/multilingual-e5-large")
    p.add_argument("--gen_model", default=GEN_MODEL_NAME)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_llm(gen_model_name):
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
    hf_pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=350, do_sample=True, temperature=0.5, top_p=0.9, return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    return ChatHuggingFace(llm=llm, tokenizer=tokenizer)


def format_passages(passages):
    return "\n\n".join(f"[{i + 1}] {p['title']}: {p['text']}" for i, p in enumerate(passages))


def extract_followup_query(critic_text, fallback):
    for line in critic_text.splitlines():
        if "追加検索クエリ" in line:
            q = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            if q:
                return q
    return fallback


def build_graph(chat_model, index, meta, embed_model, top_k, max_rounds):

    def search_node(state: GraphState) -> GraphState:
        query_vec = embed_model.encode(["query: " + state["query"]], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = index.search(query_vec, top_k)
        new_passages = [
            {**meta[idx], "score": float(s)} for s, idx in zip(scores[0], indices[0]) if idx != -1
        ]
        print(f"【検索者 (round {state['round']})】query={state['query']!r}\n{format_passages(new_passages)}\n")
        return {**state, "passages": state["passages"] + new_passages}

    def critic_node(state: GraphState) -> GraphState:
        user_content = f"質問: {state['question']}\n\n検索結果:\n{format_passages(state['passages'])}"
        response = chat_model.invoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT), HumanMessage(content=user_content),
        ])
        critic_text = response.content
        print(f"【批評者 (round {state['round']})】\n{critic_text}\n")
        return {**state, "critic_text": critic_text}

    def route_after_critic(state: GraphState) -> str:
        followup_query = extract_followup_query(state["critic_text"], "")
        if state["round"] >= max_rounds or followup_query.strip() in ("なし", "無し", ""):
            return "judge"
        return "research_again"

    def prepare_next_round(state: GraphState) -> GraphState:
        followup_query = extract_followup_query(state["critic_text"], state["question"])
        return {**state, "query": followup_query, "round": state["round"] + 1}

    def judge_node(state: GraphState) -> GraphState:
        user_content = (
            f"質問: {state['question']}\n\n検索結果(全ラウンド分):\n{format_passages(state['passages'])}\n\n"
            f"批評担当の指摘:\n{state['critic_text']}\n\n上記を踏まえて最終回答を書いてください。"
        )
        response = chat_model.invoke([
            SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=user_content),
        ])
        print(f"【判定者】\n{response.content}\n")
        return {**state, "answer": response.content}

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("search", search_node)
    graph_builder.add_node("critic", critic_node)
    graph_builder.add_node("prepare_next_round", prepare_next_round)
    graph_builder.add_node("judge", judge_node)

    graph_builder.add_edge(START, "search")
    graph_builder.add_edge("search", "critic")
    graph_builder.add_conditional_edges(
        "critic", route_after_critic, {"research_again": "prepare_next_round", "judge": "judge"}
    )
    graph_builder.add_edge("prepare_next_round", "search")
    graph_builder.add_edge("judge", END)

    return graph_builder.compile()


def main():
    args = parse_args()

    print("Loading index...")
    index, meta = load_index(args.index)
    embed_model = SentenceTransformer(args.embed_model, device=args.device)

    print(f"Loading LLM: {args.gen_model}")
    chat_model = load_llm(args.gen_model)

    graph = build_graph(chat_model, index, meta, embed_model, args.top_k, args.max_rounds)

    initial_state: GraphState = {
        "question": args.question, "query": args.question, "passages": [],
        "critic_text": "", "round": 0, "answer": "",
    }
    result = graph.invoke(initial_state)

    print("=== 最終回答 ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
