# rag-basic-toolkit

RAG(Retrieval-Augmented Generation)の最小構成ツールキット。外部知識をローカルの埋め込み+FAISSインデックスで検索し、検索結果に基づいてLLMが回答を生成する。コーパスは外部のJSONLファイルとして自分で用意する形式で、特定の作品などの権利に関わる内容は一切含まれていない。

対象モデルはデフォルトで [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)(回答生成)と [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)(埋め込み)。いずれもオプションで変更できる。

## 動作環境

- NVIDIA GPU (CUDA)、VRAM 11GB以上を推奨(7Bモデル・4bit量子化での動作確認は GTX 1080 Ti で実施)
- 検索のみ(`--no_generate`)であれば、埋め込みモデルのみロードするため要求VRAMはかなり少なくて済む
- Python 3.11

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate   # Windows は .venv\Scripts\activate
pip install -r requirements.txt
```

## データ形式

コーパスは JSONL(1行1パッセージ)、`title` と `text` のペア:

```json
{"title": "検索結果に表示する見出し", "text": "埋め込み・検索対象の本文"}
```

`data/sample_corpus.jsonl` に、フォーマット確認用の汎用サンプル(8件、LoRA/RAG関連の一般的な技術解説、著作権フリーの自作文章)を同梱している。実際に使う際は、自前のドキュメントをこの形式に変換して差し替えること(長い文書は数百文字程度のパッセージに分割してから1行ずつ入れる)。

## 使い方

### 1. インデックス構築

```bash
cd scripts
python build_index.py --data ../data/sample_corpus.jsonl --output_dir ../out
```

`<output_dir>/faiss.index` と `<output_dir>/metadata.jsonl` が生成される。

### 2. 検索(+回答生成)

```bash
# 検索のみ(LLMをロードしない、軽量)
python query.py --index ../out --question "質問文をここに" --no_generate

# 検索+検索結果に基づく回答生成
python query.py --index ../out --question "質問文をここに"
```

## 実行イメージ

### 例1: コーパスがカバーしている質問

```
$ python query.py --index ../out --question "RAGとは何ですか？"

【検索結果】(query: RAGとは何ですか？)
[1] RAG: RAG(Retrieval-Augmented Generation)は、大規模言語モデルの回答生成に、外部知識源
からの検索結果を組み合わせる手法。モデル自体のパラメータを変更せずに最新情報や専門知識を
反映できるため、事実に基づいた正確な回答を生成しやすくなる。

[2] FAISS: FAISSはMeta(旧Facebook)が開発した、高速な近似最近傍探索ライブラリ。...

[3] LoRA: LoRA(Low-Rank Adaptation)は、大規模言語モデルを効率的にファインチューニング
する手法の一つ。...

【回答】
RAG(Retrieval-Augmented Generation)は、大規模言語モデルの回答生成に、外部知識源からの
検索結果を組み合わせる手法です。これにより、モデル自体のパラメータを変更せずに最新情報
や専門知識を反映し、事実に基づいた正確な回答を生成しやすくなるという利点があります。
```

検索結果[1]の内容がほぼそのまま回答に反映されており、コーパスに基づいた正確な回答ができていることが分かる。

### 例2: コーパスがカバーしていない質問(誠実さのテスト)

```
$ python query.py --index ../out --question "フランス料理のおすすめレシピを教えて"

【検索結果】(query: フランス料理のおすすめレシピを教えて)
[1] FAISS: FAISSはMeta(旧Facebook)が開発した、高速な近似最近傍探索ライブラリ。...
[2] LoRA: LoRA(Low-Rank Adaptation)は、大規模言語モデルを効率的にファインチューニング...
[3] RAG: RAG(Retrieval-Augmented Generation)は、大規模言語モデルの回答生成に...

【回答】
資料からは分かりません。
```

コーパス(技術解説8件)に該当する情報が存在しないため、検索結果には無関係な文書しかヒットしない。この状態でモデルに単純に質問を投げると、モデルが持つ一般知識で推測混じりの回答をしてしまうことがあるが、`query.py` のプロンプトには「参考情報に無い場合は推測で補わず『資料からは分かりません』と述べる」という指示が入っており、その通りに正直な回答が返っている。

**この2つの例からわかること**: RAGの価値は「検索結果があれば正確に答えられる」ことだけでなく、**「検索結果が無ければ無理に答えない」という誠実さを引き出せる**点にもある。ただしこれはプロンプトの指示に依存する挙動であり、指示を外すと無関係な検索結果からもそれらしい回答を捏造することがある点には注意(`GROUNDED_SYSTEM_PROMPT` の指示文を参照)。

## 既知の注意点

- **Windows + mmap の相性問題**: ネットワークドライブ等にモデルキャッシュを置いた環境で、7B級モデルを `from_pretrained` でロードするとセグメンテーション違反が発生することがある。本スクリプトでは `disable_mmap=True` を指定して回避している。
- **埋め込みモデルの一致**: `build_index.py` と `query.py` は同じ `--embed_model` を使う必要がある(異なるモデルで埋め込んだベクトル同士は比較できない)。
- **検索精度はコーパスの粒度に依存**: パッセージが長すぎたり短すぎたりすると検索精度が落ちる。数百文字程度への分割が目安。

## ライセンス

このリポジトリのコード自体に付属する追加のライセンス条件はない。使用するベースモデル(Qwen2.5シリーズ、multilingual-e5等)のライセンス、および自分で用意するコーパスの権利については、各自で確認すること。
