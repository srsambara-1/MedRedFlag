#!/usr/bin/env bash
set -euo pipefail

# Run from the root of your MedRAG clone (copy this folder's scripts there first;
# see README.md). Unlike the other pipeline runners, the scripts are executed in
# place from the MedRAG root, so no self-locating DIR is used.

INPUT_CSV="PATH_TO_INPUT.csv"
RETRIEVAL_OUT="PATH_TO_RETRIEVAL_RESULTS.json"
RAG_OUT="PATH_TO_RAG_OUTPUT.csv"

MODEL="MODEL_NAME"
API_TYPE="API_TYPE"

VLLM_HOST="localhost"
VLLM_PORT="8000"

RETRIEVER="RRF-2"
CORPUS="MedText"
TOP_K=5

case "$API_TYPE" in
  openai|vllm) : "${OPENAI_API_KEY:?Set OPENAI_API_KEY for RAG generation}";;
  anthropic) : "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY for RAG generation}";;
  *) echo "Unsupported API: $API_TYPE"; exit 1;;
esac

mkdir -p "$(dirname "$RETRIEVAL_OUT")" "$(dirname "$RAG_OUT")"

# run_retrieval.py additional arguments:
#     --n_rows <int>               # only retrieve for the first N rows
#     --k <int>                    # documents to retrieve per query (default 32)
#     --db_dir <path>              # corpus/index directory (default ./corpus)
python run_retrieval.py \
  --input "$INPUT_CSV" \
  --output "$RETRIEVAL_OUT" \
  --retriever "$RETRIEVER" \
  --corpus "$CORPUS" \
  --k 32

# rag.py additional arguments:
#     --output-column <name>   # rename the RAG response column (default model_answer_rag)
#     --max-tokens <int>       # cap response length (default 2048)
#     --temperature <float>    # sampling temperature (default 0.0)
#     --max-rows <int>         # only process the first N rows
#     --resume                 # append without reprocessing rows already in the output
python rag.py \
  --input "$INPUT_CSV" \
  --output "$RAG_OUT" \
  --retrieval-results "$RETRIEVAL_OUT" \
  --model "$MODEL" \
  --api-type "$API_TYPE" \
  --api-key "${OPENAI_API_KEY:-${ANTHROPIC_API_KEY:-}}" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --top-k "$TOP_K"

echo "Retrieval output: $RETRIEVAL_OUT"
echo "RAG output: $RAG_OUT"
