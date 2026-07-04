#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_CSV="PATH_TO_INPUT.csv"
BASELINE_OUT="PATH_TO_BASELINE_OUTPUT.csv"
IDENTIFY_OUT="PATH_TO_IDENTIFY_OUTPUT.csv"
ORACLE_OUT="PATH_TO_ORACLE_OUTPUT.csv"

MODEL="MODEL_NAME"
API_TYPE="API_TYPE"

VLLM_HOST="localhost"
VLLM_PORT="8000"

case "$API_TYPE" in
  openai|vllm) : "${OPENAI_API_KEY:?Set OPENAI_API_KEY for baseline generation}";;
  anthropic) : "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY for baseline generation}";;
  *) echo "Unsupported baseline API: $API_TYPE"; exit 1;;
esac

mkdir -p "$(dirname "$BASELINE_OUT")" "$(dirname "$IDENTIFY_OUT")" "$(dirname "$ORACLE_OUT")"

# baseline.py arguments additional arguments:
#     --output-column <name>   # override column storing baseline answers
#     --max-rows <int>         # only score the first N rows
#     --resume                 # continue writing to an existing CSV without repeats
python "$DIR/baseline.py" \
  --input "$INPUT_CSV" \
  --output "$BASELINE_OUT" \
  --model "$MODEL" \
  --api-type "$API_TYPE" \
  --api-key "${OPENAI_API_KEY:-${ANTHROPIC_API_KEY:-}}" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT"

# identify_and_respond.py additional arguments:
#     --output-column <name>   # rename the identify response column
#     --max-rows <int>         # restrict how many patient questions are processed
#     --resume                 # append without reprocessing IDs already in the output
python "$DIR/identify_and_respond.py" \
  --input "$BASELINE_OUT" \
  --output "$IDENTIFY_OUT" \
  --model "$MODEL" \
  --api-type "$API_TYPE" \
  --api-key "${OPENAI_API_KEY:-${ANTHROPIC_API_KEY:-}}" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT"

# oracle_assumptions_provided.py additional arguments:
#     --assumption-column <name>  # choose which column of presuppositions to feed in
#     --output-column <name>      # rename the oracle response column
#     --max-rows <int>            # limit number of rows processed
#     --resume                    # skip redirection IDs already present in the output
python "$DIR/oracle_assumptions_provided.py" \
  --input "$IDENTIFY_OUT" \
  --output "$ORACLE_OUT" \
  --model "$MODEL" \
  --api-type "$API_TYPE" \
  --api-key "${OPENAI_API_KEY:-${ANTHROPIC_API_KEY:-}}" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT"

echo "Baseline output: $BASELINE_OUT"
echo "Identify output: $IDENTIFY_OUT"
echo "Oracle output: $ORACLE_OUT"
