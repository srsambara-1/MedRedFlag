#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for evaluation}"

EVAL_INPUT="PATH_TO_INPUT.csv"
ADDRESSED_OUT="PATH_TO_ADDRESSED.csv"
ADDRESSED_METRICS="PATH_TO_ADDRESSED_STATS.csv"
ACCOM_OUT="PATH_TO_ACCOMMODATED.csv"
ACCOM_METRICS="PATH_TO_ACCOMMODATED_STATS.csv"
ANSWER_COLUMN="MODEL_ANSWER_COLUMN"

# eval_addressed.py other arguments:
#   --model (default gpt-5): llm judge
#   --assumption-column (default cleaned_presuppositions): column holding problematic assumptions
#   --max-rows (optional): limit rows processed
#   --resume (flag): resume from existing output
#   --api-key (optional): OpenAI key; falls back to env var
python "$DIR/eval_addressed.py" \
  --input "$EVAL_INPUT" \
  --output "$ADDRESSED_OUT" \
  --metrics-output "$ADDRESSED_METRICS" \
  --answer-column "$ANSWER_COLUMN" \
  --api-key "$OPENAI_API_KEY"

# eval_accommodated.py arguments (from script):
#   --model (default gpt-5): llm judge
#   --support-column (default support_condition): column with support questions; in MedRedFlag this is clinician-written
#   --max-rows (optional): limit rows processed
#   --resume (flag): resume from existing output
#   --api-key (optional): OpenAI key; falls back to env var
python "$DIR/eval_accommodated.py" \
  --input "$EVAL_INPUT" \
  --output "$ACCOM_OUT" \
  --metrics-output "$ACCOM_METRICS" \
  --answer-column "$ANSWER_COLUMN" \
  --api-key "$OPENAI_API_KEY"

echo "Addressed scores: $ADDRESSED_OUT"
echo "Addressed metrics: $ADDRESSED_METRICS"
echo "Accommodated scores: $ACCOM_OUT"
echo "Accommodated metrics: $ACCOM_METRICS"
