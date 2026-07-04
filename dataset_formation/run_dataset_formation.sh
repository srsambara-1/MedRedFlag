#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for GPT-5 access}"

INPUT_DIR="PATH_TO_medredqa"
PREFILTER_OUT="PATH_TO_PREFILTER_OUTPUT.csv"
REDIR_ALL="PATH_TO_REDIRECTION_LABEL_OUTPUT.csv"
REDIR_ONLY="PATH_TO_REDIRECTION_CASES_OUTPUT.csv"
POSTPROC_OUT="PATH_TO_POSTPROCESS_OUTPUT.csv"
PRESUPP_OUT="PATH_TO_PRESUPP_OUTPUT.csv"
MODEL="gpt-5"

python "$DIR/prefilter.py" \
  --input_dir "$INPUT_DIR" \
  --output "$PREFILTER_OUT"

# additional arguments:
#     --max-rows <int>            # limit number of rows processed
#     --resume                    # skip redirection IDs already present in the output
python "$DIR/redirection_tagging.py" \
  --input "$PREFILTER_OUT" \
  --output-all "$REDIR_ALL" \
  --output-redirected "$REDIR_ONLY" \
  --model "$MODEL" \
  --api-key "$OPENAI_API_KEY"

# additional arguments:
#     --max-rows <int>            # limit number of rows processed
#     --resume                    # skip redirection IDs already present in the output
python "$DIR/postprocessing.py" \
  --input "$REDIR_ONLY" \
  --output "$POSTPROC_OUT" \
  --model "$MODEL" \
  --api-key "$OPENAI_API_KEY"

# additional arguments:
#     --max-rows <int>            # limit number of rows processed
#     --resume                    # skip redirection IDs already present in the output
python "$DIR/presupposition_extraction.py" \
  --input "$POSTPROC_OUT" \
  --output "$PRESUPP_OUT" \
  --model "$MODEL" \
  --api-key "$OPENAI_API_KEY"
