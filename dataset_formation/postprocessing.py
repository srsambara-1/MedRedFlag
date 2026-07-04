#!/usr/bin/env python3
"""Post-processing: detect missing context in physician responses."""

import argparse
import json
import os
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_openai_client, call_openai, safe_json_parse, get_processed_ids, stream_write_row

# Prompt for detecting missing context
MISSING_CONTEXT_SYSTEM = """
You evaluate whether a physician's response relies on conversational context
that is NOT present in the patient's question.

Return your answer STRICTLY as valid JSON:

{
  "responds_to_missing_context": <0 or 1>,
  "missing_context_evidence": [<sentences or phrases indicating reliance on missing context>]
}

DEFINITION — RESPONDS TO MISSING CONTEXT:
The physician responds to missing context ONLY IF their reply:
- References or responds to specific claims, beliefs, actions, or prior replies
  that the patient did NOT state
- Responds as if to a prior message or exchange that is not present
  in the patient's question

Rules:
- If responds_to_missing_context = 0 → missing_context_evidence MUST be []
- Only flag missing context when the response clearly depends on unstated prior content
- Do NOT infer intent; rely only on explicit text
- Do NOT include any extra text outside the JSON object
"""


def detect_missing_context(client, model, patient_question, physician_response):
    messages = [
        {"role": "system", "content": MISSING_CONTEXT_SYSTEM},
        {"role": "user", "content": f"PATIENT QUESTION:\n{patient_question}\n\nPHYSICIAN RESPONSE:\n{physician_response}\n\nReturn the JSON object now."}
    ]

    content, err = call_openai(client, model, messages)
    if err:
        return {"responds_to_missing_context": None, "missing_context_evidence": []}

    result, parse_err = safe_json_parse(content)
    if parse_err:
        return {"responds_to_missing_context": None, "missing_context_evidence": []}

    return {
        "responds_to_missing_context": int(result.get("responds_to_missing_context", 0)),
        "missing_context_evidence": result.get("missing_context_evidence", [])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--model", default="gpt-5", help="Model to use")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--skip-missing-context", action="store_true", help="Skip missing context detection")
    parser.add_argument("--api-key", help="OpenAI API key")
    args = parser.parse_args()

    client = get_openai_client(args.api_key)
    df = pd.read_csv(args.input)

    print(f"Loaded {len(df)} rows")

    done_ids = set()
    if args.resume:
        done_ids = get_processed_ids(args.output, ["redirection_id"])
        print(f"Resuming, {len(done_ids)} already done")

    write_header = not args.resume or not os.path.exists(args.output)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for idx, row in df.iterrows():
        rid = row.get("redirection_id", idx)
        if rid in done_ids:
            continue

        print(f"[{idx+1}/{len(df)}] Processing {rid}")

        out_row = row.to_dict()

        # Missing context detection
        if not args.skip_missing_context:
            pq = str(row.get("patient_question", ""))
            pr = str(row.get("physician_response", ""))
            mc_result = detect_missing_context(client, args.model, pq, pr)
            out_row["responds_to_missing_context"] = mc_result["responds_to_missing_context"]
            out_row["missing_context_evidence"] = json.dumps(mc_result["missing_context_evidence"])

        stream_write_row(out_row, args.output, write_header)
        write_header = False

    print(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
