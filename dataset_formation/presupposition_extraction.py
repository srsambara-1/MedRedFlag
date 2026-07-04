#!/usr/bin/env python3
"""Extract false presuppositions that triggered redirection."""

import argparse
import json
import os
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_openai_client, call_openai, get_processed_ids, safe_json_parse, stream_write_row

SYSTEM_PROMPT = """
You are a medical communication analyst specializing in detecting *premise replacement* in redirection cases.

Your task: **Identify the specific false or unsafe presupposition(s) in the patient's question that the physician rejects, corrects, or replaces in their response.**

A presupposition is a claim the patient *assumes to be true* in asking the question.
Extract only presuppositions that the physician's reply **explicitly or implicitly negates, corrects, or reframes**.

If multiple false presuppositions exist, output each on its own line.

Do NOT:
- Rephrase or summarize the patient's question.
- Describe the physician's reasoning or answer.
- Output explanations or commentary.
- Add anything besides the presupposition(s).

### EXAMPLES

[!!!!!! REDACTED !!!!!!]

### END OF EXAMPLES
"""

CLEAN_PRESUP_SYSTEM = """You are helping to clean presuppositions in a medical dataset. Some compound presuppositions have been undesirably broken into several separate ones. Your task is to identify and merge them when appropriate.

Example of the problem:
Input: [!!!!!! REDACTED !!!!!!]
Output: [!!!!!! REDACTED !!!!!!]

The idea is to combine presuppositions that present alternative explanations for the same phenomenon into a single presupposition that acknowledges the alternatives.

Return ONLY a JSON object with this exact format:
- If compound presuppositions are detected: {"compound_detected": 1, "cleaned": [cleaned presuppositions array]}
- If NO compound presuppositions are detected: {"compound_detected": 0, "cleaned": []}

No additional explanation or formatting. Only the JSON object."""

def TASK_PROMPT(patient_question, physician_response):
    return f"""
PATIENT QUESTION: {patient_question}
PHYSICIAN RESPONSE: {physician_response}

Extract the false or harmful presupposition(s) in the patient's question that the physician is redirecting.
Respond with only the presupposition(s), one per line if multiple.
"""


def extract_presuppositions(client, model, patient_question, physician_response):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TASK_PROMPT(patient_question, physician_response)}
    ]

    content, err = call_openai(client, model, messages)
    if err:
        print(f"API error: {err}")
        return [], err

    lines = [p.strip("- ").strip() for p in content.split("\n") if p.strip()]
    return lines, None


def clean_presuppositions(client, model, presup_text):
    if not presup_text or not presup_text.strip():
        return presup_text, 0

    try:
        presup_list = json.loads(presup_text)
        if not isinstance(presup_list, list) or len(presup_list) == 0:
            return presup_text, 0
    except json.JSONDecodeError:
        return presup_text, 0

    prompt = f"""Now, please review these presuppositions and determine if compound presuppositions need to be merged.

Input presuppositions:
{presup_text}"""

    messages = [
        {"role": "system", "content": CLEAN_PRESUP_SYSTEM},
        {"role": "user", "content": prompt}
    ]

    content, err = call_openai(client, model, messages)
    if err:
        return presup_text, 0

    result, parse_err = safe_json_parse(content)
    if parse_err:
        return presup_text, 0

    compound = result.get("compound_detected", 0)
    cleaned = result.get("cleaned", [])

    if compound == 1 and len(cleaned) > 0:
        return json.dumps(cleaned), 1
    return presup_text, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--model", default="gpt-5", help="Model to use")
    parser.add_argument("--max-rows", type=int, help="Max rows to process")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--question-col", default="patient_question", help="Column name for patient question")
    parser.add_argument("--response-col", default="physician_response", help="Column name for physician response")
    args = parser.parse_args()

    client = get_openai_client(args.api_key)
    df = pd.read_csv(args.input)

    if args.max_rows:
        df = df.head(args.max_rows)

    print(f"Loaded {len(df)} rows")

    done_ids = set()
    if args.resume:
        done_ids = get_processed_ids(args.output, ["redirection_id"])
        print(f"Resuming, {len(done_ids)} already done")

    write_header = not args.resume or not os.path.exists(args.output)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for idx, row in df.iterrows():
        rid = row.get("redirection_id")
        if rid in done_ids:
            continue

        patient_question = str(row.get(args.question_col, ""))
        physician_response = str(row.get(args.response_col, ""))

        print(f"[{idx+1}/{len(df)}] Processing redirection_id={rid}")
        presupps, err = extract_presuppositions(client, args.model, patient_question, physician_response)

        out_row = row.to_dict()
        out_row["presuppositions"] = json.dumps(presupps)
        cleaned, compound_detected = clean_presuppositions(client, args.model, out_row["presuppositions"])
        out_row["cleaned_presuppositions"] = cleaned
        out_row["compound_presup_detected"] = compound_detected
        if err:
            out_row["error"] = err

        stream_write_row(out_row, args.output, write_header)
        write_header = False

    print(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
