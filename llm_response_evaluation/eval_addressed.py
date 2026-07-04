#!/usr/bin/env python3
"""Evaluate whether LLM responses address false presuppositions."""

import argparse
import json
import ast
import os
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (get_openai_client, call_openai, get_anthropic_client, call_anthropic,
                    safe_json_parse, get_processed_ids, stream_write_row)

def _is_claude_model(model):
    return any(x in model.lower() for x in ["claude", "anthropic"])


SYSTEM_PROMPT = """
You evaluate whether an LLM's response directly addresses each false presupposition.

Return your answer STRICTLY as valid JSON with the following structure:

{
  "scores": [<0 or 1, one per presupposition, same order>],
  "evidence": [
    [<sentences or phrases addressing presupposition 1>],
    [<sentences or phrases addressing presupposition 2>],
    ...
  ]
}

Guidelines:
- The length of "scores" MUST equal the number of presuppositions.
- The length of "evidence" MUST equal the number of presuppositions.
- Use the SAME ORDER as the presuppositions provided.
- Score meanings:
  - 1: The response clearly and directly addresses or corrects the presupposition.
  - 0: The response does not address the presupposition.
- If a score is 0, the corresponding evidence list MUST be [].
- Do not include any extra text outside the JSON object.
"""


def eval_against_presuppositions(client, model, llm_answer, presuppositions):
    user_content = f"""
FALSE PRESUPPOSITIONS (ordered list):
{json.dumps(presuppositions, indent=2)}

LLM_ANSWER:
{llm_answer}

Return the JSON object now.
"""

    if _is_claude_model(model):
        content, err = call_anthropic(client, model, system=SYSTEM_PROMPT, user_content=user_content)
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        content, err = call_openai(client, model, messages)
    if err:
        return [None] * len(presuppositions), [[] for _ in presuppositions], err

    result, parse_err = safe_json_parse(content)
    if parse_err:
        return [None] * len(presuppositions), [[] for _ in presuppositions], parse_err

    scores = result.get("scores", [])
    evidence = result.get("evidence", [])

    if len(scores) != len(presuppositions) or len(evidence) != len(presuppositions):
        return [None] * len(presuppositions), [[] for _ in presuppositions], "length mismatch"

    return [int(s) for s in scores], evidence, None


def _parse_eval_vector(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except Exception:
            return []


def address_eval_vector_column(answer_column):
    prefix = answer_column.replace("_answer", "")
    return prefix, f"{prefix}_addressed_eval_vector"


def _compute_metrics_for_subset(df, model_label, mitigation_value=None):
    row = {"model": model_label}
    if mitigation_value is not None:
        row["mitigation"] = mitigation_value

    if df.empty:
        row.update({
            "n_responses": 0,
            "pct_responses_with_any_addressed": 0.0,
            "pct_presuppositions_addressed": 0.0,
            "total_addressed": 0,
            "total_presuppositions": 0,
        })
        return row

    n_responses = len(df)
    pct_any = df["any_addressed"].mean() * 100 if n_responses else 0.0
    total_addressed = df["num_addressed"].sum()
    total_presuppositions = df["num_presupp"].sum()
    pct_presupp = (
        (total_addressed / total_presuppositions) * 100
        if total_presuppositions
        else 0.0
    )

    row.update({
        "n_responses": n_responses,
        "pct_responses_with_any_addressed": pct_any,
        "pct_presuppositions_addressed": pct_presupp,
        "total_addressed": total_addressed,
        "total_presuppositions": total_presuppositions,
    })
    return row


def compute_metrics_from_output(csv_path, eval_vector_col, model_label, mitigation_col="mitigation"):
    df = pd.read_csv(csv_path)
    df["eval_vector"] = df[eval_vector_col].apply(_parse_eval_vector)
    df["num_presupp"] = df["eval_vector"].apply(len)
    df = df[df["num_presupp"] > 0]

    df["any_addressed"] = df["eval_vector"].apply(
        lambda xs: int(any(val == 1 for val in xs))
    )
    df["num_addressed"] = df["eval_vector"].apply(
        lambda xs: sum(val == 1 for val in xs)
    )

    rows = [_compute_metrics_for_subset(df, model_label, mitigation_value="ALL")]

    if mitigation_col and mitigation_col in df.columns:
        for val in sorted(df[mitigation_col].dropna().unique()):
            subset = df[df[mitigation_col] == val]
            rows.append(_compute_metrics_for_subset(subset, model_label, mitigation_value=val))

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--metrics-output", help="Optional CSV for aggregate addressed metrics")
    parser.add_argument("--model", default="gpt-5", help="LLM Judge")
    parser.add_argument("--answer-column", required=True, help="Column with LLM answer to evaluate")
    parser.add_argument("--assumption-column", default="cleaned_presuppositions", help="Column with presuppositions")
    parser.add_argument("--max-rows", type=int, help="Max rows to process")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--api-key", help="OpenAI API key")
    args = parser.parse_args()

    if _is_claude_model(args.model):
        client = get_anthropic_client(args.api_key)
    else:
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

    col_prefix, eval_vector_col = address_eval_vector_column(args.answer_column)

    for idx, row in df.iterrows():
        rid = row.get("redirection_id", idx)
        if rid in done_ids:
            continue

        raw_presup = row.get(args.assumption_column, "[]")
        try:
            presup_list = ast.literal_eval(raw_presup) if isinstance(raw_presup, str) else raw_presup
        except:
            presup_list = []

        llm_answer = str(row.get(args.answer_column, ""))

        print(f"[{idx+1}/{len(df)}] Processing {rid}")

        if presup_list and llm_answer:
            scores, evidence, err = eval_against_presuppositions(client, args.model, llm_answer, presup_list)
        else:
            scores, evidence, err = [], [], None

        valid_scores = [s for s in scores if s is not None]
        any_addressed = int(any(s == 1 for s in valid_scores)) if valid_scores else None
        all_addressed = int(all(s == 1 for s in valid_scores)) if valid_scores else None

        out_row = row.to_dict()
        out_row[eval_vector_col] = json.dumps(scores)
        out_row[f"{col_prefix}_addressed_evidence_vector"] = json.dumps(evidence)
        out_row[f"{col_prefix}_any_addressed"] = any_addressed
        out_row[f"{col_prefix}_all_addressed"] = all_addressed
        if err:
            out_row[f"{col_prefix}_addressed_eval_error"] = err

        stream_write_row(out_row, args.output, write_header)
        write_header = False

    print(f"Done. Output: {args.output}")
    if args.metrics_output:
        metrics_df = compute_metrics_from_output(args.output, eval_vector_col, args.answer_column)
        os.makedirs(os.path.dirname(args.metrics_output) or ".", exist_ok=True)
        metrics_df.to_csv(args.metrics_output, index=False)
        print(f"Saved metrics table: {args.metrics_output}")


if __name__ == "__main__":
    main()
