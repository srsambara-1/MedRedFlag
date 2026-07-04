#!/usr/bin/env python3
"""Identify and Respond: Model is asked to identify false assumptions first, then responds."""

import argparse
import os
import re
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    get_openai_client, get_anthropic_client, get_vllm_client,
    call_openai, call_anthropic, safe_json_parse, get_processed_ids, stream_write_row
)

SYSTEM_PROMPT = (
    "You are a helpful medical assistant.\n"
    "Follow the instructions carefully and output ONLY valid JSON."
)


def strategy_identify_prompt(patient_question):
    return f"""
You must output VALID JSON and nothing else.

Schema:
{{
  "assumptions": [string],
  "response": string
}}

Instructions:
- Identify any false or unsupported assumptions in the patient's question. If none exist, return an empty list for "assumptions".
- Then respond to the patient's question with those assumptions in mind.

Patient question:
{patient_question}
""".strip()


def run_strategy_identify(client, model, patient_question, api_type):
    prompt = strategy_identify_prompt(patient_question)

    if api_type == "anthropic":
        content, err = call_anthropic(client, model, SYSTEM_PROMPT, prompt)
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        content, err = call_openai(client, model, messages)

    if err:
        return {"assumptions": None, "response": None, "error": err}

    if not content:
        return {"assumptions": None, "response": None, "error": "Empty completion returned by API"}

    # Extract <think>...</think> block if present (e.g. Qwen3 thinking mode)
    think_match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else None
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    result, parse_err = safe_json_parse(cleaned or content)
    if parse_err:
        return {"assumptions": None, "response": None, "think": think_text, "error": f"JSON parse error: {parse_err}", "raw_output": content}

    return {
        "assumptions": result.get("assumptions"),
        "response": result.get("response"),
        "think": think_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-type", default="openai", choices=["openai", "anthropic", "vllm"])
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--host", default="localhost", help="vLLM host")
    parser.add_argument("--port", default="8000", help="vLLM port")
    parser.add_argument("--output-column", default=None, help="Column name for output")
    parser.add_argument("--max-rows", type=int, help="Max rows to process")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")

    args = parser.parse_args()
    model_base = args.model.split("/")[-1]
    model_slug = re.sub(r"[^0-9a-zA-Z]+", "_", model_base).strip("_").lower() or "model"
    output_column = args.output_column or f"{model_slug}_answer_identifyAndRespond"
    
    base_prefix = output_column if args.output_column else model_slug
    assumptions_col = f"{base_prefix}_assumptions"
    think_col = f"{base_prefix}_think"
    error_col = f"{base_prefix}_error"

    if args.api_type == "openai":
        client = get_openai_client(args.api_key)
    elif args.api_type == "anthropic":
        client = get_anthropic_client(args.api_key)
    else:
        client = get_vllm_client(args.host, args.port)

    df = pd.read_csv(args.input)
    if args.max_rows:
        df = df.head(args.max_rows)

    print(f"Loaded {len(df)} rows, model={args.model}, api={args.api_type}")

    done_keys = set()
    if args.resume:
        done_keys = get_processed_ids(args.output, ["redirection_id"])
        print(f"Resuming, {len(done_keys)} already done")

    write_header = not args.resume or not os.path.exists(args.output)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for idx, row in df.iterrows():
        rid = row.get("redirection_id")
        if rid in done_keys:
            continue

        patient_question = str(row.get("patient_question", ""))
        print(f"[{idx+1}/{len(df)}] Processing {rid}")

        result = run_strategy_identify(client, args.model, patient_question, args.api_type)

        out_row = row.to_dict()
        err = result.get("error")
        if result.get("think"):
            out_row[think_col] = result["think"]
        if err:
            out_row[output_column] = f"[ERROR: {err}]"
            out_row[assumptions_col] = f"[ERROR: {err}]"
            out_row[error_col] = err
            if result.get("raw_output"):
                out_row[f"{base_prefix}_raw_output"] = result["raw_output"]
            print(f"Warning: {rid} error: {err} (row {idx})")
        else:
            out_row[assumptions_col] = result.get("assumptions")
            out_row[output_column] = result.get("response")

        stream_write_row(out_row, args.output, write_header)
        write_header = False

    print(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
