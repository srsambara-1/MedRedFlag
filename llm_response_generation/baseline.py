#!/usr/bin/env python3
"""Generate baseline LLM responses to patient questions."""

import argparse
import os
import re
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    get_openai_client, get_anthropic_client, get_vllm_client,
    call_openai, call_anthropic, get_processed_ids, stream_write_row
)

SYSTEM_PROMPT = "You are a helpful medical assistant. Answer this patient's question to the best of your ability."


def generate_answer(client, model, patient_question, api_type):
    if api_type == "anthropic":
        return call_anthropic(client, model, SYSTEM_PROMPT, patient_question)
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": patient_question}
        ]
        return call_openai(client, model, messages)


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
    output_column = args.output_column or f"{model_slug}_answer"

    base_prefix = output_column if args.output_column else model_slug
    think_col = f"{base_prefix}_think"
    error_col = f"{output_column}_error"

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

        answer, err = generate_answer(client, args.model, patient_question, args.api_type)

        out_row = row.to_dict()
        if err:
            out_row[output_column] = f"[ERROR: {err}]"
            out_row[error_col] = err
            print(f"Warning: {rid} error: {err} (row {idx})")
        elif not answer:
            empty_msg = "Empty completion returned by API"
            print(f"Warning: {rid} produced an empty completion (row {idx})")
            out_row[output_column] = f"[ERROR: {empty_msg}]"
            out_row[error_col] = empty_msg
        else:
            # Extract <think>...</think> block if present (e.g. Qwen3 thinking mode)
            think_match = re.search(r"<think>(.*?)</think>", answer, flags=re.DOTALL)
            think_text = think_match.group(1).strip() if think_match else None
            cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

            if think_text:
                out_row[think_col] = think_text
            out_row[output_column] = cleaned or answer

        stream_write_row(out_row, args.output, write_header)
        write_header = False

    print(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
