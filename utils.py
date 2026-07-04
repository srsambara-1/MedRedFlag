import os
import json
import pandas as pd
from openai import OpenAI
from anthropic import Anthropic


def get_openai_client(api_key=None):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OpenAI API key")
    return OpenAI(api_key=key)


def get_anthropic_client(api_key=None):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Missing Anthropic API key")
    return Anthropic(api_key=key)


def get_vllm_client(host="localhost", port="8000", api_key="EMPTY"):
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)


def call_openai(client, model, messages, temperature=0, max_tokens=None):
    """Returns (content, error). If error is None, call succeeded."""
    try:
        params = {
            "model": model,
            "messages": messages,
        }
        is_gpt5_or_similar = any(x in model.lower() for x in ["gpt-5", "gpt5", "gpt"])
        if not is_gpt5_or_similar:
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
        resp = client.chat.completions.create(**params)
        choice = resp.choices[0]
        message = getattr(choice, "message", None)
        text = ""
        if message:
            raw_content = getattr(message, "content", "")
            if isinstance(raw_content, str):
                text = raw_content.strip()
        if not text:
            finish_reason = getattr(choice, "finish_reason", None)
            dump = ""
            if message and hasattr(message, "model_dump"):
                dump = message.model_dump()
            print(f"[DEBUG] Empty completion from OpenAI (model={model}, finish_reason={finish_reason}, message={dump or message})")
        return text, None
    except Exception as e:
        return None, str(e)


def call_anthropic(client, model, system, user_content, temperature=0, max_tokens=4096):
    """Returns (content, error). If error is None, call succeeded."""
    try:
        resp = client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp.content[0].text.strip(), None
    except Exception as e:
        return None, str(e)


def safe_json_parse(text):
    """Parse JSON, stripping markdown code blocks if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


def get_processed_ids(output_path, id_columns):
    """Get set of already processed IDs from existing output file."""
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_csv(output_path, usecols=id_columns)
        if len(id_columns) == 1:
            return set(df[id_columns[0]].tolist())
        return set(zip(*[df[col] for col in id_columns]))
    except Exception:
        return set()


def stream_write_row(row_dict, output_path, write_header):
    """Write a single row to CSV, appending."""
    pd.DataFrame([row_dict]).to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False
    )
