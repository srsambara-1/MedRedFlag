#!/usr/bin/env python3
"""RAG: Model is provided with retrieved texts, then responds."""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import anthropic
import pandas as pd
from openai import APIError, APITimeoutError, BadRequestError, OpenAI, RateLimitError


SYSTEM_PROMPT = """
You are a helpful medical assistant. You will be provided with retrieved medical information and a patient's question.

Answer the patient's question based on the retrieved information. Important guidelines:
- Use the retrieved information to inform your answer
- You do NOT need to mention or cite all retrieved documents
- You do NOT need to use information from all documents
- Only use information that is relevant to answering the patient's specific question
- If the retrieved information doesn't fully address the question, acknowledge this
- Write in a clear, patient-friendly manner
"""


def get_client(api_type: str, host: str = None, port: str = None, api_key: str = None):
    """
    Get API client configured for vLLM, OpenAI, or Anthropic API.

    Args:
        api_type: "vllm", "openai", or "anthropic"
        host: vLLM server host (for vllm type)
        port: vLLM server port (for vllm type)
        api_key: API key (EMPTY for vLLM, actual key for OpenAI/Anthropic)

    Returns:
        OpenAI client (for vllm/openai) or Anthropic client (for anthropic)
    """
    if api_type == "vllm":
        base_url = f"http://{host}:{port}/v1"
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    elif api_type == "openai":
        return OpenAI(api_key=api_key)
    elif api_type == "anthropic":
        return anthropic.Anthropic(api_key=api_key)
    else:
        raise ValueError(f"Unsupported api_type: {api_type}. Use 'vllm', 'openai', or 'anthropic'.")


def load_retrieval_results(retrieval_file: str) -> Dict[str, List[Dict]]:
    """
    Load retrieval results from JSON file.

    Supports two formats:
    1. List format (from run_retrieval.py):
       [{"redirection_id": "123", "documents": [...]}, ...]
    2. Dict format:
       {"123": [{"id": "...", "title": "...", "content": "..."}, ...], ...}

    Returns:
        Dict mapping redirection_id (str) to list of retrieved documents
    """
    with open(retrieval_file, 'r') as f:
        data = json.load(f)

    # Handle list format (from run_retrieval.py)
    if isinstance(data, list):
        result = {}
        for item in data:
            # Support both new format (redirection_id) and legacy format (query_idx)
            if 'redirection_id' in item:
                redirection_id = str(item['redirection_id'])
            elif 'query_idx' in item:
                # Legacy support: treat query_idx as redirection_id
                redirection_id = str(item['query_idx'])
                print(f"WARNING: Using legacy 'query_idx' format. Consider regenerating retrieval results with 'redirection_id'.")
            else:
                raise ValueError("Missing 'redirection_id' or 'query_idx' in retrieval results")

            documents = item['documents']
            result[redirection_id] = documents
        return result

    # Handle dict format
    elif isinstance(data, dict):
        return {str(k): v for k, v in data.items()}

    else:
        raise ValueError(f"Unsupported retrieval results format: {type(data)}")


def format_retrieved_docs(retrieved_docs: List[Dict], top_k: int = 5) -> str:
    """Format retrieved documents for inclusion in prompt."""
    if not retrieved_docs:
        return "[No relevant medical information was retrieved.]"

    # Take only top-k documents
    docs_to_use = retrieved_docs[:top_k]

    formatted = []
    for i, doc in enumerate(docs_to_use, 1):
        title = doc.get('title', 'Untitled')
        content = doc.get('content', '')

        formatted.append(f"[Document {i}] {title}\n{content}")

    return "\n\n".join(formatted)


def call_anthropic_with_rag(
    client,
    model_name: str,
    user_content: str,
    *,
    max_tokens: int,
    temperature: float
) -> Tuple[Optional[str], Optional[str]]:
    """Anthropic wrapper mirroring utils.call_anthropic but tailored for RAG."""
    try:
        anthropic_max_tokens = max_tokens if max_tokens and max_tokens > 0 else 8192
        response = client.messages.create(
            model=model_name,
            max_tokens=anthropic_max_tokens,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}]
        )
        return response.content[0].text.strip(), None
    except anthropic.BadRequestError as e:
        error_str = str(e)
        if "maximum context length" in error_str.lower() or "context_length_exceeded" in error_str.lower():
            return "[ERROR: Context length exceeded - prompt too long for model]", "ContextLengthError"
        return f"[ERROR: Bad request - {error_str[:200]}]", "BadRequestError"
    except anthropic.RateLimitError:
        return "[ERROR: Rate limit exceeded]", "RateLimitError"
    except anthropic.APITimeoutError:
        return "[ERROR: API timeout]", "TimeoutError"
    except anthropic.APIError as e:
        return f"[ERROR: API error - {str(e)[:200]}]", "APIError"
    except Exception as e:
        return f"[ERROR: {type(e).__name__} - {str(e)[:200]}]", type(e).__name__


def call_openai_with_rag(
    client,
    model_name: str,
    user_content: str,
    *,
    temperature: float,
    max_tokens: int
) -> Tuple[Optional[str], Optional[str]]:
    """OpenAI/vLLM wrapper mirroring utils.call_openai but with RAG defaults."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    api_params: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }

    is_gpt5_or_similar = any(x in model_name.lower() for x in ["gpt-5", "gpt5", "o1", "o3"])

    if not is_gpt5_or_similar:
        if max_tokens and max_tokens > 0:
            api_params["max_tokens"] = max_tokens
        if temperature is not None:
            api_params["temperature"] = temperature

    try:
        resp = client.chat.completions.create(**api_params)
        return resp.choices[0].message.content.strip(), None
    except BadRequestError as e:
        error_str = str(e)
        if "maximum context length" in error_str.lower() or "context_length_exceeded" in error_str.lower():
            return "[ERROR: Context length exceeded - prompt too long for model]", "ContextLengthError"
        return f"[ERROR: Bad request - {error_str[:200]}]", "BadRequestError"
    except RateLimitError:
        return "[ERROR: Rate limit exceeded]", "RateLimitError"
    except APITimeoutError:
        return "[ERROR: API timeout]", "TimeoutError"
    except APIError as e:
        return f"[ERROR: API error - {str(e)[:200]}]", "APIError"
    except Exception as e:
        return f"[ERROR: {type(e).__name__} - {str(e)[:200]}]", type(e).__name__


def generate_rag_answer(
    client,
    model_name: str,
    patient_question: str,
    retrieved_docs: List[Dict],
    api_type: str,
    *,
    top_k: int = 5,
    max_tokens: int = 2048,
    temperature: float = 0.0
) -> Tuple[str, Optional[str]]:
    """
    Prompt LLM to answer the patient's question using retrieved medical documents.

    Returns (response_text, error_type). Error is None when successful.
    """
    retrieved_info = format_retrieved_docs(retrieved_docs, top_k=top_k)
    user_content = f"""RETRIEVED MEDICAL INFORMATION:
{retrieved_info}

---

PATIENT QUESTION:
{patient_question}

---

Please answer the patient's question. Use the retrieved medical information above to inform your response, but only include information that is directly relevant to answering this specific question."""

    if api_type == "anthropic":
        return call_anthropic_with_rag(
            client,
            model_name,
            user_content,
            max_tokens=max_tokens,
            temperature=temperature
        )

    return call_openai_with_rag(
        client,
        model_name,
        user_content,
        temperature=temperature,
        max_tokens=max_tokens
    )


def process_row(
    client,
    row: pd.Series,
    redirection_id: str,
    retrieval_results: Dict[str, List[Dict]],
    model_name: str,
    api_type: str,
    *,
    top_k: int = 5,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> Tuple[str, int, Optional[str]]:
    """Generate the answer, number of docs used, and error flag for a row."""
    patient_question = str(row.get("patient_question", ""))
    retrieved_docs = retrieval_results.get(redirection_id, [])

    llm_answer, error_type = generate_rag_answer(
        client=client,
        model_name=model_name,
        patient_question=patient_question,
        retrieved_docs=retrieved_docs,
        api_type=api_type,
        top_k=top_k,
        max_tokens=max_tokens,
        temperature=temperature
    )

    num_docs_used = len(retrieved_docs[:top_k])
    return llm_answer, num_docs_used, error_type


def get_temp_output_path(output_path: str) -> str:
    """Generate temporary output file path."""
    base, ext = os.path.splitext(output_path)
    return f"{base}_temp{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM responses with RAG (Retrieval-Augmented Generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--retrieval-results", type=str, required=True, help="Path to retrieval results JSON file")
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    parser.add_argument("--api-type", type=str, default="openai", choices=["vllm", "openai", "anthropic"], help="API type: vllm, openai, or anthropic")
    parser.add_argument("--host", type=str, default="localhost", help="vLLM server host")
    parser.add_argument("--port", type=str, default="8000", help="vLLM server port")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API key")
    parser.add_argument("--output-column", type=str, default=None, help="Optional custom output column name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top retrieved documents to include")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens in response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum number of rows to process")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")

    args = parser.parse_args()
    model_base = args.model.split("/")[-1]
    model_slug = re.sub(r"[^0-9a-zA-Z]+", "_", model_base).strip("_").lower() or "model"
    output_column = args.output_column or f"{model_slug}_answer_rag"


    print(f"Loading input CSV: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}")

    # Load retrieval results
    print(f"Loading retrieval results: {args.retrieval_results}")
    retrieval_results = load_retrieval_results(args.retrieval_results)
    print(f"  Retrieval results loaded for {len(retrieval_results)} questions")

    # Ensure redirection_id column exists
    if "redirection_id" not in df.columns:
        raise ValueError("ERROR: Input CSV must contain 'redirection_id' column")

    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"  Limited to first {args.max_rows} rows")

    # Reset index but keep redirection_id for matching
    df = df.reset_index(drop=True)


    print(f"\nSetting up {args.api_type.upper()} client...")
    client = get_client(
        api_type=args.api_type,
        host=args.host,
        port=args.port,
        api_key=args.api_key
    )


    temp_output = get_temp_output_path(args.output)

    if args.resume and os.path.exists(temp_output):
        print(f"Resuming from: {temp_output}")
        existing_df = pd.read_csv(temp_output)
        already_processed = set(existing_df.index.tolist())
        print(f"Already processed: {len(already_processed)} rows")
    else:
        already_processed = set()
        if os.path.exists(temp_output):
            os.remove(temp_output)


    print(f"\nGenerating RAG-enhanced responses...")
    print(f"  Model: {args.model}")
    print(f"  Output column: {output_column}")
    print(f"  Top-K documents: {args.top_k}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print("")

    error_count = 0
    success_count = 0

    for idx, row in df.iterrows():
        if idx in already_processed:
            continue

        redirection_id = str(row['redirection_id'])

        print(f"Processing row {idx + 1}/{len(df)} (redirection_id={redirection_id})...", end=" ")

        try:
            answer, num_docs_used, error_type = process_row(
                client=client,
                row=row,
                redirection_id=redirection_id,
                retrieval_results=retrieval_results,
                model_name=args.model,
                api_type=args.api_type,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )

            df.at[idx, output_column] = answer
            df.at[idx, f"{output_column}_num_docs_used"] = num_docs_used

            if error_type:
                df.at[idx, f"{output_column}_error"] = error_type
                print(error_type)
                error_count += 1
            else:
                success_count += 1

            # Save progress after each row
            df.to_csv(temp_output, index=False)

        except Exception as e:
            # Catch any unexpected errors that weren't handled in generate_rag_answer
            print(f"UNEXPECTED ERROR: {e}")
            df.at[idx, output_column] = f"[SYSTEM ERROR: {str(e)[:100]}]"
            df.at[idx, f"{output_column}_error"] = "SystemError"
            df.to_csv(temp_output, index=False)
            error_count += 1
            # Continue processing instead of raising


    print(f"\nSaving final output: {args.output}")
    df.to_csv(args.output, index=False)

    if os.path.exists(temp_output):
        os.remove(temp_output)

    print("\n" + "="*80)
    print("Processing Complete!")
    print("="*80)
    print(f"Total rows processed: {len(df)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    if error_count > 0:
        print(f"Check '{output_column}_error' column for error types")
    print(f"Output saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
