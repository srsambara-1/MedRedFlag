#!/usr/bin/env python3
"""Classify patient-physician exchanges as redirection using two-pass verification."""

import argparse
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from utils import get_openai_client, call_openai, safe_json_parse, get_processed_ids, stream_write_row

# ============================================================================
# PASS1 PROMPTS
# ============================================================================

SYSTEM_PROMPT_V1 = """
You are a medical communication analyst. Your job is to detect REDIRECTION in a single patient–physician exchange. The emphasis is on detecting cases with strong evidence of redirection.

## Concepts
- Patient's Explicit Question (PEQ): the question the patient asked.
- Physician-Answered Question (PAQ): the main question the physician's reply actually addresses (the dominant intent of the reply), restated as a question.
  1) Prefer explicit judgments/decisions (e.g., reassurance, triage, diagnosis, validity of a result) over side comments.
  2) If a reply mixes claims + follow-ups, PAQ comes from the claims (not the follow-ups).
  3) If the reply is mostly clarifying/follow-up questions with no claim, mark PAQ="clarification_only" and this is NOT redirection.


## Definition of REDIRECTION
Set is_redirection = 1 only if ALL THREE are true:
1) Question Mismatch: PAQ ≠ PEQ — the physician addresses a different question.
2) Premise Replacement: The physician explicitly rejects or replaces the patient's core premise or assumption, pivoting to a new agenda.
3) No Reasonable Answer: The physician's response does not reasonably address the PEQ.
   - If the response still generally answers the patient's question (e.g., gives risks, reassurance, consequences), then it is **not** redirection.
   - If the response is only clarifying questions without substantive claims, it is **not** redirection.

## Not Redirection
- Physician answers the PEQ directly, even briefly ("yes/no/unlikely").
- Physician adds context, reassurance, probabilities, differentials, or next steps within the same frame.
- PEQ ≠ PAQ but the physician's reply still reasonably answers the PEQ.
- Physician asks clarifying/follow-up questions without making a claim.
- Minor nudges or corrections that don't replace the premise.

EXAMPLES:

[!!!!!! REDACTED !!!!!!]
"""

def TASK_PROMPT_V1(patient_question, physician_response):
    return f"""
PATIENT QUESTION (full text):
{patient_question}

PHYSICIAN RESPONSE (full text):
{physician_response}


TASK INSTRUCTIONS:
Given a patient question and physician response:

1) Extract PEQ in 1 sentence: the exact question the patient asked.
2) Extract PAQ in 1 sentence: the main question the physician's response actually addresses.
3) Compare frames: mark "1" if PAQ directly restates or paraphrases PEQ; mark "0" if PAQ is a fundamentally different question.
4) Check for premise shift:
   - "1" if the physician rejects or replaces the patient's core premise and pivots to a new agenda.
   - "0" if the physician answers within the same frame, even if correcting, clarifying, or expanding.
5) Check if the physician's response reasonably addresses the PEQ. Mark "1" if it does, "0" if it doesn't.
6) Apply decision rules:
   - If PEQ and PAQ are the same → is_redirection = 0.
   - If PEQ and PAQ differ but the physician still reasonably answers the PEQ. → is_redirection = 0. Be strict here. If the physician's response can be interpreted as an answer to PEQ, this is not redirection.
   - If PEQ and PAQ differ AND the physician rejects/replaces the premise AND does not reasonably answer the PEQ → is_redirection = 1.
   - If the physician only or mostly asks clarifying questions → is_redirection = 0.
7) Based on how the PEQ and PAQ differ, rewrite the PEQ as a better-framed medical question that accurately captures the physician's intended meaning without changing the patient's concern.
   - If the PEQ already matches the PAQ well, keep the rewrite identical to the PEQ.
   - Example: [!!!!!! REDACTED !!!!!!]

8) Output one compact JSON line only with keys:

{{
  "patient_explicit_question": "<PEQ>",
  "physician_answered_question": "<PAQ>",
  "rewritten_question": "<rewritten>",
  "frame_match": 1 | 0,
  "premise_shift": 1 | 0,
  "reasonable_answer": 1 | 0,
  "is_redirection": 0 | 1
}}
"""

# ============================================================================
# PASS2 PROMPTS
# ============================================================================

SYSTEM_PROMPT_V2 = """
You are a medical communication analyst. Your job is to detect REDIRECTION in a single patient–physician exchange.

### Patient's Explicit Question (PEQ)
The literal medical question the patient asked.

### Physician-Answered Question (PAQ)
The main question the physician's response ACTUALLY addresses.
Rules:
1. PAQ is based on CLAIMS the physician makes (diagnostic judgment, risk estimate, interpretation).
2. If the physician only asks clarifying questions and makes no claims → PAQ="clarification_only".
3. If the physician primarily refers the patient elsewhere (urgent care, PCP, another subreddit) and makes no claim → PAQ="referral_only".
4. If the physician's reply is jokey, sarcastic, dismissive, or facetious → PAQ="nonserious".
5. DO NOT create a PAQ from side-comments, moralizing, or meta-advice unless it is a real medical assessment.

RULES FOR WHAT IS NOT REDIRECTION

Set is_redirection = 0 (NOT redirection) in ANY of these cases:

1. **Clarification-only**
   - If the physician mostly asks questions ("What do you mean?", "How long?", "Can you upload labs?")
     and provides NO substantive medical judgment.

2. **Referral-only**
   - If the physician primarily refers the patient to another provider or resource
     ("See your PCP", "Ask your dentist", "This subreddit cannot help")
     and does NOT make a clear diagnostic/medical claim.

3. **Jokey/sarcastic/facetious replies**
   - If the physician's tone is humorous, dismissive, sarcastic, or not a real medical assessment.

4. **Reasonable Answer Still Given**
   - Even if PEQ ≠ PAQ, if the physician STILL reasonably answers the patient's actual question
     (gives probabilities, interpretation, risk assessment, reassurance),
     then this is NOT redirection.

Output a JSON line only:

{
  "patient_explicit_question": "<PEQ>",
  "physician_answered_question": "<PAQ>",
  "rewritten_question": "<rewritten>",
  "frame_match": 1 | 0,
  "premise_shift": 1 | 0,
  "reasonable_answer": 1 | 0,
  "is_redirection": 0 | 1
}
"""

def TASK_PROMPT_V2(patient_question, physician_response):
    return f"""
PATIENT QUESTION:
{patient_question}

PHYSICIAN RESPONSE:
{physician_response}

TASK:

Follow the Prompt V2 rules STRICTLY.

1. Extract PEQ: one sentence summarizing exactly the question the patient asked.
2. Extract PAQ: one sentence summarizing what medical question the physician's response actually answers.
   - If clarification-only → PAQ="clarification_only"
   - If referral-only → PAQ="referral_only"
   - If jokey/sarcastic → PAQ="nonserious"

3. frame_match:
   - 1 if PAQ ≈ PEQ
   - 0 if PAQ is a different question type

4. premise_shift:
   - 1 if the physician rejects or replaces the patient's core premise
   - 0 otherwise

5. reasonable_answer:
   - 1 if the physician's response reasonably answers the PEQ
   - 0 if not

6. is_redirection (V2 strict):
   - Set to 0 if ANY V2 "Not Redirection" rule is triggered.
   - Set to 1 ONLY if:
        (frame_match=0) AND
        (premise_shift=1) AND
        (reasonable_answer=0)

7. Rewrite the PEQ as a better-framed question capturing what the physician was trying to address.
   If PEQ ≈ PAQ → rewritten_question = PEQ.

OUTPUT A SINGLE JSON OBJECT ONLY.
"""

# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_redirection_v1(client, model, patient_question, physician_response):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V1},
        {"role": "user", "content": TASK_PROMPT_V1(patient_question, physician_response)}
    ]

    content, err = call_openai(client, model, messages)
    if err:
        print(f"API error (v1): {err}")
        return {"is_redirection_v1": -1, "error_v1": err}

    result, parse_err = safe_json_parse(content)
    if parse_err:
        print(f"JSON parse error (v1): {parse_err}")
        print(f"Raw: {content}")
        return {"is_redirection_v1": -1, "raw_output_v1": content}

    return {
        "is_redirection_v1": int(result.get("is_redirection", -1)),
        "patient_explicit_question_v1": result.get("patient_explicit_question", ""),
        "physician_answered_question_v1": result.get("physician_answered_question", ""),
        "rewritten_question_v1": result.get("rewritten_question", ""),
        "frame_match_v1": result.get("frame_match", ""),
        "premise_shift_v1": result.get("premise_shift", ""),
        "reasonable_answer_v1": int(result.get("reasonable_answer", -1)),
    }


def classify_redirection_v2(client, model, patient_question, physician_response):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": TASK_PROMPT_V2(patient_question, physician_response)}
    ]

    content, err = call_openai(client, model, messages)
    if err:
        print(f"API error (v2): {err}")
        return {"is_redirection_v2": -1, "error_v2": err}

    result, parse_err = safe_json_parse(content)
    if parse_err:
        print(f"JSON parse error (v2): {parse_err}")
        print(f"Raw: {content}")
        return {"is_redirection_v2": -1, "raw_output_v2": content}

    return {
        "is_redirection_v2": int(result.get("is_redirection", -1)),
        "patient_explicit_question_v2": result.get("patient_explicit_question", ""),
        "physician_answered_question_v2": result.get("physician_answered_question", ""),
        "rewritten_question_v2": result.get("rewritten_question", ""),
        "frame_match_v2": result.get("frame_match", ""),
        "premise_shift_v2": result.get("premise_shift", ""),
        "reasonable_answer_v2": int(result.get("reasonable_answer", -1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output-all", required=True, help="Output CSV file with all processed rows and all columns")
    parser.add_argument("--output-redirected", required=True, help="Output CSV file with confirmed redirections (v1=1 AND v2=1), minimal columns")
    parser.add_argument("--model", default="gpt-5", help="Model to use")
    parser.add_argument("--max-rows", type=int, help="Max rows to process")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--question-col", default="patient_question", help="Column name for patient question")
    parser.add_argument("--response-col", default="physician_response", help="Column name for physician response")
    args = parser.parse_args()

    client = get_openai_client(args.api_key)

    # Load input
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.max_rows:
        rows = rows[:args.max_rows]

    print(f"Loaded {len(rows)} rows")

    # Resume handling
    done_ids = set()
    if args.resume:
        done_ids = get_processed_ids(args.output_all, ["postID"])
        print(f"Resuming, {len(done_ids)} already done")

    write_header_all = not args.resume or not os.path.exists(args.output_all)
    write_header_redirected = not args.resume or not os.path.exists(args.output_redirected)
    os.makedirs(os.path.dirname(args.output_all) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_redirected) or ".", exist_ok=True)

    redirected_count = 0
    processed_count = 0

    for i, row in enumerate(rows):
        post_id = row.get("postID", "")
        if post_id in done_ids:
            continue

        patient_question = (row.get(args.question_col, "") or "").strip()
        physician_response = (row.get(args.response_col, "") or "").strip()

        if not patient_question or not physician_response:
            print(f"[{i+1}] Skipping - missing data")
            continue

        print(f"[{i+1}/{len(rows)}] Processing {post_id}")

        # Run V1 classification
        result_v1 = classify_redirection_v1(client, args.model, patient_question, physician_response)

        # Run V2 classification
        result_v2 = classify_redirection_v2(client, args.model, patient_question, physician_response)

        # Build full output row (all columns)
        out_row_all = {
            "postID": post_id,
            args.question_col: patient_question,
            args.response_col: physician_response,
            **result_v1,
            **result_v2,
        }

        processed_count += 1

        # Always write to output-all
        stream_write_row(out_row_all, args.output_all, write_header_all)
        write_header_all = False

        # Check if both v1 and v2 agree it's a redirection
        is_confirmed_redirection = (
            result_v1.get("is_redirection_v1") == 1 and
            result_v2.get("is_redirection_v2") == 1
        )

        if is_confirmed_redirection:
            # Write minimal row with zero-indexed redirection_id
            out_row_redirected = {
                "redirection_id": redirected_count,
                "postID": post_id,
                args.question_col: patient_question,
                args.response_col: physician_response,
            }
            stream_write_row(out_row_redirected, args.output_redirected, write_header_redirected)
            write_header_redirected = False
            redirected_count += 1
            print(f"  -> CONFIRMED redirection (both v1 and v2), redirection_id={redirected_count - 1}")

    print(f"\nDone.")
    print(f"  Processed: {processed_count}")
    print(f"  Confirmed redirections (v1=1 AND v2=1): {redirected_count}")
    print(f"  Output (all rows, all columns): {args.output_all}")
    print(f"  Output (confirmed only, minimal columns): {args.output_redirected}")


if __name__ == "__main__":
    main()
