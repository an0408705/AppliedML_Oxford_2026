#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from scripts.common import (
    append_jsonl,
    extract_message_text,
    get_client_completion,
    make_client,
    normalize_text,
    read_jsonl,
    result_key,
    safe_json_loads,
)

# Defaults now follow the per-run folder structure created by run_pilot.sh
DEFAULT_RUN_DIR = Path(os.getenv("RUN_DIR", "outputs"))
DEFAULT_INPUT = DEFAULT_RUN_DIR / "generations"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "judgments"
DEFAULT_RUN_NAME = os.getenv("RUN_NAME", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Judge generated answers for accuracy and groundedness."
    )
    p.add_argument("--input", type=Path, required=False, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    p.add_argument("--judge-model", type=str, default=os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini"))
    p.add_argument("--resume", action="store_true", help="Skip already judged rows.")
    p.add_argument("--max-retries", type=int, default=int(os.getenv("JUDGE_MAX_RETRIES", "2")))
    p.add_argument("--retry-wait", type=float, default=float(os.getenv("JUDGE_RETRY_WAIT", "2.0")))
    p.add_argument("--max-tokens", type=int, default=int(os.getenv("JUDGE_MAX_TOKENS", "800")))
    p.add_argument("--retry-max-tokens", type=int, default=int(os.getenv("JUDGE_MAX_TOKENS_RETRY", "1200")))
    return p.parse_args()


def system_prompt_for_judging() -> str:
    return (
        "You are a strict evaluation model for a RAG benchmark.\n"
        "You must return ONLY a valid JSON object.\n"
        "Required keys:\n"
        '- "accuracy": integer 0 or 1\n'
        '- "groundedness": integer 0 or 1\n'
        '- "notes": short string\n'
        "Do not include markdown, code fences, or any extra text."
    )


def build_judge_prompt(record: dict) -> str:
    context_condition = str(record.get("context_condition", "gold"))

    if context_condition == "retrieved":
        context = record.get("retrieved_passage_text", "")
    elif context_condition == "poisoned":
        context = record.get("poisoned_passage_text", "")
    elif context_condition == "none":
        context = ""
    else:
        context = record.get("gold_passage_text", "")

    return (
        "Evaluate the model answer.\n\n"
        "Return only JSON with keys accuracy, groundedness, notes.\n\n"
        f"Question:\n{record.get('question', '')}\n\n"
        f"Reference answer:\n{record.get('reference_answer', '')}\n\n"
        f"Provided context:\n{context if context else '[NO CONTEXT PROVIDED]'}\n\n"
        f"Model answer:\n{record.get('answer_text', '')}\n"
    )


def _extract_json_candidate(text: str) -> str:
    text = text.strip()

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1).strip()

    obj = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj:
        return obj.group(0).strip()

    return text


def parse_judge_text(text: str) -> dict[str, Any]:
    text = normalize_text(text)
    if not text:
        raise ValueError("Judge returned empty text")

    try:
        parsed = safe_json_loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    candidate = _extract_json_candidate(text)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Judge JSON did not decode to an object")
    return parsed


def get_response_finish_reason(response: Any) -> str:
    try:
        return str(response.choices[0].finish_reason)
    except Exception:
        return ""


def call_judge(
    *,
    client: Any,
    judge_model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    retry_max_tokens: int,
    max_retries: int,
    retry_wait: float,
) -> tuple[Optional[Any], str, Optional[dict[str, Any]], str, str]:
    """
    Returns:
      response, raw_text, parsed_json, error_message, finish_reason
    """
    last_error = ""
    last_finish_reason = ""

    for attempt in range(max_retries + 1):
        token_budget = max_tokens if attempt == 0 else retry_max_tokens

        try:
            response = get_client_completion(
                client=client,
                model=judge_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=token_budget,
                response_format={"type": "json_object"},
            )
            raw_text = normalize_text(extract_message_text(response))
            last_finish_reason = get_response_finish_reason(response)

            try:
                parsed = parse_judge_text(raw_text)
                return response, raw_text, parsed, "", last_finish_reason
            except Exception as e:
                last_error = f"parse_error: {type(e).__name__}: {e}"
                if attempt < max_retries:
                    time.sleep(retry_wait)
                    continue
                return response, raw_text, None, last_error, last_finish_reason

        except Exception as e:
            last_error = f"api_error: {type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(retry_wait)
                continue
            return None, "", None, last_error, last_finish_reason

    return None, "", None, last_error, last_finish_reason


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve generation file
    if args.input.is_dir():
        files = sorted(args.input.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL files found in {args.input}")
        if args.run_name:
            candidates = [f for f in files if args.run_name in f.name]
            if candidates:
                files = candidates
        input_file = files[-1]
    else:
        input_file = args.input

    records = read_jsonl(input_file)

    # Keep the judge output file aligned with the generation run
    run_name = args.run_name.strip() or input_file.stem
    out_path = args.output_dir / f"{run_name}.jsonl"

    client = make_client()

    seen = set()
    if args.resume and out_path.exists():
        for r in read_jsonl(out_path):
            seen.add(result_key(r))

    pbar = tqdm(total=len(records), desc="Judging", unit="resp")

    for record in records:
        key = (
            str(record.get("question_id", "")),
            str(record.get("model_name", "")),
            str(record.get("context_condition", "")),
            str(record.get("prompt_type", "")),
        )
        if key in seen:
            pbar.update(1)
            continue

        judge_prompt = build_judge_prompt(record)
        response, raw_text, parsed, error, finish_reason = call_judge(
            client=client,
            judge_model=args.judge_model,
            system_prompt=system_prompt_for_judging(),
            user_prompt=judge_prompt,
            max_tokens=args.max_tokens,
            retry_max_tokens=args.retry_max_tokens,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
        )

        judged = {
            **record,
            "judge_model": args.judge_model,
            "judge_raw_text": raw_text,
            "judge_finish_reason": finish_reason,
            "judge_status": "ok" if parsed is not None else "failed",
            "judge_error": error,
            "accuracy": int(parsed.get("accuracy", 0)) if parsed is not None else None,
            "groundedness": int(parsed.get("groundedness", 0)) if parsed is not None else None,
            "judge_notes": normalize_text(parsed.get("notes", "")) if parsed is not None else "",
            "judge_raw": response.model_dump() if hasattr(response, "model_dump") else (str(response) if response is not None else ""),
            "judged_at": datetime.utcnow().isoformat() + "Z",
        }

        append_jsonl(out_path, judged)
        pbar.update(1)

    pbar.close()
    print(f"Saved judgments to: {out_path}")


if __name__ == "__main__":
    main()