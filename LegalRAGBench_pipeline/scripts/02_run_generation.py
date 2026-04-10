#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from scripts.common import (
    EvalConfig,
    append_jsonl,
    build_prompt,
    extract_message_text,
    get_client_completion,
    make_client,
    normalize_text,
    read_csv_rows,
    read_jsonl,
    result_key,
    sample_rows,
    system_prompt_for_answering,
)

# Input dataset stays where it is
DEFAULT_INPUT = Path(os.getenv("INPUT_PATH", "data/clean_input.csv"))

# If RUN_DIR is set by run_pilot.sh, outputs go there.
# Example:
#   outputs/runs/gpt5mini_20260401_134423/generations/
DEFAULT_RUN_DIR = Path(os.getenv("RUN_DIR", "outputs"))
DEFAULT_RUN_NAME = os.getenv("RUN_NAME", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run answer generation for RAG evaluation.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    p.add_argument("--model", type=str, default=os.getenv("GEN_MODEL", "openai/gpt-5-mini"))
    p.add_argument("--sample-size", type=int, default=int(os.getenv("SAMPLE_SIZE", "3")))
    p.add_argument("--sample-seed", type=int, default=int(os.getenv("SAMPLE_SEED", "42")))
    p.add_argument(
        "--context-conditions",
        type=str,
        default=os.getenv("CONTEXT_CONDITIONS", "gold"),
        help="Comma-separated list: gold,retrieved,poisoned,none",
    )
    p.add_argument(
        "--prompt-types",
        type=str,
        default=os.getenv("PROMPT_TYPES", "neutral"),
        help="Comma-separated list: non_rag,neutral,skeptical,faithful",
    )
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.0")))
    p.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "1024")))
    p.add_argument("--resume", action="store_true", help="Skip rows already present in output file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_csv_rows(args.input)
    rows = sample_rows(rows, args.sample_size, args.sample_seed)

    context_conditions = [x.strip() for x in args.context_conditions.split(",") if x.strip()]
    prompt_types = [x.strip() for x in args.prompt_types.split(",") if x.strip()]

    # Use the run folder name as the run_name if one was not supplied.
    # This works well with run_pilot.sh creating a unique RUN_DIR.
    if args.run_name.strip():
        run_name = args.run_name.strip()
    else:
        run_name = args.run_dir.name if args.run_dir.name and args.run_dir.name != "outputs" else datetime.now().strftime("gen_%Y%m%d_%H%M%S")

    output_dir = args.run_dir / "generations"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{run_name}.jsonl"

    client = make_client()
    seen = set()
    if args.resume and out_path.exists():
        for r in read_jsonl(out_path):
            seen.add(result_key(r))

    config = EvalConfig(
        model_name=args.model,
        judge_model_name="",
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        context_conditions=context_conditions,
        prompt_types=prompt_types,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    total = len(rows) * len(context_conditions) * len(prompt_types)
    pbar = tqdm(total=total, desc="Generating", unit="resp")

    for row in rows:
        qid = str(row.get("question_id", ""))
        for context_condition in context_conditions:
            for prompt_type in prompt_types:
                key = (qid, args.model, context_condition, prompt_type)
                if key in seen:
                    pbar.update(1)
                    continue

                user_prompt = build_prompt(row, context_condition, prompt_type)
                response = get_client_completion(
                    client=client,
                    model=args.model,
                    system_prompt=system_prompt_for_answering(),
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                answer_text = normalize_text(extract_message_text(response))

                record = {
                    "run_name": run_name,
                    "run_dir": str(args.run_dir),
                    "question_id": qid,
                    "model_name": args.model,
                    "context_condition": context_condition,
                    "prompt_type": prompt_type,
                    "question": normalize_text(row.get("question", "")),
                    "reference_answer": normalize_text(row.get("reference_answer", "")),
                    "gold_passage_text": normalize_text(row.get("gold_passage_text", "")),
                    "retrieved_passage_text": normalize_text(row.get("retrieved_passage_text", "")),
                    "poisoned_passage_text": normalize_text(row.get("poisoned_passage_text", "")),
                    "prompt_text": user_prompt,
                    "answer_text": answer_text,
                    "raw_response": response.model_dump() if hasattr(response, "model_dump") else str(response),
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }

                append_jsonl(out_path, record)
                pbar.update(1)

    pbar.close()
    print(f"Saved generation outputs to: {out_path}")


if __name__ == "__main__":
    main()