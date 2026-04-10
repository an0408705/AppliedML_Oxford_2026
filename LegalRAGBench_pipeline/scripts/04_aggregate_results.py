#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd

from scripts.common import read_jsonl


# Follow the per-run folder structure created by run_pilot.sh
DEFAULT_RUN_DIR = Path(os.getenv("RUN_DIR", "outputs"))
DEFAULT_INPUT = DEFAULT_RUN_DIR / "judgments"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "summaries"

CONTEXT_ORDER = ["gold", "retrieved", "poisoned", "none"]
PROMPT_ORDER = ["non_rag", "neutral", "skeptical", "faithful"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate judged results into summary tables."
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", type=str, default="")
    return p.parse_args()


def collect_input_files(input_path: Path, run_name: str) -> List[Path]:
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL files found in {input_path}")
        if run_name:
            filtered = [f for f in files if run_name in f.name]
            if filtered:
                files = filtered
        return files

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    return [input_path]


def load_all_rows(files: List[Path]) -> pd.DataFrame:
    all_rows = []
    for fp in files:
        rows = read_jsonl(fp)
        for r in rows:
            r = dict(r)
            r["source_file"] = fp.name
            all_rows.append(r)

    if not all_rows:
        raise ValueError("No rows found in the provided judgment files.")

    return pd.DataFrame(all_rows)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["accuracy", "groundedness"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    df["judge_ok"] = df.get("judge_status", "").astype(str).eq("ok")

    df["both_correct_and_grounded"] = (
        (df["accuracy"] == 1) & (df["groundedness"] == 1)
    ).astype("float")

    df["hallucination"] = (1 - df["groundedness"]).where(df["groundedness"].notna())

    df["ungrounded_correct"] = (
        (df["accuracy"] == 1) & (df["groundedness"] == 0)
    ).astype("float")

    df["grounded_incorrect"] = (
        (df["accuracy"] == 0) & (df["groundedness"] == 1)
    ).astype("float")

    df["ungrounded_incorrect"] = (
        (df["accuracy"] == 0) & (df["groundedness"] == 0)
    ).astype("float")

    # Positive values mean groundedness exceeds accuracy;
    # negative values mean correctness exceeds groundedness.
    df["faithfulness_gap"] = df["groundedness"] - df["accuracy"]

    return df


def summarize_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_total=("question_id", "size"),
            n_accuracy=("accuracy", "count"),
            n_groundedness=("groundedness", "count"),
            n_judge_ok=("judge_ok", "sum"),
            accuracy_mean=("accuracy", "mean"),
            groundedness_mean=("groundedness", "mean"),
            both_correct_and_grounded_mean=("both_correct_and_grounded", "mean"),
            hallucination_rate=("hallucination", "mean"),
            ungrounded_correct_rate=("ungrounded_correct", "mean"),
            grounded_incorrect_rate=("grounded_incorrect", "mean"),
            ungrounded_incorrect_rate=("ungrounded_incorrect", "mean"),
            faithfulness_gap_mean=("faithfulness_gap", "mean"),
            judge_failure_rate=("judge_ok", lambda s: 1.0 - float(s.mean())),
        )
        .reset_index()
    )


def build_context_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each model/prompt pair, compare the four context conditions.
    Useful diagnostics:
      - context reliance
      - retrieval-error sensitivity
      - poisoning susceptibility
    """
    base = (
        df.groupby(["model_name", "prompt_type", "context_condition"], dropna=False)
        .agg(
            n=("question_id", "size"),
            accuracy_mean=("accuracy", "mean"),
            groundedness_mean=("groundedness", "mean"),
            both_correct_and_grounded_mean=("both_correct_and_grounded", "mean"),
            hallucination_rate=("hallucination", "mean"),
            faithfulness_gap_mean=("faithfulness_gap", "mean"),
        )
        .reset_index()
    )

    acc_pivot = base.pivot_table(
        index=["model_name", "prompt_type"],
        columns="context_condition",
        values="accuracy_mean",
        aggfunc="first",
    ).reindex(columns=CONTEXT_ORDER)

    grd_pivot = base.pivot_table(
        index=["model_name", "prompt_type"],
        columns="context_condition",
        values="groundedness_mean",
        aggfunc="first",
    ).reindex(columns=CONTEXT_ORDER)

    both_pivot = base.pivot_table(
        index=["model_name", "prompt_type"],
        columns="context_condition",
        values="both_correct_and_grounded_mean",
        aggfunc="first",
    ).reindex(columns=CONTEXT_ORDER)

    hall_pivot = base.pivot_table(
        index=["model_name", "prompt_type"],
        columns="context_condition",
        values="hallucination_rate",
        aggfunc="first",
    ).reindex(columns=CONTEXT_ORDER)

    gap_pivot = base.pivot_table(
        index=["model_name", "prompt_type"],
        columns="context_condition",
        values="faithfulness_gap_mean",
        aggfunc="first",
    ).reindex(columns=CONTEXT_ORDER)

    comp = pd.DataFrame(index=acc_pivot.index)
    for c in CONTEXT_ORDER:
        comp[f"accuracy_{c}"] = acc_pivot[c]
        comp[f"groundedness_{c}"] = grd_pivot[c]
        comp[f"both_correct_and_grounded_{c}"] = both_pivot[c]
        comp[f"hallucination_rate_{c}"] = hall_pivot[c]
        comp[f"faithfulness_gap_{c}"] = gap_pivot[c]

    comp["context_reliance_accuracy_gold_minus_none"] = (
        comp["accuracy_gold"] - comp["accuracy_none"]
    )
    comp["context_reliance_groundedness_gold_minus_none"] = (
        comp["groundedness_gold"] - comp["groundedness_none"]
    )

    comp["retrieval_error_susceptibility_accuracy_gold_minus_retrieved"] = (
        comp["accuracy_gold"] - comp["accuracy_retrieved"]
    )
    comp["retrieval_error_susceptibility_groundedness_gold_minus_retrieved"] = (
        comp["groundedness_gold"] - comp["groundedness_retrieved"]
    )

    comp["poisoning_susceptibility_accuracy_gold_minus_poisoned"] = (
        comp["accuracy_gold"] - comp["accuracy_poisoned"]
    )
    comp["poisoning_susceptibility_groundedness_gold_minus_poisoned"] = (
        comp["groundedness_gold"] - comp["groundedness_poisoned"]
    )

    comp["none_to_gold_accuracy_gain"] = comp["accuracy_gold"] - comp["accuracy_none"]
    comp["none_to_gold_groundedness_gain"] = (
        comp["groundedness_gold"] - comp["groundedness_none"]
    )

    comp = comp.reset_index().sort_values(["model_name", "prompt_type"])
    return comp


def build_prompt_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measures how much each model changes across prompts within each context.
    """
    base = (
        df.groupby(["model_name", "context_condition", "prompt_type"], dropna=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            groundedness_mean=("groundedness", "mean"),
            both_correct_and_grounded_mean=("both_correct_and_grounded", "mean"),
            hallucination_rate=("hallucination", "mean"),
            faithfulness_gap_mean=("faithfulness_gap", "mean"),
        )
        .reset_index()
    )

    out = (
        base.groupby(["model_name", "context_condition"], dropna=False)
        .agg(
            prompt_n=("prompt_type", "nunique"),
            accuracy_min=("accuracy_mean", "min"),
            accuracy_max=("accuracy_mean", "max"),
            accuracy_range=("accuracy_mean", lambda s: float(s.max() - s.min())),
            accuracy_std=("accuracy_mean", "std"),
            groundedness_min=("groundedness_mean", "min"),
            groundedness_max=("groundedness_mean", "max"),
            groundedness_range=("groundedness_mean", lambda s: float(s.max() - s.min())),
            groundedness_std=("groundedness_mean", "std"),
            hallucination_min=("hallucination_rate", "min"),
            hallucination_max=("hallucination_rate", "max"),
            hallucination_range=("hallucination_rate", lambda s: float(s.max() - s.min())),
            faithfulness_gap_min=("faithfulness_gap_mean", "min"),
            faithfulness_gap_max=("faithfulness_gap_mean", "max"),
            faithfulness_gap_range=("faithfulness_gap_mean", lambda s: float(s.max() - s.min())),
        )
        .reset_index()
        .sort_values(["model_name", "context_condition"])
    )

    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_input_files(args.input, args.run_name)
    df = load_all_rows(files)
    df = add_derived_columns(df)

    # If we are using the run-folder setup, the stem is the run name;
    # otherwise fall back to a combined label.
    if args.input.is_dir():
        stem = args.run_name.strip() or "combined"
    else:
        stem = args.input.stem

    enriched_path = args.output_dir / f"{stem}_enriched_rows.csv"
    df.to_csv(enriched_path, index=False)

    by_condition_prompt = summarize_group(
        df, ["model_name", "context_condition", "prompt_type"]
    ).sort_values(["model_name", "context_condition", "prompt_type"])

    by_model = summarize_group(df, ["model_name"]).sort_values(["model_name"])

    by_context = summarize_group(df, ["context_condition"]).sort_values(
        ["context_condition"]
    )
    by_context["context_condition"] = pd.Categorical(
        by_context["context_condition"], categories=CONTEXT_ORDER, ordered=True
    )
    by_context = by_context.sort_values(["context_condition"])

    by_prompt = summarize_group(df, ["prompt_type"]).sort_values(["prompt_type"])
    by_prompt["prompt_type"] = pd.Categorical(
        by_prompt["prompt_type"], categories=PROMPT_ORDER, ordered=True
    )
    by_prompt = by_prompt.sort_values(["prompt_type"])

    context_comparison = build_context_comparison(df)
    prompt_sensitivity = build_prompt_sensitivity(df)

    by_model_context = summarize_group(df, ["model_name", "context_condition"]).sort_values(
        ["model_name", "context_condition"]
    )

    # Write outputs
    by_condition_prompt.to_csv(
        args.output_dir / f"{stem}_by_model_context_prompt.csv", index=False
    )
    by_model.to_csv(args.output_dir / f"{stem}_by_model.csv", index=False)
    by_context.to_csv(args.output_dir / f"{stem}_by_context.csv", index=False)
    by_prompt.to_csv(args.output_dir / f"{stem}_by_prompt.csv", index=False)
    by_model_context.to_csv(
        args.output_dir / f"{stem}_by_model_context.csv", index=False
    )
    context_comparison.to_csv(
        args.output_dir / f"{stem}_context_comparison.csv", index=False
    )
    prompt_sensitivity.to_csv(
        args.output_dir / f"{stem}_prompt_sensitivity.csv", index=False
    )

    print("\n=== By model / context / prompt ===")
    print(by_condition_prompt.to_string(index=False))

    print("\n=== By model ===")
    print(by_model.to_string(index=False))

    print("\n=== By context ===")
    print(by_context.to_string(index=False))

    print("\n=== By prompt ===")
    print(by_prompt.to_string(index=False))

    print("\n=== Context comparisons (gold vs none / retrieved / poisoned) ===")
    print(context_comparison.to_string(index=False))

    print("\n=== Prompt sensitivity by model and context ===")
    print(prompt_sensitivity.to_string(index=False))

    print(f"\nSaved summaries to: {args.output_dir}")
    print(f"Enriched row-level file: {enriched_path}")


if __name__ == "__main__":
    main()