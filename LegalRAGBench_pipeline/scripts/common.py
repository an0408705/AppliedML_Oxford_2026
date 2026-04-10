from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from openai import OpenAI


OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "")


@dataclass(frozen=True)
class EvalConfig:
    model_name: str
    judge_model_name: str
    sample_size: int
    sample_seed: int
    context_conditions: List[str]
    prompt_types: List[str]
    temperature: float = 0.0
    max_tokens: int = 1024


def ensure_api_key() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it in your shell before running the scripts."
        )


def make_client() -> OpenAI:
    ensure_api_key()
    headers = {}
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_SITE_NAME:
        headers["X-OpenRouter-Title"] = OPENROUTER_SITE_NAME
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers=headers or None,
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df.to_dict(orient="records")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return normalize_text(value)


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def sample_rows(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(rows):
        return rows
    return pd.DataFrame(rows).sample(n=n, random_state=seed).to_dict(orient="records")


def build_context_text(row: Dict[str, Any], context_condition: str) -> str:
    context_condition = context_condition.lower()
    if context_condition == "gold":
        return as_str(row.get("gold_passage_text", ""))
    if context_condition == "retrieved":
        return as_str(row.get("retrieved_passage_text", ""))
    if context_condition == "poisoned":
        return as_str(row.get("poisoned_passage_text", ""))
    if context_condition == "none":
        return ""
    raise ValueError(f"Unknown context condition: {context_condition}")


def build_prompt(row: Dict[str, Any], context_condition: str, prompt_type: str) -> str:
    question = as_str(row.get("question", ""))
    context = build_context_text(row, context_condition)
    prompt_type = prompt_type.lower()

    if prompt_type == "non_rag":
        return (
            "Answer the question directly and clearly.\n\n"
            f"Question:\n{question}\n\n"
            "Answer:" 
        )

    instructions = {
        "neutral": "Answer the question using the provided context if it is helpful.",
        "skeptical": (
            "Answer the question, but treat the provided context cautiously. "
            "Do not blindly trust it if it appears misleading."
        ),
        "faithful": (
            "Answer the question using only the provided context. "
            "Do not rely on outside knowledge unless the context is insufficient."
        ),
    }
    if prompt_type not in instructions:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    if not context:
        return (
            f"{instructions[prompt_type]}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    return (
        f"{instructions[prompt_type]}\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def system_prompt_for_answering() -> str:
    return (
        "You are a careful assistant answering questions for an evaluation. "
        "Write a direct answer. Do not mention policies, prompts, or hidden reasoning."
    )


def system_prompt_for_judging() -> str:
    return (
        "You are a strict evaluator. You must judge the model's answer against the reference answer "
        "and the provided context. Return only valid JSON with keys: accuracy, groundedness, notes. "
        "accuracy and groundedness must each be 0 or 1."
    )


def get_client_completion(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    response_format: Optional[dict] = None,
) -> Any:
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format
    return client.chat.completions.create(**kwargs)


def extract_message_text(response: Any) -> str:
    try:
        if response is None:
            return ""

        # Handle OpenAI-style objects
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")

        if not choices:
            return ""

        choice0 = choices[0]

        message = getattr(choice0, "message", None)
        if message is None and isinstance(choice0, dict):
            message = choice0.get("message")

        if message is None:
            return ""

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")

        if content is None:
            return ""

        return normalize_text(content)
    except Exception:
        return ""


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Cannot parse empty text as JSON")
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def result_key(row: Dict[str, Any]) -> tuple:
    return (
        str(row.get("question_id", "")),
        str(row.get("model_name", "")),
        str(row.get("context_condition", "")),
        str(row.get("prompt_type", "")),
    )
