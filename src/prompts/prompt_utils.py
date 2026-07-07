"""Load prompt rows from CSV files and normalize prompt-answer handling.

This module resolves prompt CSVs from the repository and provides helpers for
loading prompt records, formatting them for model input, and extracting expected
answers or tokens for evaluation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_csv_path(filename: str) -> Path:
    candidates = [
        _repo_root() / filename,
        _repo_root() / "data" / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_prompts_from_csv(path: str | Path) -> List[Dict[str, Any]]:
    """Load prompt rows from a CSV file containing a sentence and answer column."""
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        record = dict(row)
        if "sentence" not in record and "Sentence" not in record:
            raise KeyError(f"CSV {path} is missing a 'sentence' column")
        if "Answer" not in record and "answer" not in record and "Capital" not in record and "Expected" not in record:
            raise KeyError(f"CSV {path} is missing an answer column")
        rows.append(record)
    return rows


def load_prompts(behaviour: str) -> Dict[str, Any]:
    """Load prompt rows for a supported behaviour."""
    if behaviour == "capitals":
        rows = load_prompts_from_csv(_resolve_csv_path("capitals_data.csv"))
    elif behaviour == "addition":
        rows = load_prompts_from_csv(_resolve_csv_path("addition_data.csv"))
    elif behaviour == "units":
        rows = load_prompts_from_csv(_resolve_csv_path("units_data.csv"))
    else:
        raise ValueError(f"Unsupported behaviour: {behaviour}")

    prompts = []
    for idx, row in enumerate(rows):
        prompt = {
            "id": f"{behaviour}-{idx}",
            "sentence": row.get("sentence") or row.get("Sentence") or "",
            "Answer": row.get("Answer") or row.get("answer") or row.get("Capital") or row.get("Expected") or "",
        }
        if not prompt["sentence"]:
            raise ValueError(f"Prompt {idx} is missing a sentence")
        prompts.append(prompt)
    return {"behaviour": behaviour, "prompts": prompts}


def format_prompt(prompt: Dict[str, Any]) -> str:
    """Return the prompt text used for model inference."""
    sentence = str(prompt.get("sentence", "")).strip()
    if not sentence:
        raise ValueError("Prompt is missing a sentence")
    return sentence


def _parse_answer_values(value: Any) -> List[str]:
    """Convert answer fields to a list of acceptable answer strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        return [stripped]
    return [str(value).strip()]


def get_expected_answers(prompt: Dict[str, Any]) -> List[str]:
    """Extract all acceptable answer strings from a prompt record."""
    for key in ("Answer", "answer", "Capital", "Expected"):
        if key in prompt and prompt[key] is not None:
            values = _parse_answer_values(prompt[key])
            if values:
                return values
    raise KeyError("Prompt is missing an expected token field")


def get_expected_token(prompt: Dict[str, Any]) -> str:
    """Extract the expected answer token from a prompt record."""
    answers = get_expected_answers(prompt)
    return answers[0]
