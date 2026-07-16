"""Generate 10,000 position-matched arithmetic prompts for carry analysis.

Every sum is three digits and every prompt ends after the teacher-forced
hundreds digit.  The captured final-token MLP activation therefore occupies the
same tens-prediction position used by the carry interventions.  The corpus is
balanced over carry status and as evenly as possible over feasible tens digits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_SEED = 787
EXPLICIT_HELDOUT = {
    (58, 83),
    (44, 83),
    (54, 83),
    (32, 42),
}

CORE_TEMPLATES = [
    "Question: What is {a} + {b}? Answer: {hundreds}",
    "Question: Calculate {a} + {b}. Answer: {hundreds}",
    "Problem: Add {a} and {b}. Result: {hundreds}",
    "Arithmetic: {a} plus {b} equals {hundreds}",
    "Question: Find the sum of {a} and {b}. Answer: {hundreds}",
    "Compute {a} + {b}. The answer is {hundreds}",
    "Question: Add {a} to {b}. Answer: {hundreds}",
    "Solve the addition problem {a} + {b}. Result: {hundreds}",
    "Question: What total is obtained from {a} plus {b}? Answer: {hundreds}",
    "Calculation: {a} + {b} = {hundreds}",
]

CONTEXT_PREFIXES = [
    "",
    "Arithmetic exercise. ",
    "Mental calculation. ",
    "Independent check. ",
    "Worked example. ",
]

TEMPLATES = [prefix + template for prefix in CONTEXT_PREFIXES for template in CORE_TEMPLATES]


def canonical_pair(a: int, b: int) -> Tuple[int, int]:
    return min(a, b), max(a, b)


def is_training_pair(a: int, b: int, seed: int) -> bool:
    pair = canonical_pair(a, b)
    if pair in {canonical_pair(*value) for value in EXPLICIT_HELDOUT}:
        return False
    digest = hashlib.sha256(f"{seed}:{pair[0]}:{pair[1]}".encode("ascii")).digest()
    # Reserve three quarters of unordered operand pairs for fresh interventions.
    return int.from_bytes(digest[:4], "big") % 4 == 0


def candidate_rows(seed: int) -> Dict[Tuple[int, int], List[Dict[str, str]]]:
    strata: Dict[Tuple[int, int], List[Dict[str, str]]] = {
        (carry, digit): [] for carry in (0, 1) for digit in range(10)
    }
    for a in range(10, 100):
        for b in range(10, 100):
            total = a + b
            if total < 100 or not is_training_pair(a, b, seed):
                continue
            carry = int((a % 10) + (b % 10) >= 10)
            tens_digit = (total // 10) % 10
            contrast_digit = (tens_digit - 1) % 10 if carry else (tens_digit + 1) % 10
            for template_index, template in enumerate(TEMPLATES, start=1):
                strata[(carry, tens_digit)].append(
                    {
                        "Operand1": str(a),
                        "Operand2": str(b),
                        "Answer": str(total),
                        "DistractorAnswer": str(total - 10 if carry else total + 10),
                        "IsCarry": str(carry),
                        "OnesDigitPair": f"{a % 10}+{b % 10}",
                        "OutputTensDigit": str(tens_digit),
                        "ContrastTensDigit": str(contrast_digit),
                        "TeacherForcedPrefix": str(total // 100),
                        "TemplateID": str(template_index),
                        "PairPartition": "train",
                        "sentence": template.format(a=a, b=b, hundreds=total // 100),
                    }
                )
    return strata


def build_rows(count: int = 10_000, seed: int = DEFAULT_SEED) -> List[Dict[str, str]]:
    if count % 2:
        raise ValueError("count must be even for exact carry/no-carry balance")
    per_class = count // 2
    carry_quota = {digit: per_class // 10 for digit in range(10)}
    for digit in range(per_class % 10):
        carry_quota[digit] += 1
    # A no-carry sum of two two-digit operands cannot have output tens digit 9:
    # reaching 190 necessarily requires a carry from the ones column.
    no_carry_quota = {digit: per_class // 9 for digit in range(9)}
    for digit in range(per_class % 9):
        no_carry_quota[digit] += 1
    quotas = {(1, digit): value for digit, value in carry_quota.items()}
    quotas.update({(0, digit): value for digit, value in no_carry_quota.items()})
    strata = candidate_rows(seed)
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    for key in sorted(quotas):
        candidates = strata[key]
        rng.shuffle(candidates)
        required = quotas[key]
        if len(candidates) < required:
            raise ValueError(
                f"Stratum carry={key[0]}, tens={key[1]} has {len(candidates)} candidates; "
                f"{required} required"
            )
        rows.extend(candidates[:required])
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["PromptID"] = f"math-large-{index:05d}"

    sentences = [row["sentence"] for row in rows]
    if len(sentences) != len(set(sentences)):
        raise AssertionError("Generated mathematics prompts are not unique")
    balance = Counter((row["IsCarry"], row["OutputTensDigit"]) for row in rows)
    carry_counts = Counter(row["IsCarry"] for row in rows)
    if carry_counts != Counter({"0": per_class, "1": per_class}):
        raise AssertionError(f"Carry classes are not balanced: {carry_counts}")
    if max(no_carry_quota.values()) - min(no_carry_quota.values()) > 1:
        raise AssertionError(f"No-carry output digits are not near-balanced: {balance}")
    return rows


def write_rows(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "PromptID",
        "Operand1",
        "Operand2",
        "Answer",
        "DistractorAnswer",
        "IsCarry",
        "OnesDigitPair",
        "OutputTensDigit",
        "ContrastTensDigit",
        "TeacherForcedPrefix",
        "TemplateID",
        "PairPartition",
        "sentence",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate position-matched carry prompts")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default="data/addition_large_10000.csv")
    args = parser.parse_args()

    rows = build_rows(args.count, args.seed)
    write_rows(rows, Path(args.output))
    print(f"Saved {len(rows)} unique prompts to {args.output}")
    print("Carry counts:", dict(sorted(Counter(row["IsCarry"] for row in rows).items())))
    print("Distinct unordered operand pairs:", len({canonical_pair(int(row['Operand1']), int(row['Operand2'])) for row in rows}))


if __name__ == "__main__":
    main()
