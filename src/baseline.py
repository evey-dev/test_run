"""Run the baseline evaluation for capitals, addition, and units prompts.

For each behaviour, this script loads the configured local language model,
queries the prompts, records top-k token probabilities at the final input
position, and writes JSON results to the baseline output directory.

Optional CLI arguments:
- --model-config: path to the YAML file describing the model to load
- --config: path to the baseline YAML configuration file
- --prefix: number of prefix-guided continuation attempts for capitals/units (0 disables)
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
import yaml

from src.model_loader import load_model_and_tokenizer, set_seed
from src.prompts.prompt_utils import load_prompts, format_prompt, get_expected_answers, get_expected_token


def get_top_k_logits(model, tokenizer, text: str, top_k: int, cfg: dict) -> dict:
    """
    Tokenise text, run a single forward pass, return top-k tokens and their
    probabilities at the *last* input token position (i.e. what comes next).
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)

    # logits shape: [1, seq_len, vocab_size]
    last_logits = outputs.logits[0, -1, :]           # [vocab_size]
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k)

    top_tokens = [tokenizer.decode(tid, skip_special_tokens=True) for tid in top_ids]
    return {
        "top_tokens": top_tokens,
        "top_probs": top_probs.cpu().float().tolist(),
        "top_ids": top_ids.cpu().tolist(),
        "input_tokens": tokenizer.convert_ids_to_tokens(input_ids[0]),
        "n_input_tokens": input_ids.shape[1],
    }


def get_greedy_completion(model, tokenizer, text: str, max_new_tokens: int = 4) -> str:
    """Generate a short completion for a prompt and return the newly generated text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text[len(text):].strip()


def normalize_for_prefix_match(text: str) -> str:
    """Collapse whitespace and punctuation so prefix checks focus on the letters/digits."""
    return re.sub(r"[^A-Za-z0-9]", "", text).lower()


def normalize_for_answer_match(text: str) -> str:
    """Canonicalize answer strings so singular/plural unit forms compare as equivalent."""
    normalized = normalize_for_prefix_match(text)
    if not normalized:
        return ""
    if normalized.endswith("ies") and len(normalized) > 3:
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("s") and not normalized.endswith("ss") and len(normalized) > 1:
        normalized = normalized[:-1]
    return normalized


def get_prefix_aware_completion(model, tokenizer, text: str, seed_token: str, expected_answers: list[str], max_new_tokens: int = 8) -> str:
    """Greedily continue generation from a promising seed token while it stays a prefix of an acceptable answer."""
    expected_norms = {normalize_for_answer_match(ans) for ans in expected_answers if normalize_for_answer_match(ans)}
    seed_norm = normalize_for_prefix_match(seed_token)
    if not expected_norms or not seed_norm or not any(expected.startswith(seed_norm) for expected in expected_norms):
        return seed_token.strip()

    prompt_text = text + seed_token
    input_ids = tokenizer(prompt_text, return_tensors="pt").to(model.device)["input_ids"]
    generated_text = seed_token

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
            if not next_token:
                break

            generated_text += next_token
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], device=input_ids.device)],
                dim=1,
            )

            generated_norm = normalize_for_prefix_match(generated_text)
            if not any(expected.startswith(generated_norm) for expected in expected_norms):
                break
            if generated_norm in expected_norms:
                break

    return generated_text.strip()


def run_baseline(behaviour: str, model, tokenizer, cfg: dict, baseline_cfg: dict, prefix_limit: int = 1) -> list[dict]:
    """Run all prompts for one behaviour and return result records."""
    prompt_data = load_prompts(behaviour)
    top_k = baseline_cfg.get("save_top_k_tokens", 20)
    results = []

    print(f"\n=== Behaviour: {behaviour} ({len(prompt_data['prompts'])} prompts) ===")
    for p in prompt_data["prompts"]:
        text = format_prompt(p)
        expected = get_expected_token(p)
        expected_answers = get_expected_answers(p) if behaviour == "units" else [expected]

        t0 = time.time()
        logit_info = get_top_k_logits(model, tokenizer, text, top_k, cfg)
        elapsed = time.time() - t0

        # Check whether the model gives the expected answer. For arithmetic, we
        # evaluate the actual generated completion rather than only the first
        # next-token logit.
        expected_rank = None
        expected_prob = None
        greedy_completion = ""
        top1_token = logit_info["top_tokens"][0]
        top1_prob = logit_info["top_probs"][0]

        if behaviour == "addition":
            greedy_completion = get_greedy_completion(model, tokenizer, text, max_new_tokens=4)
            completion_clean = greedy_completion.strip().rstrip(" .,:;")
            if completion_clean == expected.strip() or completion_clean.startswith(expected.strip()):
                expected_rank = 0
                expected_prob = 1.0
            top1_token = completion_clean or greedy_completion
        elif behaviour in {"capitals", "units"}:
            initial_top1 = logit_info["top_tokens"][0]
            initial_norm = normalize_for_prefix_match(initial_top1)
            expected_norms = {normalize_for_answer_match(ans) for ans in expected_answers if normalize_for_answer_match(ans)}
            top1_token = initial_top1

            if prefix_limit > 0:
                for rank in range(min(prefix_limit, len(logit_info["top_tokens"]))):
                    candidate_token = logit_info["top_tokens"][rank]
                    candidate_norm = normalize_for_prefix_match(candidate_token)
                    if not candidate_norm or not any(expected.startswith(candidate_norm) for expected in expected_norms):
                        continue

                    candidate_completion = get_prefix_aware_completion(
                        model,
                        tokenizer,
                        text,
                        candidate_token,
                        expected_answers,
                        max_new_tokens=8,
                    )
                    candidate_completion_norm = normalize_for_answer_match(candidate_completion)
                    if candidate_completion_norm in expected_norms:
                        greedy_completion = candidate_completion
                        expected_rank = rank
                        expected_prob = logit_info["top_probs"][rank]
                        if rank==0:
                            top1_token = greedy_completion
                        break

                    if candidate_completion_norm and len(candidate_completion_norm) > len(normalize_for_prefix_match(top1_token)):
                        greedy_completion = candidate_completion
                        if rank==0:
                            top1_token = greedy_completion

                if not greedy_completion and initial_norm and any(expected.startswith(initial_norm) for expected in expected_norms):
                    greedy_completion = get_prefix_aware_completion(model, tokenizer, text, initial_top1, expected_answers, max_new_tokens=8)
                    if normalize_for_answer_match(greedy_completion) in expected_norms:
                        expected_rank = 0
                        expected_prob = logit_info["top_probs"][0]
                    if rank==0:
                        top1_token = greedy_completion or initial_top1
            else:
                for rank, (tok, prob) in enumerate(zip(logit_info["top_tokens"], logit_info["top_probs"])):
                    if tok.strip() == expected.strip():
                        expected_rank = rank
                        expected_prob = prob
                        break

            if not greedy_completion:
                for rank, (tok, prob) in enumerate(zip(logit_info["top_tokens"], logit_info["top_probs"])):
                    if tok.strip() == expected.strip():
                        expected_rank = rank
                        expected_prob = prob
                        break
        else:
            for rank, (tok, prob) in enumerate(zip(logit_info["top_tokens"], logit_info["top_probs"])):
                if tok.strip() == expected.strip():
                    expected_rank = rank
                    expected_prob = prob
                    break
        print(logit_info["top_tokens"])
        display_rank = expected_rank
        display_status = "✓" if expected_rank == 0 else (f"rank {expected_rank}" if expected_rank is not None else "NOT IN TOP-K")
        if behaviour in {"capitals", "units"} and greedy_completion and expected_rank is not None and expected_rank != 0:
            display_status = f"rank {expected_rank}*"
        record = {
            "id": p["id"],
            "behaviour": behaviour,
            "prompt_text": text,
            "expected_token": expected,
            "expected_rank": expected_rank,        # None if not in top-k
            "expected_prob": expected_prob,
            "top1_token": top1_token,
            "top1_prob": top1_prob,
            "greedy_completion": greedy_completion,
            "top_k_tokens": logit_info["top_tokens"],
            "top_k_probs": logit_info["top_k_probs"] if "top_k_probs" in logit_info else logit_info["top_probs"],
            "n_input_tokens": logit_info["n_input_tokens"],
            "elapsed_s": round(elapsed, 3),
            # Metadata for reproducibility pack
            "model_id": cfg["model_id"],
            "seed": cfg["seed"],
            "torch_dtype": cfg.get("torch_dtype", "bfloat16"),
        }
        print(f"  [{p['id']}] expected='{expected.strip()}' top1='{record['top1_token'].strip()}' | {display_status} | {elapsed:.1f}s")
        results.append(record)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--config", default="configs/baseline_config.yaml")
    parser.add_argument("--prefix", type=int, default=1, choices=range(0, 11), help="For capitals, try prefix-guided continuation from the top N ranks while the generated text remains consistent with the expected answer (0 disables).")
    args = parser.parse_args()

    with open(args.config) as f:
        baseline_cfg = yaml.safe_load(f)

    model, tokenizer, cfg = load_model_and_tokenizer(args.model_config)

    out_dir = Path(baseline_cfg.get("output_dir", "outputs/baselines"))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for behaviour in baseline_cfg["behaviours"]:
        results = run_baseline(behaviour, model, tokenizer, cfg, baseline_cfg, prefix_limit=args.prefix)
        all_results[behaviour] = results

        # Save per-behaviour JSON
        out_path = out_dir / f"{behaviour}_baselines.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved -> {out_path}")

    # Save combined summary
    summary_path = out_dir / "all_baselines.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll baselines saved -> {summary_path}")

    # Print a quick accuracy summary
    print("\n=== Summary ===")
    for behaviour, results in all_results.items():
        n = len(results)
        correct = sum(1 for r in results if r["expected_rank"] == 0)
        in_topk = sum(1 for r in results if r["expected_rank"] is not None)
        print(f"  {behaviour}: {correct}/{n} top-1 correct, {in_topk}/{n} in top-k")


if __name__ == "__main__":
    main()