"""Prompt templates for fine-tuning and inference.

The evaluator extracts the final answer from a `\\boxed{...}` LaTeX command,
so every prompt MUST instruct the model to wrap the final answer that way.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are solving an 'Alice's Wonderland' reasoning puzzle. "
    "Read the input/output examples to infer the hidden rule, then apply the rule "
    "to the final input. Reason step by step. Place your final answer inside "
    "\\boxed{...}. Do not output anything after the closing brace."
)


def build_chat(prompt: str) -> list[dict[str, str]]:
    """Chat-template messages compatible with Nemotron's HF tokenizer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def build_sft_text(prompt: str, answer: str) -> dict[str, str]:
    """Return a single SFT row with separable `prompt` and `completion` fields.

    Use with TRL's `SFTTrainer` or your own collator. The completion ends with
    the boxed answer so the model learns the required output format.
    """
    completion = f"\\boxed{{{answer}}}"
    return {"prompt": prompt, "completion": completion}
