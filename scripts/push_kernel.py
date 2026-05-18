"""Stage and push the Kaggle training kernel.

This wraps `kaggle kernels push` so the user can fire-and-forget. It stages
the training notebook alongside a fresh `kernel-metadata.json` that wires up
the three required inputs (competition data + base model + private SFT
dataset).

GPU TYPE: Kaggle's CLI metadata exposes only `enable_gpu: true/false`, not a
specific accelerator. The free P100 (sm_60) is NOT supported by current
PyTorch wheels, so you must manually switch the kernel to **GPU T4 x2** (free)
or **GPU L4 x4** (Pro) via:
    https://www.kaggle.com/code/<user>/<slug>/edit  →  Settings → Accelerator

Usage:
    python scripts/push_kernel.py            # push (Kaggle versions automatically)
    python scripts/push_kernel.py --poll 600 # also poll status for up to N seconds
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "02_train_kaggle.ipynb"
STAGE = REPO / "data" / "_kaggle_kernel_stage"
DEFAULT_SLUG = "wonderland-nemotron-train"
DEFAULT_DATASET_SLUG = "wonderland-sft-v1"


def _load_env() -> None:
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME"):
        if k in os.environ:
            continue
        env = REPO / ".env"
        if not env.exists():
            continue
        for line in env.read_text().splitlines():
            if line.startswith(f"{k}="):
                os.environ[k] = line.split("=", 1)[1].strip()


def _kaggle_username() -> str:
    _load_env()
    u = os.environ.get("KAGGLE_USERNAME")
    if not u:
        print("error: KAGGLE_USERNAME not set in env or .env", file=sys.stderr)
        sys.exit(1)
    return u


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slug", default=DEFAULT_SLUG)
    p.add_argument("--title", default="Wonderland Nemotron Train")
    p.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    p.add_argument(
        "--wheels-dataset",
        default="aditya2909rb/nemotron-native-wheels",
        help="Public Kaggle Dataset with mamba_ssm + causal_conv1d + triton + einops wheels (cp312)",
    )
    p.add_argument(
        "--model",
        default="metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
        help="Kaggle Models slug (must include version suffix)",
    )
    p.add_argument(
        "--competition",
        default="nvidia-nemotron-model-reasoning-challenge",
    )
    p.add_argument("--poll", type=int, default=0, help="If >0, poll status for up to N seconds")
    args = p.parse_args()

    if not NOTEBOOK.exists():
        print(f"error: {NOTEBOOK} missing — run scripts/build_notebooks.py", file=sys.stderr)
        return 1

    user = _kaggle_username()
    kernel_id = f"{user}/{args.slug}"
    print(f"kernel id: {kernel_id}")

    # Stage notebook + metadata. Kaggle requires the notebook to sit next to the
    # metadata file inside the staging directory.
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    staged_nb = STAGE / NOTEBOOK.name
    shutil.copy2(NOTEBOOK, staged_nb)

    metadata = {
        "id": kernel_id,
        "title": args.title,
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        # Internet is competition-restricted under non-T4 accelerators. We rely on
        # pre-installed packages + the ryanholbrook/nvidia-utility-script kernel
        # source for mamba_ssm / CUTLASS DSL.
        "enable_internet": "false",
        "dataset_sources": [
            f"{user}/{args.dataset_slug}",
            args.wheels_dataset,
        ],
        "competition_sources": [args.competition],
        "kernel_sources": [],
        "model_sources": [args.model],
    }
    (STAGE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"staged at {STAGE}")
    print(f"  inputs: dataset={user}/{args.dataset_slug}, model={args.model}, competition={args.competition}")

    push_cmd = ["kaggle", "kernels", "push", "-p", str(STAGE)]
    print("running:", " ".join(push_cmd))
    r = subprocess.run(push_cmd, check=False)
    if r.returncode != 0:
        return r.returncode

    print()
    print("=" * 70)
    print("IMPORTANT — set the GPU type in the kernel settings (P100 will fail):")
    print(f"  https://www.kaggle.com/code/{kernel_id}/edit")
    print("  Settings → Accelerator → 'GPU T4 x2' (or 'GPU L4 x4' on Pro)")
    print("=" * 70)
    print()
    print(f"check status:   kaggle kernels status {kernel_id}")
    print(f"pull output:    kaggle kernels output -p submissions/{args.slug} {kernel_id}")
    print(f"submit zip:     kaggle competitions submit -c {args.competition} \\")
    print(f"                  -f submissions/{args.slug}/submission.zip -m '<message>'")

    if args.poll > 0:
        print()
        print(f"polling status every 30s for up to {args.poll}s...")
        deadline = time.time() + args.poll
        while time.time() < deadline:
            time.sleep(30)
            out = subprocess.run(
                ["kaggle", "kernels", "status", kernel_id],
                capture_output=True, text=True, check=False,
            )
            line = (out.stdout or "").strip()
            print(f"  [{time.strftime('%H:%M:%S')}] {line}")
            low = line.lower()
            if "complete" in low or "error" in low or "cancel" in low:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
