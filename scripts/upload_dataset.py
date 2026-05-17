"""Push the SFT parquet to Kaggle as a private dataset so the training notebook
can attach it as input.

Run locally after building sft_v1.parquet. Subsequent runs create new versions
via `--version <message>`.

Usage:
    python scripts/upload_dataset.py                            # first upload
    python scripts/upload_dataset.py --version "add cipher"     # new version
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SFT_PATH = REPO / "data" / "processed" / "sft_v1.parquet"
STAGE = REPO / "data" / "_kaggle_dataset_stage"
DEFAULT_SLUG = "wonderland-sft-v1"


def _need_token() -> None:
    if not os.environ.get("KAGGLE_API_TOKEN"):
        # fall back to .env
        env = REPO / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("KAGGLE_API_TOKEN="):
                    os.environ["KAGGLE_API_TOKEN"] = line.split("=", 1)[1].strip()
                    return
        print("error: KAGGLE_API_TOKEN not set (neither in env nor in .env)", file=sys.stderr)
        sys.exit(1)


def _kaggle_username() -> str:
    """Resolve the Kaggle username.

    Order of preference:
      1. `KAGGLE_USERNAME` env var (set in .env)
      2. `~/.kaggle/kaggle.json` classic credentials file (username field)
      3. Fail with a clear message

    `kaggle whoami` was removed in CLI 2.1.x, so we can't lean on it for the
    new short-lived KGAT_ token format.
    """
    if u := os.environ.get("KAGGLE_USERNAME"):
        return u

    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("KAGGLE_USERNAME="):
                return line.split("=", 1)[1].strip()

    cred = Path.home() / ".kaggle" / "kaggle.json"
    if cred.exists():
        try:
            data = json.loads(cred.read_text())
            u = data.get("username")
            if u and u != "YOUR_USERNAME":
                return u
        except Exception:
            pass

    print(
        "error: Kaggle username unknown. Set KAGGLE_USERNAME=<your-username> in .env "
        "or in ~/.kaggle/kaggle.json.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slug", default=DEFAULT_SLUG, help="Dataset slug (under your user)")
    p.add_argument("--title", default="Wonderland SFT v1", help="Human-readable title")
    p.add_argument("--version", default=None, help="If set, push a new version with this changelog message")
    p.add_argument("--public", action="store_true", help="Make the dataset public (default: private)")
    args = p.parse_args()

    _need_token()
    if not SFT_PATH.exists():
        print(f"error: {SFT_PATH} missing — run scripts/build_sft_data.py first", file=sys.stderr)
        return 1

    user = _kaggle_username()
    print(f"kaggle user: {user}")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy2(SFT_PATH, STAGE / SFT_PATH.name)

    metadata = {
        "title": args.title,
        "id": f"{user}/{args.slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"staged dataset at {STAGE} (id={metadata['id']})")

    if args.version is None:
        # Kaggle CLI 2.1.x defaults new datasets to private; `--private` was removed.
        cmd = ["kaggle", "datasets", "create", "-p", str(STAGE)]
        if args.public:
            cmd.append("--public")
    else:
        cmd = ["kaggle", "datasets", "version", "-p", str(STAGE), "-m", args.version, "--dir-mode", "zip"]

    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
