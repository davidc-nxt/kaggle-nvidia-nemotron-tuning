# Kaggle · NVIDIA Nemotron Model Reasoning Challenge

LoRA fine-tuning of **NVIDIA Nemotron-3-Nano-30B-A3B** to solve "Alice's Wonderland" reasoning puzzles.

- **Competition:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- **Deadline:** 2026-06-15 23:59 UTC (entry/merger cut-off 2026-06-08)
- **Prize pool:** $106,388 USD · top live score 0.87 (3,117 teams)
- **Submission format:** `submission.zip` containing a LoRA adapter (rank ≤ 32) with `adapter_config.json`

See [`references/notes.md`](references/notes.md) for the full task / evaluation / rules summary, and [`references/model_card/README.md`](references/model_card/README.md) for base-model specifics.

---

## Repository layout

```
.
├── data/
│   ├── raw/           train.csv, test.csv (downloaded from Kaggle — gitignored)
│   ├── processed/     cleaned / categorised splits
│   └── synthetic/     home-grown puzzle generations
├── notebooks/
│   ├── 01_eda.ipynb              Inspect the six puzzle families
│   └── 02_train_kaggle.ipynb     Kaggle-runnable LoRA training notebook
├── references/
│   ├── demo/          Official submission demo (Ryan Holbrook)
│   ├── model_card/    Local notes on Nemotron-3-Nano-30B
│   └── notes.md       Competition rules + evaluation summary
├── src/
│   ├── data.py        load_train / load_test / category labeller
│   ├── metric.py      local re-implementation of the NVIDIA Nemotron Metric
│   ├── prompts.py     system prompt + chat / SFT formatters
│   └── puzzles/       (placeholders) per-category synthetic generators
├── submissions/       built submission.zip artifacts (gitignored)
├── requirements.txt
├── .env.example       → copy to .env, then `export $(cat .env | xargs)`
└── README.md
```

---

## First-time setup (local — already done by the bootstrap)

```bash
# Python 3.13 venv
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Auth (the new short-lived Kaggle API token)
cp .env.example .env  # paste your KGAT_... token
export KAGGLE_API_TOKEN=$(grep KAGGLE_API_TOKEN .env | cut -d= -f2)

# Pull data
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/raw
unzip -o data/raw/nvidia-nemotron-model-reasoning-challenge.zip -d data/raw
```

The repo currently has the data already pulled at `data/raw/{train,test}.csv` (gitignored).

---

## Day-to-day commands

```bash
# Activate the env every shell
source .venv/bin/activate
export KAGGLE_API_TOKEN=$(grep KAGGLE_API_TOKEN .env | cut -d= -f2)

# Launch JupyterLab for EDA / local development
jupyter lab

# Check leaderboard
kaggle competitions leaderboard nvidia-nemotron-model-reasoning-challenge -s

# Check your submissions
kaggle competitions submissions nvidia-nemotron-model-reasoning-challenge

# Submit (once you have submissions/submission.zip ready)
kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge \
    -f submissions/submission.zip -m "describe approach here"
```

---

## Where training actually happens

Local M5 / 16 GB cannot train this model — the base alone is ~63 GB BF16. Training runs on
**Kaggle Notebooks** with GPU. Workflow:

1. Iterate on data / prompts / generators **locally** (this repo).
2. Upload `data/processed/*.parquet` (or `data/synthetic/*`) as a private **Kaggle Dataset**.
3. Open `notebooks/02_train_kaggle.ipynb` on Kaggle, attach:
   - the competition data (input)
   - the base model `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` (input)
   - your private dataset of curated/synthetic training rows (input)
4. Run training, save adapter to `/kaggle/working`, zip into `submission.zip`.
5. Submit either directly from the notebook or by downloading + `kaggle competitions submit ...`.

---

## Approach scratchpad

Initial intuition (refine as we go):

- **Synthetic data is the lever.** Every puzzle family has a closed-form generator; we can mint
  arbitrarily many train rows with ground-truth answers.
- **CoT with boxed format compliance.** Train completions in the exact shape the grader expects:
  reasoning trace ending in `\boxed{...}`.
- **One adapter, balanced mix.** Curriculum/mixture-weighting across the six categories rather than
  per-category adapters (the grader uses one adapter).
- **Eval locally with `src/metric.py`** before each submission to avoid burning the 5/day cap.

Track progress in `notebooks/01_eda.ipynb` first; capture decisions in commit messages.
