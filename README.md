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
│   ├── raw/           train.csv, test.csv (gitignored — pulled from Kaggle)
│   ├── processed/     cipher_vocab.json, sft_v1.parquet (gitignored)
│   └── synthetic/     synth_v1.parquet (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb              Six puzzle families, sample inspection
│   └── 02_train_kaggle.ipynb     Kaggle-runnable LoRA training notebook
├── references/
│   ├── demo/          Official submission demo (Ryan Holbrook)
│   ├── model_card/    Local notes on Nemotron-3-Nano-30B
│   └── notes.md       Competition rules + evaluation summary
├── scripts/
│   ├── build_notebooks.py       Regenerate the .ipynb files from Python
│   ├── build_cipher_vocab.py    Extract the 77-word cipher vocabulary
│   ├── build_sft_data.py        Merge orig + synth, add CoT, emit SFT parquet
│   ├── eval_solvers.py          Score per-category solvers on train.csv
│   ├── generate_synthetic.py    Mint synthetic (prompt, answer) rows
│   └── upload_dataset.py        Push sft_v1.parquet to Kaggle as private dataset
├── src/
│   ├── data.py        Loader + category labeller
│   ├── metric.py      Local re-impl of the NVIDIA Nemotron Metric
│   ├── prompts.py     System prompt + chat / SFT formatters
│   ├── reasoning.py   Build CoT completions per category
│   └── puzzles/
│       ├── numeral.py   int -> Roman (100% train acc)
│       ├── units.py     y = k·x fit              (100% train acc)
│       ├── gravity.py   d = 0.5·g·t² fit          (100% train acc)
│       ├── binary.py    SHA-2-style template brute force (78.2% train acc)
│       ├── cipher.py    Vocab-constrained substitution (100% train acc)
│       ├── generators.py  Synthetic generators per category
│       └── registry.py    Dispatch by category
├── submissions/       built submission.zip artifacts (gitignored)
├── requirements.txt
├── .env.example       Copy to .env with your KAGGLE_API_TOKEN
└── README.md
```

## Current solver coverage

Five of six categories solved in code on `train.csv`:

| Category | Rows | Solver acc | Notes                                                |
| -------- | ---: | ---------: | ---------------------------------------------------- |
| numeral  | 1576 |     100.0% | Roman 1-99                                           |
| units    | 1594 |     100.0% | `y = k·x`, median-ratio fit                          |
| gravity  | 1597 |     100.0% | `d = 0.5·g·t²`, median-g fit                         |
| cipher   | 1576 |     100.0% | Substitution + 77-word closed-class vocabulary match |
| binary   | 1602 |      78.2% | Brute-force over SHA-2-style template library        |
| equation | 1555 |      n/a   | Open problem — per-operator rewrite rules            |
| **all**  | 9500 |    **95.6%** | (across covered rows, 7596 / 7945)                |

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

## End-to-end pipeline

Local M5 / 16 GB cannot host the 63 GB BF16 base — heavy training runs on Kaggle Notebooks.
Data prep and oracle solving happen locally; training and submission happen on Kaggle.

```bash
source .venv/bin/activate
export KAGGLE_API_TOKEN=$(grep KAGGLE_API_TOKEN .env | cut -d= -f2)

# 1. Pull competition data
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/raw
unzip -o data/raw/nvidia-nemotron-model-reasoning-challenge.zip -d data/raw

# 2. Score local solvers on train
python scripts/eval_solvers.py

# 3. Build derived artifacts
python scripts/build_cipher_vocab.py                            # data/processed/cipher_vocab.json
python scripts/generate_synthetic.py --per-category 5000        # data/synthetic/synth_v1.parquet
python scripts/build_sft_data.py                                # data/processed/sft_v1.parquet

# 4. Upload SFT data as a private Kaggle dataset (slug "wonderland-sft-v1")
python scripts/upload_dataset.py

# 5. On Kaggle: open notebooks/02_train_kaggle.ipynb, attach
#    - Competition data        (input)
#    - metric/nemotron-3-nano-30b-a3b-bf16/transformers/default  (input)
#    - <you>/wonderland-sft-v1                                   (input)
#    Then run top-to-bottom. Output: /kaggle/working/submission.zip.

# 6. Submit
kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge \
    -f submissions/submission.zip -m "first SFT pass on synth+orig"
```

## Approach scratchpad

- **Synthetic data is the lever.** Five of six categories have closed-form generators producing
  unlimited verified (prompt, answer) rows. SFT mix is 9.5k original + 25k synthetic.
- **CoT with boxed compliance.** Every completion in `sft_v1.parquet` ends in `\boxed{...}` to
  match the grader's extractor exactly.
- **One adapter, balanced mix.** Grader runs a single LoRA — we balance categories during data
  generation rather than train per-category heads.
- **Local metric for offline scoring.** Use `src/metric.py` before burning the 5/day submission cap.

Open items: crack equation puzzles (1,555 rows, ~16% of train, still on `\boxed{answer}` only);
improve binary template coverage past 78%; add longer / more varied CoT traces.
