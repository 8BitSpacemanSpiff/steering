# Running With Public Open Causal LMs

This branch keeps the original Apple `ml-selfcond` AP pipeline, but adds response
collection support for modern decoder-only Hugging Face causal LMs.

Use models that do not require gated Hugging Face approval. A small first target:

```bash
MODEL_NAME="Qwen/Qwen2.5-0.5B"
RESULTS_DIR="/tmp/selfcond-qwen05"
```

Install modern dependencies on the VM:

```bash
python -m venv env
source env/bin/activate
pip install -U pip wheel
pip install -r requirements-modern.txt
pip install -e .
python -c "import nltk; nltk.download('punkt')"
```

Step 1 is still the AP baseline. First cache responses:

```bash
python scripts/compute_responses.py \
  --model-name-or-path "$MODEL_NAME" \
  --data-path assets/football_small \
  --responses-path "$RESULTS_DIR" \
  --device cuda \
  --seq-len 128 \
  --inf-batch-size 8
```

Then compute AP expertise:

```bash
python scripts/compute_expertise.py \
  --root-dir "$RESULTS_DIR" \
  --model-name "$MODEL_NAME" \
  --concepts assets/football_small/concept_list.csv
```

Expected output:

```text
$RESULTS_DIR/Qwen/Qwen2.5-0.5B/sense/football-1_04_00__/expertise/expertise.csv
```

Notes:

- Response collection currently keeps decoder MLP projection outputs for Qwen,
  Mistral, Phi, and Llama-style models.
- The scoring code is model-agnostic after responses are cached.
- Avoid gated models for the first pass; Qwen 0.5B is public and small enough
  for quick iteration.

## First GMM Probe

After the AP baseline exists, run a small class-conditional GMM probe on the top
AP units:

```bash
CONCEPT="football-1_04_00__"
CONCEPT_DIR="$RESULTS_DIR/$MODEL_NAME/sense/$CONCEPT"

python scripts/probe_gmm.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --top-n 50 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_probe_top50.csv"
```

The important columns are:

- `gmm_ap`: AP of the GMM density-ratio score.
- `gmm_auc`: AUROC of the same score.
- `pos_k` / `neg_k`: how many Gaussian modes BIC selected for positive and
  negative responses.
- `pos_means` / `neg_means`: component means back in the original activation
  scale.
- `on_mode_mean`: highest positive-mode mean, which is the candidate steering
  value for later.

For a rough early AP-vs-GMM comparison, sample candidate units across AP ranges:

```bash
python scripts/compare_gmm_candidates.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --top-n 100 \
  --random-n 100 \
  --per-bin 50 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_candidate_compare.csv"
```

This prints:

- correlations between `ap`, `diff_mean`, and `gmm_ap`;
- how often positive/negative responses choose 1, 2, or 3 modes;
- the top candidate units by GMM;
- units where `gmm_ap - ap` is largest.

If the candidate comparison finds medium-AP units that GMM improves, scan that
AP band more thoroughly:

```bash
python scripts/scan_gmm_band.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --min-ap 0.75 \
  --max-ap 0.9 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_scan_ap075_090.csv"
```

This is still cheaper than fitting GMMs for every unit, but it targets the most
interesting question: can GMM lift units that AP did not rank at the very top?

To inspect one lifted unit, print the sentences assigned to each GMM mode:

```bash
python scripts/inspect_gmm_unit.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --concept-json assets/football/sense/football-1_04_00__.json \
  --concept "$CONCEPT" \
  --layer "model.layers.0.mlp.gate_proj:0" \
  --unit 2691
```

Start with units from `gmm_scan_ap075_090.csv` where `gmm_minus_ap` is large and
`gmm_ap` is high. The output is not proof of sub-concepts yet; it is the first
human-readable check for whether modes look meaningful or like artifacts.

For a more compact terminal summary of each mode:

```bash
python scripts/characterize_gmm_unit_modes.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --concept-json assets/football/sense/football-1_04_00__.json \
  --concept "$CONCEPT" \
  --layer "model.layers.16.mlp.gate_proj:0" \
  --unit 4039
```

This prints simple statistics such as mode size, average activation, sentence
length, digit rate, keyword rates, and top terms.

To summarize several lifted units at once:

```bash
python scripts/summarize_gmm_scan_modes.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --concept-json assets/football/sense/football-1_04_00__.json \
  --concept "$CONCEPT" \
  --scan-csv "$CONCEPT_DIR/expertise/gmm_scan_ap075_090.csv" \
  --sort-by gmm_ap \
  --top-units 10 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_scan_top10_mode_summary.csv"
```

Use `--sort-by gmm_minus_ap` to focus on the units GMM most strongly lifts over
AP.

To aggregate a mode-summary CSV into evidence categories:

```bash
python scripts/summarize_mode_evidence.py \
  --mode-summary-csv "$CONCEPT_DIR/expertise/gmm_scan_top10_mode_summary.csv"

python scripts/summarize_mode_evidence.py \
  --mode-summary-csv "$CONCEPT_DIR/expertise/gmm_scan_top10_lift_mode_summary.csv"
```

This is a crude keyword-based pass, but it is useful for terminal-only triage:
it surfaces negative sports-confound modes, positive governance/history modes,
positive match/action modes, and possible artifact-heavy modes.

## GMM Expertise Table

Once the probes look useful, write GMM scores to a separate experiment-side CSV:

```bash
python scripts/compute_gmm_expertise.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --min-ap 0.75 \
  --max-ap 0.9 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_expertise_ap075_090.csv"
```

For a full all-unit run, omit `--min-ap` and `--max-ap`. This can be slow:

```bash
python scripts/compute_gmm_expertise.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --out-csv "$CONCEPT_DIR/expertise/gmm_expertise_all.csv"
```

The original AP `expertise.csv` is not modified. The new table includes
`gmm_score`, `gmm_ap`, `gmm_auc`, `diff_mean`, `pos_k`, `neg_k`,
`on_mode_mean`, component means, and the original AP/forcing columns.

To create a generation-friendly expertise CSV containing both AP and GMM
columns:

```bash
python scripts/merge_gmm_expertise.py \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --gmm-csv "$CONCEPT_DIR/expertise/gmm_expertise_ap075_090.csv" \
  --keep gmm \
  --out-csv "$CONCEPT_DIR/expertise/expertise_with_gmm_ap075_090.csv"
```

Use `--metric gmm_score` and `--forcing on_mode_mean` with this merged table
once generation support is ready.

## Steering Smoke Test

Compare AP-ranked forcing against GMM-ranked forcing with the same model and
prompt:

```bash
python scripts/generate_seq.py \
  --model-name-or-path "$MODEL_NAME" \
  --expertise "$CONCEPT_DIR/expertise/expertise.csv" \
  --length 40 \
  --prompt "The team" \
  --seed 0 5 \
  --temperature 0.8 \
  --top-p 0.9 \
  --metric ap \
  --forcing on_p50 \
  --num-units 10 \
  --device cuda \
  --no-save
```

```bash
python scripts/generate_seq.py \
  --model-name-or-path "$MODEL_NAME" \
  --expertise "$CONCEPT_DIR/expertise/expertise_with_gmm_ap075_090.csv" \
  --length 40 \
  --prompt "The team" \
  --seed 0 5 \
  --temperature 0.8 \
  --top-p 0.9 \
  --metric gmm_score \
  --forcing on_mode_mean \
  --num-units 10 \
  --device cuda \
  --no-save
```

This is only a smoke test. If it runs, do a controlled sweep over prompts,
`num-units`, and AP-vs-GMM tables next.
