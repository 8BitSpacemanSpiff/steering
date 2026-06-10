# Modern Open-Model GMM Steering Runbook

This branch keeps Apple's original `ml-selfcond` AP expertise pipeline, but adds
support for public decoder-only Hugging Face causal LMs and a class-conditional
GMM expertise score.

The current research question is narrow:

> Can GMMs find useful concept experts, especially below the absolute top AP
> tier, by modeling multimodal positive/negative activation structure?

The current answer from the football/Qwen 0.5B run is:

> Full AP top units are very strong. GMM is not yet a replacement for AP.
> Within the mid-AP candidate band `0.75 <= AP < 0.9`, GMM picked units that
> steered generation toward sports/football more clearly than AP picked from
> the same band. That is the strongest result so far.

All commands below are written for the VM. The local machine was used for code
editing and GitHub pushes only; experiments should run on the H100 VM.

## Repository State

Branch:

```bash
codex/baseline-modern-1b
```

Remote:

```bash
https://github.com/8BitSpacemanSpiff/steering.git
```

Pull the latest code on the VM:

```bash
cd ~/steering
git pull
```

## What Was Added

Modern model support:

- `AutoModelForCausalLM` loading in `selfcond/models.py`.
- Public model families: Qwen, Mistral, Phi, Llama-style decoder-only models.
- Decoder MLP response collection for layers like:
  - `model.layers.N.mlp.gate_proj:0`
  - `model.layers.N.mlp.up_proj:0`
  - `model.layers.N.mlp.down_proj:0`
- Generation fix for modern causal LMs: generation recomputes the full prefix
  each token with `use_cache=False`, avoiding the old GPT-2 cache/mask path.
- BPE-safe decode with `clean_up_tokenization_spaces=False`.

GMM support:

- `selfcond/gmm.py`
- `scripts/probe_gmm.py`
- `scripts/compare_gmm_candidates.py`
- `scripts/scan_gmm_band.py`
- `scripts/inspect_gmm_unit.py`
- `scripts/characterize_gmm_unit_modes.py`
- `scripts/summarize_gmm_scan_modes.py`
- `scripts/summarize_mode_evidence.py`
- `scripts/compute_gmm_expertise.py`
- `scripts/merge_gmm_expertise.py`

Steering support:

- `scripts/generate_seq.py` can rank by `ap` or `gmm_score`.
- Original replacement-style steering remains the default:
  - `--intervention-mode set`
- Experimental additive steering still exists:
  - `--intervention-mode add`
- For this project, the recommended path is replacement steering.

## Environment

Use a public model that does not require Hugging Face approval. The current
small target is:

```bash
export MODEL_NAME="Qwen/Qwen2.5-0.5B"
export RESULTS_DIR="/tmp/selfcond-qwen05-football"
export CONCEPT="football-1_04_00__"
export CONCEPT_DIR="$RESULTS_DIR/$MODEL_NAME/sense/$CONCEPT"
```

Install on the VM:

```bash
cd ~/steering
python -m venv env
source env/bin/activate
pip install -U pip wheel
pip install -r requirements-modern.txt
pip install -e .
python -c "import nltk; nltk.download('punkt')"
```

If `pip install -r requirements-modern.txt` fails with a missing
`LICENSE.txt`, pull latest code. This branch now includes a `LICENSE` file and
install metadata compatible with editable install.

## Step 1: AP Baseline

Always get AP working first. This is the baseline we compare against.

Small smoke run:

```bash
python scripts/compute_responses.py \
  --model-name-or-path "$MODEL_NAME" \
  --data-path assets/football_small \
  --responses-path "$RESULTS_DIR" \
  --device cuda \
  --seq-len 128 \
  --inf-batch-size 8

python scripts/compute_expertise.py \
  --root-dir "$RESULTS_DIR" \
  --model-name "$MODEL_NAME" \
  --concepts assets/football_small/concept_list.csv
```

Full football run:

```bash
python scripts/compute_responses.py \
  --model-name-or-path "$MODEL_NAME" \
  --data-path assets/football \
  --responses-path "$RESULTS_DIR" \
  --device cuda \
  --seq-len 128 \
  --inf-batch-size 8

python scripts/compute_expertise.py \
  --root-dir "$RESULTS_DIR" \
  --model-name "$MODEL_NAME" \
  --concepts assets/football/concept_list.csv
```

Expected AP file:

```text
$CONCEPT_DIR/expertise/expertise.csv
```

Terminal summary command:

```bash
python - <<'PY'
import pandas as pd

csv = "/tmp/selfcond-qwen05-football/Qwen/Qwen2.5-0.5B/sense/football-1_04_00__/expertise/expertise.csv"
df = pd.read_csv(csv)

cols = ["ap", "layer", "unit", "on_p50", "on_p90", "off_mean"]
print(df.sort_values("ap", ascending=False)[cols].head(30).to_string(index=False))
print()
print("rows:", len(df))
print("max ap:", df["ap"].max())
print("mean ap:", df["ap"].mean())
print("ap > 0.9:", (df["ap"] > 0.9).sum())
print("ap > 0.8:", (df["ap"] > 0.8).sum())
PY
```

Observed full-football AP summary:

```text
rows: 254976
max ap: 0.98596257
mean ap: 0.4157373306293534
ap > 0.9: 102
ap > 0.8: 353
```

Top AP units included:

```text
0.985963  model.layers.3.mlp.gate_proj:0  unit 3886
0.985670  model.layers.3.mlp.up_proj:0    unit 3886
0.985046  model.layers.1.mlp.up_proj:0    unit 922
```

## Step 2: First GMM Probe

Run a small top-AP GMM probe:

```bash
python scripts/probe_gmm.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --top-n 50 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_probe_top50.csv"
```

GMM method:

- For each unit, standardize that unit's responses.
- Fit one GMM to positive responses and one GMM to negative responses.
- Try `k = 1, 2, 3`.
- Select `k` by lowest BIC separately for positive and negative responses.
- Score each sentence by log density ratio:

```text
log p(response | positive GMM) - log p(response | negative GMM)
```

Important output columns:

- `gmm_ap`: AP of the GMM density-ratio score.
- `gmm_auc`: AUROC of the same score.
- `gmm_score`: same as `gmm_ap`, convenient for generation ranking.
- `gmm_minus_ap`: `gmm_ap - ap`.
- `diff_mean`: positive mean activation minus negative mean activation.
- `pos_k`, `neg_k`: selected number of modes.
- `pos_means`, `neg_means`: component means in original activation scale.
- `on_mode_mean`: largest positive component mean, used as a GMM steering target.

Observed top-50 AP probe:

```text
pos_k: 1 -> 36 units, 2 -> 13 units, 3 -> 1 unit
neg_k: 1 -> 3 units, 2 -> 22 units, 3 -> 25 units
```

Interpretation:

- Top AP units often have simple positive structure.
- Negative responses frequently split into modes.
- That means GMM is useful for identifying confounds as well as positive
  substructure.

## Step 3: Compare Candidate Units Across AP Bands

Run:

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

Observed result:

```text
rows: 399
corr(AP, GMM): 0.9858
```

Interpretation:

- GMM is highly correlated with AP overall.
- The useful difference is not global ranking replacement.
- The useful place to look is mid/high AP units that GMM lifts.

## Step 4: Scan The Useful AP Band

The interesting band from early probes was:

```text
0.75 <= AP < 0.9
```

Run:

```bash
python scripts/scan_gmm_band.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --min-ap 0.75 \
  --max-ap 0.9 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_scan_ap075_090.csv"
```

Observed AP `0.75-0.9` scan:

```text
rows: 494
corr(AP, GMM): 0.8919
gmm_ap > 0.95: 5
gmm_ap > 0.9: 30
gmm_ap > 0.85: 151
gmm_ap > 0.8: 283
```

Top GMM unit in the band:

```text
AP 0.897945 -> GMM 0.974432
model.layers.16.mlp.gate_proj:0
unit 4039
```

Largest lift unit:

```text
AP 0.751786 -> GMM 0.913057
model.layers.0.mlp.gate_proj:0
unit 2691
```

Other bands:

```text
AP 0.9-1.0:
  rows: 102
  all gmm_ap > 0.9
  36 units with gmm_ap > 0.95
  corr(AP, GMM): 0.787

AP 0.5-0.75 sampled 1000:
  no units with gmm_ap > 0.9
  one unit with gmm_ap > 0.8
```

Interpretation:

- `0.75-0.9` is currently the best search band.
- Below `0.75`, GMM has not yet found many strong units.
- Above `0.9`, AP is already strong.

## Step 5: Inspect Whether Modes Mean Anything

For a compact terminal summary of unit `4039`:

```bash
python scripts/characterize_gmm_unit_modes.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --concept-json assets/football/sense/football-1_04_00__.json \
  --concept "$CONCEPT" \
  --layer "model.layers.16.mlp.gate_proj:0" \
  --unit 4039
```

Observed summary for unit `4039`:

```text
layer: model.layers.16.mlp.gate_proj:0
unit: 4039
response_mean: 1.3971469482238856
response_std: 1.9901781979398387
pos_k: 2
neg_k: 3
```

Positive mode 0:

```text
count: 507
mean_activation: 2.82284
top terms: football, australian, rules, rugby, league, association, club,
played, game, ball, union, team, national, australia, sport, soccer
distinctive terms included: governing, fifa, official, popular, women,
states, body, leagues
```

Positive mode 1:

```text
count: 193
mean_activation: 4.87653
top terms: football, league, club, played, australian, rules, rugby, game,
team, ball, association, american, match, season, stadium, career, melbourne
distinctive terms included: offside, signed, score, goals, scoring, ncaa,
buffalo, manchester
```

Negative high-activation mode:

```text
count: 55
mean_activation: 3.697
top terms: team, season, championship, player, teams, game, league, final,
record, conference, won
distinctive terms included: teams, championship, ufc, stadium, sevens, rugby,
basketball, ball, league, scoring, playing
```

Interpretation:

- Unit `4039` has real football/sports structure.
- Positive modes look like governance/history/code and match/action/team
  subthemes.
- The high negative mode is a broader sports/event confound.
- This supports the claim that GMM can separate concept-relevant modes from
  confounds that AP compresses into one score.

For unit `2691`, the lift looked less semantically clean:

- Positive modes were football-related.
- Some modes had artifact-like text-pattern terms such as `including`,
  `especially`, `particularly`.
- This is not a good headline example for subconcepts.

Important caution:

> Do not claim every GMM mode is a clean subconcept. Some modes are confounds or
> text artifacts. The defensible claim is that GMM exposes the mode structure so
> we can inspect it instead of pretending the unit is a single scalar concept
> expert.

## Step 6: Summarize Modes In Bulk

Top GMM units:

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

Largest GMM-over-AP lift:

```bash
python scripts/summarize_gmm_scan_modes.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --concept-json assets/football/sense/football-1_04_00__.json \
  --concept "$CONCEPT" \
  --scan-csv "$CONCEPT_DIR/expertise/gmm_scan_ap075_090.csv" \
  --sort-by gmm_minus_ap \
  --top-units 10 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_scan_top10_lift_mode_summary.csv"
```

Keyword evidence summary:

```bash
python scripts/summarize_mode_evidence.py \
  --mode-summary-csv "$CONCEPT_DIR/expertise/gmm_scan_top10_mode_summary.csv"

python scripts/summarize_mode_evidence.py \
  --mode-summary-csv "$CONCEPT_DIR/expertise/gmm_scan_top10_lift_mode_summary.csv"
```

This is a crude terminal-only triage pass. It is useful for deciding what to
inspect manually, but not enough to prove a subconcept claim.

## Step 7: Build GMM Expertise Tables

Compute the GMM table for the useful band:

```bash
python scripts/compute_gmm_expertise.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --min-ap 0.75 \
  --max-ap 0.9 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_expertise_ap075_090.csv"
```

Optional top-AP table:

```bash
python scripts/compute_gmm_expertise.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --min-ap 0.9 \
  --max-ap 1.0 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_expertise_ap090_100.csv"
```

Optional sampled lower band:

```bash
python scripts/compute_gmm_expertise.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --min-ap 0.5 \
  --max-ap 0.75 \
  --max-units 1000 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_expertise_ap050_075_sample1000.csv"
```

Merge AP and GMM for generation:

```bash
python scripts/merge_gmm_expertise.py \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --gmm-csv "$CONCEPT_DIR/expertise/gmm_expertise_ap075_090.csv" \
  --keep gmm \
  --out-csv "$CONCEPT_DIR/expertise/expertise_with_gmm_ap075_090.csv"
```

Observed merged table:

```text
saved: $CONCEPT_DIR/expertise/expertise_with_gmm_ap075_090.csv
rows: 494
columns include:
ap, off_mean, on_p50, on_p90, layer, unit, uuid, concept, group,
gmm_score, gmm_ap, gmm_auc, gmm_minus_ap, diff_mean, pos_k, neg_k,
pos_bic, neg_bic, on_mode_mean, pos_means, neg_means
```

Top merged GMM rows included:

```text
AP 0.897945  GMM 0.974432  model.layers.16.mlp.gate_proj:0  unit 4039
AP 0.858750  GMM 0.955909  model.layers.0.mlp.gate_proj:0   unit 4619
AP 0.887976  GMM 0.955179  model.layers.4.mlp.gate_proj:0   unit 3389
AP 0.894205  GMM 0.953253  model.layers.0.mlp.gate_proj:0   unit 2216
AP 0.896687  GMM 0.952699  model.layers.20.mlp.gate_proj:0  unit 2426
```

## Step 7b: Fit GMM On All Neurons

The mid-AP band was useful for fast iteration, but it still depends on AP as a
prefilter. To select neurons without depending on AP ranking, fit GMMs for every
unit and rank only by `gmm_score`.

The script still reads `expertise.csv` because that file is the convenient unit
inventory and contains `layer`, `unit`, `on_p50`, `on_p90`, and `off_mean`.
However, if you omit `--min-ap` and `--max-ap`, no AP threshold is used for
selection. The final ranking is pure GMM when you sort by `gmm_score`.

First, check how many CPU cores the VM exposes:

```bash
nproc
```

The H100 helps response collection and generation, but GMM fitting is mostly
CPU work. More CPU workers matter here.

### GPU Backend On A100/H100

This branch also has an experimental PyTorch CUDA backend for the 1D GMM fit.
It does not use sklearn's CPU `GaussianMixture`; it fits batches of units on the
GPU and writes the same CSV columns.

First compare CPU and GPU on a small sample:

```bash
python scripts/compute_gmm_expertise.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --max-units 2000 \
  --backend sklearn-cpu \
  --cpus "$(nproc)" \
  --chunksize 128 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_expertise_cpu_sample2000.csv"

python scripts/compute_gmm_expertise.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --max-units 2000 \
  --backend torch-gpu \
  --gpu-device cuda \
  --gpu-batch-size 2048 \
  --gpu-max-iter 50 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_expertise_gpu_sample2000.csv"
```

Then compare the top rows:

```bash
python scripts/select_top_gmm_units.py \
  --gmm-csv "$CONCEPT_DIR/expertise/gmm_expertise_gpu_sample2000.csv" \
  --sort-by gmm_score \
  --top-n 30

python scripts/compare_gmm_tables.py \
  --left-csv "$CONCEPT_DIR/expertise/gmm_expertise_cpu_sample2000.csv" \
  --right-csv "$CONCEPT_DIR/expertise/gmm_expertise_gpu_sample2000.csv" \
  --left-name cpu \
  --right-name gpu \
  --top-n 50
```

If the sample looks reasonable, run the full GPU all-neuron fit:

```bash
python scripts/compute_gmm_expertise.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --backend torch-gpu \
  --gpu-device cuda \
  --gpu-batch-size 4096 \
  --gpu-max-iter 50 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_expertise_all_gpu.csv"
```

If CUDA runs out of memory, lower `--gpu-batch-size` to `2048`, `1024`, or
`512`. For a faster exploratory pass, use `--n-init 1`; for serious comparison,
keep the default `--n-init 3`.

After the GPU full run, either use the GPU CSV directly with
`select_top_gmm_units.py`, or copy/rename it to the standard all-GMM filename:

```bash
cp "$CONCEPT_DIR/expertise/gmm_expertise_all_gpu.csv" \
   "$CONCEPT_DIR/expertise/gmm_expertise_all.csv"
```

### CPU Backend

Run a smaller all-layer pilot if you want a time estimate:

```bash
CORES=$(nproc)

python scripts/compute_gmm_expertise.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --max-units 5000 \
  --cpus "$CORES" \
  --chunksize 128 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_expertise_all_sample5000.csv"
```

Then run the full all-neuron GMM fit with all available CPU workers:

```bash
CORES=$(nproc)

python scripts/compute_gmm_expertise.py \
  --responses-dir "$CONCEPT_DIR/responses" \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --concept "$CONCEPT" \
  --cpus "$CORES" \
  --chunksize 128 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_expertise_all.csv"
```

Expected scale for Qwen 0.5B football:

```text
254976 units
```

If one giant process is awkward, split the all-neuron run into shards. This
example runs 8 shards in parallel with 8 workers each, so it uses about 64 CPU
workers total:

Prefer the single-process `--cpus "$CORES"` command first. Parallel shards can
be faster, but each shard process loads the cached responses, so it uses more
RAM.

```bash
mkdir -p "$CONCEPT_DIR/expertise/gmm_shards"

for SHARD in 0 1 2 3 4 5 6 7; do
  python scripts/compute_gmm_expertise.py \
    --responses-dir "$CONCEPT_DIR/responses" \
    --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
    --concept "$CONCEPT" \
    --num-shards 8 \
    --shard-index "$SHARD" \
    --cpus 8 \
    --chunksize 128 \
    --out-csv "$CONCEPT_DIR/expertise/gmm_shards/gmm_expertise_all_shard${SHARD}.csv" \
    > "$CONCEPT_DIR/expertise/gmm_shards/shard${SHARD}.log" 2>&1 &
done

wait
```

Merge the shards:

```bash
python scripts/merge_gmm_expertise_shards.py \
  --shard-glob "$CONCEPT_DIR/expertise/gmm_shards/gmm_expertise_all_shard*.csv" \
  --out-csv "$CONCEPT_DIR/expertise/gmm_expertise_all.csv"
```

If the VM has fewer CPU cores, reduce either the number of shards, `--cpus`, or
both. The product `num_parallel_shards * cpus_per_shard` should be near, but not
wildly above, `nproc`.

For a faster but less stable exploratory pass, add:

```bash
--n-init 1
```

Use the default `--n-init 3` for the table you plan to compare seriously.

After the full table finishes, print the top pure-GMM neurons:

```bash
python scripts/select_top_gmm_units.py \
  --gmm-csv "$CONCEPT_DIR/expertise/gmm_expertise_all.csv" \
  --sort-by gmm_score \
  --top-n 50
```

Save the top pure-GMM neurons:

```bash
python scripts/select_top_gmm_units.py \
  --gmm-csv "$CONCEPT_DIR/expertise/gmm_expertise_all.csv" \
  --sort-by gmm_score \
  --top-n 200 \
  --out-csv "$CONCEPT_DIR/expertise/gmm_top200_all.csv"
```

If you specifically want multimodal positive experts:

```bash
python scripts/select_top_gmm_units.py \
  --gmm-csv "$CONCEPT_DIR/expertise/gmm_expertise_all.csv" \
  --sort-by gmm_score \
  --min-pos-k 2 \
  --top-n 50
```

If you want to focus on units where GMM most disagrees with AP, sort by lift:

```bash
python scripts/select_top_gmm_units.py \
  --gmm-csv "$CONCEPT_DIR/expertise/gmm_expertise_all.csv" \
  --sort-by gmm_minus_ap \
  --top-n 50
```

Merge the all-neuron GMM table for generation:

```bash
python scripts/merge_gmm_expertise.py \
  --expertise-csv "$CONCEPT_DIR/expertise/expertise.csv" \
  --gmm-csv "$CONCEPT_DIR/expertise/gmm_expertise_all.csv" \
  --keep gmm \
  --out-csv "$CONCEPT_DIR/expertise/expertise_with_gmm_all.csv"
```

Then steer from the all-neuron pure-GMM ranking:

```bash
python scripts/generate_seq.py \
  --model-name-or-path "$MODEL_NAME" \
  --expertise "$CONCEPT_DIR/expertise/expertise_with_gmm_all.csv" \
  --length 40 \
  --prompt "The team" \
  --seed 0 10 \
  --temperature 0.8 \
  --top-p 0.9 \
  --metric gmm_score \
  --forcing on_mode_mean \
  --num-units 3 5 10 \
  --only-last-token \
  --device cuda \
  --no-save
```

Important comparisons after this run:

```text
1. Full AP top units:
   expertise.csv, metric ap, forcing on_p50

2. Full GMM top units:
   expertise_with_gmm_all.csv, metric gmm_score, forcing on_mode_mean

3. GMM-lift units:
   gmm_expertise_all.csv sorted by gmm_minus_ap, then inspect/steer selected units
```

Do not judge the all-neuron result only by whether top GMM beats top AP. Also
check whether the top GMM list contains units with lower AP but meaningful modes
or better steering at the same number of units.

## Step 8: Verify Generation Is Sane Before Steering

This is mandatory. If unforced generation is broken, steering results are not
interpretable.

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
  --num-units 0 \
  --device cuda \
  --no-save
```

Observed after the modern generation fix:

```text
Unforced generation became coherent.
Perplexity was low, roughly 7-16.
Example outputs discussed project teams, skill sets, international partners,
wildlife action plans, and one football-ish European Football Championship
continuation.
```

Interpretation:

- The model and generation loop are now sane.
- Earlier garbled steering outputs were produced before the generation-loop
  fix and should not be used.

## Step 9: Replacement Steering

Use the original replacement-style hook unless deliberately testing additive:

```text
--intervention-mode set
```

This is the default, so it can be omitted.

GMM mid-AP steering:

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
  --num-units 3 \
  --only-last-token \
  --device cuda \
  --no-save
```

Observed GMM mid-AP replacement steering:

```text
seed 0: basketball league / sports table
seed 1: generic team / scholarship
seed 2: plays, passes, touchdown passes
seed 3: Premier League, club
seed 4: offensive attack
```

Interpretation:

- Coherent.
- Pushes toward sports/team context.
- Not always football-specific.
- Strong enough to be useful for comparison.

Full AP top-3 steering:

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
  --num-units 3 \
  --only-last-token \
  --device cuda \
  --no-save
```

Observed full AP top-3 replacement steering:

```text
seed 0: FIFA World Cup
seed 1: volleyball and basketball
seed 2: score / England / cornering tactics
seed 3: National Soccer League, Raiders, 49ers
seed 4: first World Cup in 1930
```

Interpretation:

- Full AP top units are very strong.
- GMM is not currently beating full AP top units.
- This is not a failure; it tells us the right claim is more specific.

Fair within-band AP comparison:

```bash
python scripts/generate_seq.py \
  --model-name-or-path "$MODEL_NAME" \
  --expertise "$CONCEPT_DIR/expertise/expertise_with_gmm_ap075_090.csv" \
  --length 40 \
  --prompt "The team" \
  --seed 0 5 \
  --temperature 0.8 \
  --top-p 0.9 \
  --metric ap \
  --forcing on_p50 \
  --num-units 3 \
  --only-last-token \
  --device cuda \
  --no-save
```

Observed AP within-band replacement steering:

```text
seed 0: hospital / friend in need
seed 1: scientists / ocean / climate change
seed 2: Panthers, Eagles, quarterback
seed 3: factory quality inspection
seed 4: school / National Lottery project
```

Interpretation:

- AP within the same 494-row band was mostly generic team text.
- GMM within the same band produced more sports/football context.
- This is the cleanest steering evidence so far.

Current defensible steering claim:

> Full AP top units are strong. But among mid-AP candidate units, GMM selects
> units that steer more toward the football/sports concept than AP selects from
> the same candidate pool.

## Tomorrow's Experiment Queue

Run these first. They extend the clean within-band comparison.

GMM, within AP `0.75-0.9`, more units and more seeds:

```bash
python scripts/generate_seq.py \
  --model-name-or-path "$MODEL_NAME" \
  --expertise "$CONCEPT_DIR/expertise/expertise_with_gmm_ap075_090.csv" \
  --length 40 \
  --prompt "The team" \
  --seed 0 10 \
  --temperature 0.8 \
  --top-p 0.9 \
  --metric gmm_score \
  --forcing on_mode_mean \
  --num-units 5 10 \
  --only-last-token \
  --device cuda \
  --no-save
```

AP, same 494-row candidate pool:

```bash
python scripts/generate_seq.py \
  --model-name-or-path "$MODEL_NAME" \
  --expertise "$CONCEPT_DIR/expertise/expertise_with_gmm_ap075_090.csv" \
  --length 40 \
  --prompt "The team" \
  --seed 0 10 \
  --temperature 0.8 \
  --top-p 0.9 \
  --metric ap \
  --forcing on_p50 \
  --num-units 5 10 \
  --only-last-token \
  --device cuda \
  --no-save
```

Then repeat with a more football-loaded prompt:

```bash
python scripts/generate_seq.py \
  --model-name-or-path "$MODEL_NAME" \
  --expertise "$CONCEPT_DIR/expertise/expertise_with_gmm_ap075_090.csv" \
  --length 40 \
  --prompt "The football team" \
  --seed 0 10 \
  --temperature 0.8 \
  --top-p 0.9 \
  --metric gmm_score \
  --forcing on_mode_mean \
  --num-units 3 5 10 \
  --only-last-token \
  --device cuda \
  --no-save
```

AP with the same prompt and same band:

```bash
python scripts/generate_seq.py \
  --model-name-or-path "$MODEL_NAME" \
  --expertise "$CONCEPT_DIR/expertise/expertise_with_gmm_ap075_090.csv" \
  --length 40 \
  --prompt "The football team" \
  --seed 0 10 \
  --temperature 0.8 \
  --top-p 0.9 \
  --metric ap \
  --forcing on_p50 \
  --num-units 3 5 10 \
  --only-last-token \
  --device cuda \
  --no-save
```

Optional full-AP reference:

```bash
python scripts/generate_seq.py \
  --model-name-or-path "$MODEL_NAME" \
  --expertise "$CONCEPT_DIR/expertise/expertise.csv" \
  --length 40 \
  --prompt "The football team" \
  --seed 0 10 \
  --temperature 0.8 \
  --top-p 0.9 \
  --metric ap \
  --forcing on_p50 \
  --num-units 3 5 10 \
  --only-last-token \
  --device cuda \
  --no-save
```

## How To Judge Tomorrow's Outputs

Do not rely only on vibes. For each block, count outputs in these buckets:

```text
football/soccer-specific:
  football, soccer, FIFA, World Cup, Premier League, club, goal, stadium,
  match, league, team in explicit football context

broader sports:
  basketball, volleyball, Panthers, Eagles, Raiders, 49ers, quarterback,
  tournament, championship, plays, passes, touchdowns

generic team:
  scientists, managers, school teams, work teams, project teams

artifact/math/problem:
  word problem, multiple choice answer, odd formula-like continuation

garbled:
  broken text or very high perplexity with nonsense
```

The comparison should be reported as:

```text
method | expertise table | forcing target | num_units | prompt | seeds |
football/soccer | broader sports | generic team | artifact | garbled |
mean perplexity
```

The strongest possible next result would be:

```text
GMM within-band has more football/sports outputs than AP within-band at the
same num_units, prompt, seeds, decoding settings, and model.
```

## Optional Terminal-Friendly CSV Viewing

If you cannot open CSV files on the VM, use:

```bash
python - <<'PY'
import pandas as pd

csv = "/tmp/selfcond-qwen05-football/Qwen/Qwen2.5-0.5B/sense/football-1_04_00__/expertise/expertise_with_gmm_ap075_090.csv"
df = pd.read_csv(csv)
cols = [
    "ap", "gmm_score", "gmm_minus_ap", "on_p50", "on_mode_mean",
    "pos_k", "neg_k", "layer", "unit",
]
print(df.sort_values("gmm_score", ascending=False)[cols].head(30).to_string(index=False))
PY
```

Inspect biggest GMM-over-AP lift:

```bash
python - <<'PY'
import pandas as pd

csv = "/tmp/selfcond-qwen05-football/Qwen/Qwen2.5-0.5B/sense/football-1_04_00__/expertise/expertise_with_gmm_ap075_090.csv"
df = pd.read_csv(csv)
cols = [
    "ap", "gmm_score", "gmm_minus_ap", "diff_mean", "on_p50",
    "on_mode_mean", "pos_k", "neg_k", "layer", "unit",
]
print(df.sort_values("gmm_minus_ap", ascending=False)[cols].head(30).to_string(index=False))
PY
```

## Known Caveats

- GMM fitting uses labels because it is class-conditional: one GMM is fit to
  positive responses and one to negative responses. This is not unsupervised
  discovery of concepts from unlabeled text.
- The differentiator from AP is not "we avoid labels." The differentiator is
  that AP treats each unit as a single scalar ranking signal, while GMM models
  multiple positive/negative activation modes.
- GMM and AP are highly correlated overall.
- Full top-AP units are strong.
- The best current GMM claim is about mid-AP units and mode interpretability.
- Positive modes can be real subthemes, but negative modes and artifacts matter.
- `gate_proj` interventions can be strong. Use `--only-last-token`, compare
  against AP, and watch perplexity.
- Additive steering is available but is not the recommended path right now.

## Possible Next Code Additions

These are not implemented yet:

- A generation sweep script that runs AP/GMM, prompts, seeds, and `num_units`
  in one command.
- A scoring script for generated outputs using keyword buckets.
- A held-out evaluation split for GMM scoring/steering decisions.
- Multivariate GMM over groups of expert neurons.

The next high-value code addition is probably a terminal-friendly generation
scorer so tomorrow's outputs become a table instead of a long manual read.

## Research Story So Far

A careful framing:

1. AP is a strong baseline and must remain in every comparison.
2. GMM uses labels, like AP, but gives a richer class-conditional density model.
3. The positive/negative GMM modes reveal structure that AP hides.
4. Unit `4039` is the strongest current interpretability example: football
   positive modes plus a broader sports-confound negative mode.
5. In generation, full AP top units are strongest overall.
6. In the mid-AP band, GMM picks better steering units than AP from the same
   candidate pool.
7. The next experiment should test whether that result survives more seeds,
   more `num_units`, and more prompts.
