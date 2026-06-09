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
