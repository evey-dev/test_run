# Mechanistic Interpretability Reproduction Project

This repository is an independent small-scale reproducibility study inspired by
Anthropic's *On the Biology of a Large Language Model*. It asks whether similar
attribution-graph and intervention methods can recover mechanisms in
`Qwen3-4B-Instruct`.

The project currently studies three behaviours:

- physics units: strongest positive result
- arithmetic carry: partial reproduction with useful diagnostics
- capitals/factual recall: weak or negative result

The aim is reproducibility analysis, not exact replication. Partial and negative
results are part of the scientific finding when they are documented carefully.

## Environment

Create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate mphil-project
```

The model path defaults to `models/Qwen3-4B-Instruct` through
`configs/model_config.yaml`. If the local model is absent, the scripts may fall
back to Hugging Face Hub where implemented, but the expected reproducible setup
is to restore or download the model locally.

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct --local-dir models/Qwen3-4B-Instruct
```

## Why Colab Notebooks Exist

The Colab notebooks were used because CSD3 access was unavailable during the
project work. They are convenience wrappers around standalone repository scripts;
the core pipeline can be run directly from the command line.

The notebooks now pull code from GitHub first. Google Drive is still used for
large artifacts such as SAE checkpoints, activation files, graphs, and final JSON
outputs. Drive zip/copy cells are retained as backup paths.

## Recommended Notebooks

Use these for final results:

- `run_gpu_final_validation.ipynb`
  - Short post-hoc validation notebook; run after the three behaviour notebooks.
  - Restores existing checkpoints and graphs without retraining or rebuilding
    attribution graphs.
  - Measures held-out reconstruction fidelity, achieved sparsity and decoder
    norms, then runs fixed graph-held-out arithmetic and unit interventions.
  - Generates report-ready diagnostic, generalisation and attribution-graph
    figures. On a Colab GPU this should take minutes rather than the roughly
    30 minutes required by each attribution-graph cell.

- `run_gpu_units_hypertrain.ipynb`
  - Final physics-units notebook.
  - Trains/restores units-specific SAE checkpoints.
  - Runs contrast attribution graphs and intervention tests.
  - This is the strongest behaviour to foreground in the report.

- `run_gpu_math_final.ipynb`
  - Final arithmetic/carry notebook.
  - Trains/restores math-specific SAE checkpoints.
  - Runs contrast attribution for `58 + 83`, intervention scans, swaps, and
    controls.
  - Use this for the partial-reproduction story: broad MLP interventions move
    probability toward dropped-carry alternatives, but sparse SAE features do not
    isolate a clean carry circuit.

- `run_gpu_capitals_final.ipynb`
  - Final capitals/factual-recall notebook.
  - Trains/restores capitals-specific SAE checkpoints.
  - Includes Dallas/Oakland and higher-confidence Zarqa/Basra controls.
  - Use as a negative or weak reproduction result, not as the headline.

Older notebooks are kept for provenance and debugging:

- `run_gpu.ipynb`: original combined Colab workflow.
- `run_gpu_math.ipynb`: exploratory arithmetic workflow with older settings.
- `run_gpu_units.ipynb`: exploratory units workflow.
- `run.ipynb`: early local notebook.

Do not rename the older notebooks unless there is a specific reason. Keeping
them stable avoids breaking references in Colab, Drive, or old notes.

## Standalone Pipeline

### 1. Generate Prompt Data

```bash
python data/generate_datasets.py --capitals
```

Expected files:

- `data/addition_data.csv`
- `data/units_data.csv`
- `data/capitals_data.csv`

### 2. Capture Activations

```bash
python src/capture_activations.py --output-dir mechanistic_data --seed 787
```

Useful options:

- `--layers 4 8 12 16 20 24 28`
- `--behaviours addition units capitals`
- `--model-config configs/model_config.yaml`

This writes activation arrays, metadata, and train/validation splits under
`mechanistic_data/`.

### 3. Train SAEs

General training command:

```bash
python src/train.py --config configs/sae_config.yaml
```

Final behaviour-specific configs:

```bash
python src/train.py --config configs/sae_units_final_train_config.yaml
python src/train.py --config configs/sae_math_final_train_config.yaml
python src/train.py --config configs/sae_capitals_final_train_config.yaml
```

Training saves one checkpoint per layer plus latent arrays and metadata. The
final notebooks use behaviour-specific checkpoint directories so old checkpoints
are not accidentally mixed into final runs.

### 4. Build Attribution Graphs

Example contrast graph for the arithmetic carry comparison:

```bash
python src/attribution_graph.py \
  --prompt "Question: What is 58 + 83? Answer: 1" \
  --target "4" \
  --contrast-target "3" \
  --layers 4 8 12 16 20 24 28 \
  --sae-config configs/sae_math_final_train_config.yaml \
  --output-json outputs/math_final_carry_58_83_4v3_graph.json \
  --output-html outputs/math_final_carry_58_83_4v3_graph.html \
  --output-mermaid outputs/math_final_carry_58_83_4v3_graph.md
```

Use `--contrast-target` when the scientific question is a choice between two
tokens, such as correct carry digit `4` versus dropped-carry digit `3`.

### 5. Run Interventions

Sparse graph-feature inhibition:

```bash
python src/intervention.py \
  --mode inhibit \
  --prompt "Question: What is 58 + 83? Answer: 1" \
  --target-token "4, 3, 7" \
  --layers 4 8 12 16 20 24 28 \
  --sae-config configs/sae_math_final_train_config.yaml \
  --graph-json outputs/math_final_carry_58_83_4v3_graph.json \
  --graph-feature-sign positive \
  --scan
```

Full all-position MLP knockout diagnostic:

```bash
python src/intervention.py \
  --mode inhibit \
  --prompt "Question: What is 58 + 83? Answer: 1" \
  --target-token "4, 3, 7" \
  --layers 4 8 12 16 20 24 28 \
  --sae-config configs/sae_math_final_train_config.yaml \
  --full-knockout \
  --knockout-component mlp \
  --positions all \
  --layer-scan \
  --print-tokens
```

Full latent swap:

```bash
python src/intervention.py \
  --mode swap \
  --source-prompt "Question: What is 44 + 83? Answer: 1" \
  --prompt "Question: What is 58 + 83? Answer: 1" \
  --target-token "4, 3, 7" \
  --layers 4 8 12 16 20 24 28 \
  --sae-config configs/sae_math_final_train_config.yaml \
  --positions all
```

Important intervention options:

- `--graph-json`: use features selected by an attribution graph.
- `--graph-feature-sign positive|negative|all`: filter graph features by
  attribution sign.
- `--positions last|all|0,1,2`: choose which token positions to edit.
- `--knockout-component mlp|attn|block`: choose the component for full knockout.
- `--layer-scan`: rerun intervention one layer at a time.
- `--print-tokens`: print token positions so position-based edits can be checked.
- Omitting `--features` and `--graph-json` in swap mode performs a full latent
  swap, not a sparse graph-feature swap.

### 6. Final Diagnostics and Held-out Validation

The recommended route is to run `run_gpu_final_validation.ipynb` after the
behaviour notebooks. Its standalone commands are:

```bash
python src/sae_diagnostics.py \
  --config configs/sae_math_final_train_config.yaml \
  --label math \
  --output-json outputs/final_sae_diagnostics_math.json \
  --output-csv outputs/final_sae_diagnostics_math.csv

python src/heldout_validation.py \
  --math-cases 12 \
  --unit-cases 12 \
  --output outputs/final_heldout_validation.json
```

`sae_diagnostics.py` evaluates the original train/validation activation splits;
it does not retrain the SAEs. `heldout_validation.py` reuses the fixed final
graphs, so the benchmark prompts are graph-held-out rather than used to select a
new feature set. The script records clean, sparse, full-latent and raw-MLP
conditions and saves partial results before moving from arithmetic to units.

Generate the report figures after these outputs exist:

```bash
python src/plot_validation.py \
  --diagnostics outputs/final_sae_diagnostics_math.csv \
                outputs/final_sae_diagnostics_units.csv \
                outputs/final_sae_diagnostics_capitals.csv \
  --heldout outputs/final_heldout_validation.json \
  --output-dir outputs/report_figures

python src/plot_attribution_graph.py \
  --graph outputs/math_final_carry_58_83_4v3_graph.json \
  --output-dir outputs/report_figures
```

The static graph figure is a labelled visual subset selected from the complete
JSON. The JSON and interactive HTML remain the authoritative full graph
artifacts.

## Current Scientific Bottom Line

The project should not claim a clean reproduction of Anthropic's sparse circuits.
The strongest story is:

1. Physics units show the best positive intervention behaviour.
2. Arithmetic carry shows partial causal evidence: broad all-position MLP/latent
   interventions can move probability toward dropped-carry alternatives, but a
   compact sparse SAE carry circuit was not recovered.
3. Capitals remain weak even after testing higher-confidence examples, suggesting
   factual recall is harder to patch cleanly with this setup.

This supports a rigorous reproducibility report: the pipeline was implemented,
adapted where necessary, and used to identify where the Anthropic-style method
does and does not transfer under constrained resources.
