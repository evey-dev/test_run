# Mechanistic Interpretability Reproduction Project

This repository is an independent small-scale reproducibility study inspired by
Anthropic's *On the Biology of a Large Language Model*. It asks whether similar
attribution-graph and intervention methods can recover mechanisms in
`Qwen3-4B-Instruct-2507`.

The project currently studies three behaviours:

- physics units: compact, force-associated TopK panel; strongest positive result
- arithmetic carry: initial specificity failures followed by a replicated,
  output-digit-balanced 20-feature effect
- capitals/factual recall: weak or negative reproduction

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
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct
```

## Why Colab Notebooks Exist

The Colab notebooks were used after a cooling incident during hot weather made
CSD3 unavailable during the final project period. They are convenience wrappers
around standalone repository scripts; the core pipeline can be run directly
from the command line. The project does not assume a fixed Colab compute-unit
rate because Google documents these limits and hardware availability as dynamic.

The notebooks now pull code from GitHub first. Google Drive is still used for
large artifacts such as SAE checkpoints, activation files, graphs, and final JSON
outputs. Drive zip/copy cells are retained as backup paths.

## Recommended Notebooks

Use these for the final reported results, in this order where dependencies
apply:

- `run_gpu_math_topk_retrain.ipynb`
  - Trains fixed TopK arithmetic candidates at `k=128,256,512`.
  - Selects a candidate from reconstruction/sparsity diagnostics before any
    candidate intervention result is used.
  - Builds the selected carry graph and runs a matched no-carry benchmark.
  - The completed run selected TopK-256 and found a larger but non-specific
    arithmetic effect.

- `run_gpu_math_carry_feature_followup.ipynb`
  - Restores the selected TopK-256 checkpoints and graph.
  - Runs the final-token discovery/confirmation screen on disjoint arithmetic
    pairs.
  - The frozen top-10 panel failed its predeclared carry-specificity criterion.

- `run_gpu_math_carry_balanced_localization.ipynb`
  - Final output-digit-balanced extension using the already selected TopK-256
    checkpoints.
  - Screens every SAE latent after conditioning on the predicted tens digit,
    then freezes a Top-10 panel before causal confirmation on 32 disjoint pairs.
  - Also evaluates output-digit-conditioned carry decodability in the raw MLP
    output at each selected layer.
  - The Top-10 primary failed. A secondary Top-20 effect is therefore treated
    as hypothesis generation and the final notebook section freezes those exact
    IDs for one independent test on intervention-untouched cases.
  - That one-shot case-level replication passed: paired specificity `-0.0859`,
    bootstrap 95% interval `[-0.1367, -0.0391]`, with no top-token transfer.

- `run_gpu_units_topk_retrain.ipynb`
  - Trains and diagnostically selects fixed units TopK candidates.
  - Builds one force graph, ranks features on eight discovery systems, and tests
    a frozen top-10 panel on 16 different systems against a mass donor control.
  - The completed run selected TopK-128 and produced the report's strongest
    compact causal result.

- `run_gpu_capitals_final.ipynb`
  - Trains/restores capitals-specific ReLU checkpoints.
  - Includes Dallas/Oakland and higher-confidence Zarqa/Basra controls.
  - Use as a negative or weak reproduction, not as the headline.

Supporting validation and original-ReLU provenance:

- `run_gpu_final_validation.ipynb`
  - Post-hoc diagnostics for the original ReLU checkpoints and fixed graphs.
  - Restores existing checkpoints and graphs without retraining or rebuilding
    attribution graphs.
  - Measures held-out reconstruction fidelity, achieved sparsity and decoder
    norms, then runs fixed graph-held-out arithmetic and unit interventions.
  - Generates report-ready diagnostic, generalisation and attribution-graph
    figures. On a Colab GPU this should take minutes rather than the roughly
    30 minutes required by each attribution-graph cell.

- `run_gpu_units_hypertrain.ipynb`
  - Original ReLU physics-units workflow.
  - Trains/restores units-specific SAE checkpoints.
  - Runs contrast attribution graphs and intervention tests.
  - Its broad force-to-energy patch motivated the guarded TopK follow-up.

- `run_gpu_math_final.ipynb`
  - Original ReLU arithmetic/carry workflow.
  - Trains/restores math-specific SAE checkpoints.
  - Runs contrast attribution for `58 + 83`, intervention scans, swaps, and
    controls.
  - Broad MLP interventions move digit state but do not isolate carry.

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
python -m src.capture_activations --output-dir mechanistic_data --seed 787
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
python -m src.train --config configs/sae_config.yaml
```

Final behaviour-specific configs:

```bash
python -m src.train --config configs/sae_units_final_train_config.yaml
python -m src.train --config configs/sae_math_final_train_config.yaml
python -m src.train --config configs/sae_capitals_final_train_config.yaml
```

Training saves one checkpoint per layer plus latent arrays and metadata. The
final notebooks use behaviour-specific checkpoint directories so old checkpoints
are not accidentally mixed into final runs.

### 4. Build Attribution Graphs

Example contrast graph for the arithmetic carry comparison:

```bash
python -m src.attribution_graph \
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
python -m src.intervention \
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
python -m src.intervention \
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
python -m src.intervention \
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
python -m src.sae_diagnostics \
  --config configs/sae_math_final_train_config.yaml \
  --label math \
  --output-json outputs/final_sae_diagnostics_math.json \
  --output-csv outputs/final_sae_diagnostics_math.csv

python -m src.heldout_validation \
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
python -m src.plot_validation \
  --diagnostics outputs/final_sae_diagnostics_math.csv \
                outputs/final_sae_diagnostics_units.csv \
                outputs/final_sae_diagnostics_capitals.csv \
  --heldout outputs/final_heldout_validation.json \
  --output-dir outputs/report_figures

python -m src.plot_attribution_graph \
  --graph outputs/math_final_carry_58_83_4v3_graph.json \
  --output-dir outputs/report_figures
```

The static graph figure is a labelled visual subset selected from the complete
JSON. The JSON and interactive HTML remain the authoritative full graph
artifacts.

### 7. Final TopK Mathematics Workflow

The complete guarded workflow is in `run_gpu_math_topk_retrain.ipynb`. The
candidate configurations are:

```text
configs/sae_math_topk128_config.yaml
configs/sae_math_topk256_config.yaml
configs/sae_math_topk512_config.yaml
```

The shared `SparseAutoencoder` remains backward-compatible with existing ReLU
checkpoints. TopK checkpoints record their activation type and `k` in metadata,
which attribution and intervention loaders use automatically.

Candidate selection is performed by `src/select_sae_candidate.py` without access
to intervention results. A candidate must have mean validation FVE at least
0.90, minimum layer FVE at least 0.85, and mean dead-feature fraction at most
0.80. The sparsest eligible candidate is selected. If none qualifies, the
notebook saves the diagnostics to Drive and stops before graph construction.

The selected candidate is evaluated with:

```bash
python -m src.heldout_validation \
  --math-sae-config configs/sae_math_topk256_config.yaml \
  --math-graph outputs/topk_math_retrain/math_topk256_carry_58_83_4v3_graph.json \
  --math-cases 12 \
  --skip-units \
  --math-specificity-control \
  --output outputs/topk_math_retrain/math_topk256_heldout_specificity.json
```

The `k=256` paths above are illustrative; use the configuration recorded in
`math_topk_selection.json`. The specificity control applies the same positive
carry-graph features to each matched no-carry source. A negative paired
carry-minus-control effect supports carry selectivity; a similar effect in both
conditions instead suggests generic arithmetic-answer support.

The completed selection chose TopK 256. It achieved mean validation FVE 0.941
with mean validation L0 171.6, compared with roughly two thousand active latents
for the original ReLU mathematics SAEs. Sparse inhibition became larger
(`-0.875` versus `-0.396` in correct-minus-dropped-carry logit-gap delta), but
the matched no-carry effect was nearly identical (`-0.885`). The paired
carry-minus-control estimate was `+0.010` with bootstrap 95% interval
`[-0.125, +0.167]`; this does not support carry specificity.

The final-position discovery/confirmation follow-up can be run standalone after
restoring the selected checkpoints and graph:

```bash
python -m src.heldout_validation \
  --math-sae-config configs/sae_math_topk256_config.yaml \
  --math-graph outputs/topk_math_retrain/math_topk256_carry_58_83_4v3_graph.json \
  --math-cases 12 \
  --skip-units \
  --math-specificity-control \
  --positions last \
  --output outputs/topk_math_followup/math_topk256_heldout_specificity_last.json

python -m src.math_carry_feature_screen \
  --sae-config configs/sae_math_topk256_config.yaml \
  --graph outputs/topk_math_retrain/math_topk256_carry_58_83_4v3_graph.json \
  --positions last \
  --discovery-cases 8 \
  --confirmation-cases 24 \
  --output outputs/topk_math_followup/math_topk256_carry_feature_screen.json
```

The screen excludes the 12 already inspected benchmark pairs. Confirmation data
never influence feature ordering. The primary top-10 panel supports carry
selectivity only if its paired mean is negative, its bootstrap 95% interval is
entirely below zero, and its mean carry-target effect is negative. Secondary
panel sizes must not be selected post hoc.

The optional output-digit-balanced localisation is a distinct final test of the
same selected SAE, not another hyperparameter sweep:

```bash
python -m src.math_carry_balanced_localization \
  --sae-config configs/sae_math_topk256_config.yaml \
  --candidate-pairs 149 \
  --discovery-pairs 32 \
  --confirmation-pairs 32 \
  --seed 4787 \
  --panel-sizes 1 3 5 10 20 \
  --primary-panel-size 10 \
  --random-panels 5 \
  --exclude-json \
    outputs/topk_math_retrain/math_topk256_heldout_specificity.json \
    outputs/topk_math_followup/math_topk256_carry_feature_screen.json \
  --output outputs/math_carry_localization/math_topk256_balanced_carry_localization.json \
  --activation-cache outputs/math_carry_localization/math_topk256_balanced_carry_activations.npz
```

Discovery ranks all 57,344 latents by their standardised carry-minus-no-carry
activation difference within shared output-digit strata. Confirmation data do
not affect that ordering. The primary result passes only if the frozen Top-10
retains a positive conditioned activation interval and inhibition has a
carry-minus-control interval wholly below zero. The notebook writes resumable
JSON and activation checkpoints directly to Drive.

The completed primary run found strong held-out activation separation for the
Top-10 panel but no selective causal effect. Its secondary Top-20 panel had a
paired effect of `-0.1016` with a bootstrap 95% interval of
`[-0.1484, -0.0508]`. That result was not the predeclared primary outcome. The
notebook therefore contains exactly one follow-up: an independent replication
of the unchanged Top-20 IDs on 32 eligible pairs that received baselines but no
feature intervention in the first run.

The same replication can be run standalone after the completed source JSON is
available:

```bash
python -m src.math_carry_top20_replication \
  --sae-config configs/sae_math_topk256_config.yaml \
  --source-result outputs/math_carry_localization/math_topk256_balanced_carry_localization.json \
  --replication-pairs 32 \
  --seed 9787 \
  --output outputs/math_carry_localization/math_topk256_balanced_top20_replication.json
```

No reranking occurs in this command. The completed replication changed the
carry-target gap by `-0.0586`, the no-carry control by `+0.0273`, and their
paired difference by `-0.0859` with bootstrap 95% interval
`[-0.1367, -0.0391]`. The frozen criterion therefore passed. This supports a
distributed carry-associated panel, not a monosemantic carry feature, answer
flip, or complete arithmetic circuit. Panel search stops at this point.

Generate the report-ready balanced arithmetic figure without a GPU:

```bash
python -m src.plot_math_carry_replication
```

The preferred inputs are the two completed JSON files in
`outputs/math_carry_localization/`. If they have not been copied from Drive, the
renderer verifies and reads the exact aggregates embedded in
`run_gpu_math_carry_balanced_localization.ipynb`. It writes PDF, PNG and editable
SVG versions under `report/figures/`. The older
`fig_math_compact_carry_test.pdf` remains provenance for the failed graph-derived
Top-10 panel but is no longer the report's final arithmetic figure.

### 8. Final TopK Units Workflow

The complete workflow is `run_gpu_units_topk_retrain.ipynb`. It uses:

```text
configs/sae_units_topk128_config.yaml
configs/sae_units_topk256_config.yaml
configs/sae_units_topk512_config.yaml
```

The selection thresholds and stopping rule are identical to the mathematics
TopK sweep. The selected graph and feature screen always operate at the final
token, matching SAE training. The standalone confirmation command is:

```bash
python -m src.units_feature_screen \
  --sae-config configs/sae_units_topk128_config.yaml \
  --graph outputs/topk_units_retrain/units_topk128_force_graph.json \
  --positions last \
  --discovery-cases 8 \
  --confirmation-cases 16 \
  --output outputs/topk_units_retrain/units_topk128_feature_screen.json
```

The completed diagnostic selection chose `k=128`; always verify this in
`units_topk_selection.json` before running a fresh experiment. Exact
force, mass and energy prompts are absent from the SAE corpus. Discovery uses
eight systems and confirmation uses sixteen different systems, with one prompt
per system, selected from a 64-system baseline pool. An initial baseline-only
pilot used more indirect prompt wording and yielded only fourteen qualified
systems; it was replaced before any feature intervention was run. The revised
prompts place the measured quantity immediately before the requested unit. The
clean energy target and force source must predict the expected
unit prefixes. Mass-control correctness is recorded and preferred among prompt
variants, but is not an eligibility condition because it is a negative-control
source rather than the causal target. The primary Top-10
panel succeeds only when both the force-source effect and its advantage over the
matched mass-source control have bootstrap 95% intervals wholly above zero.
Confirmation panel sizes must not be selected post hoc.

The report-ready Anthropic-style circuit summary is generated entirely from the
completed graph and feature-screen JSON files; it does not load the model, SAE
weights or activation matrices:

```bash
python -m src.plot_units_compact_circuit
```

This writes PNG, PDF and editable SVG versions to
`outputs/topk_units_retrain/figures/` and a PDF copy to
`report/figures/fig_units_compact_causal_circuit.pdf`. The figure combines
retained attribution-graph edges with the separately measured frozen-panel
swap. It must be described as a force-associated causal summary: the
intervention shifted the newtons-minus-joules logit gap but did not change the
top prediction from joules.

The completed frozen top-10 confirmation produced:

```text
force-source gap shift:       +1.2344  (95% CI +1.1484 to +1.3320)
mass-source control shift:    -0.0898  (95% CI -0.1289 to -0.0508)
force-minus-mass specificity: +1.3242  (95% CI +1.2227 to +1.4297)
```

The force effect exceeded the mass effect in all 16 confirmation systems. The
top prediction remained joules, so this is a compact force-associated logit
shift rather than answer transfer.

## Build the Written Deliverables

From `report/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error executive_summary.tex
texcount -sum -merge report.tex
texcount -sum -merge executive_summary.tex
```

The main report must remain at or below 7,000 words. The executive summary is a
separate document and must remain below 1,000 words. Report figures are vector
PDFs under `report/figures/`; their generators and source artifacts are recorded
in `report/FIGURE_PLAN.txt`.

## Current Scientific Bottom Line

The project supports a partial, bounded reproduction:

1. The full requested workflow was implemented independently.
2. A frozen ten-feature units panel produced a force-specific logit shift across
   16 disjoint systems relative to a mass donor control.
3. Initial arithmetic panels failed specificity, but a frozen 20-feature panel
   subsequently produced a smaller carry-selective effect on 32 untouched cases.
4. Neither positive result flipped the answer or establishes a monosemantic
   feature or complete circuit.
5. Capitals remained weak even with high-confidence prompts.

The scientific contribution is the combination of two bounded sparse causal
effects, controlled failed primaries, and explicit evidence about where stronger
mechanistic interpretations are not warranted.
