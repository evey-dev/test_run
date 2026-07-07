# Mechanistic Interpretability Project - Implementation Plan

## Project Overview
Reproducibility study of mechanistic interpretability circuits in Qwen3-4B-Instruct based on "On the Biology of a Large Language Model" (Anthropic). The goal is to select 1-2 behaviors, trace the computational mechanisms through SAE-discovered features, and validate findings with interventions.

---
<!-- UPDATE NOTE: generate_datasets.py has been added and can be used as an import to get the 3 df. You may change it if needed -->

<!-- When implementing, try to keep things to compactly runnable files as opposed to very separated -->
## Phase 1: Project Setup & Infrastructure (Foundation)

### 1.1 Environment Configuration
- **Status**: Partially Done (environment.yml exists)
- **Tasks**:
  - [ ] Verify all dependencies are correctly specified in environment.yml
  - [ ] Document Python version and key package versions (PyTorch, transformers, etc.)
  - [ ] Create setup.py entry points for easy module imports
  - [ ] Document system requirements (GPU/CPU, memory, disk space)
  - [ ] Create reproducible virtual environment setup instructions

### 1.2 Project Structure & Module Organization
- **Status**: Partially Started (src/ and configs/ exist)
- **Tasks**:
  - [ ] Create utils/ directory for shared utilities
  - [ ] Implement model_loader.py - centralized model/tokenizer loading with proper error handling
  - [ ] Implement config_utils.py - YAML config loading and merging
  - [ ] Implement data_utils.py - activation loading, saving, and preprocessing
  - [ ] Implement prompt_utils.py - prompt loading, formatting, and expected token extraction
  - [ ] Create logging configuration with structured output
  - [ ] Add __init__.py files for proper package structure

### 1.3 Configuration Management
- **Status**: Started (configs/baseline_config.yaml, model_config.yaml exist)
- **Tasks**:
  - [ ] Create sae_config.yaml - SAE training hyperparameters (d_latent, lr, epochs, l1_lambda)
  - [ ] Create intervention_config.yaml - intervention testing parameters
  - [ ] Create unified config schema/validation system
  - [ ] Document all config parameters with explanations

---

## Phase 2: Baseline Establishment (Capital Cities Behavior)

### 2.1 Model Loading & Verification
- **Status**: Partially Done (test.py loads model)
- **Tasks**:
  - [ ] Complete model_loader.py with proper error handling and device management
  - [ ] Verify model loaded correctly (layer count, hidden size, device placement)
  - [ ] Document model architecture (Qwen3-4B-Instruct specifications)
  - [ ] Test forward pass and token generation
  - [ ] Implement model caching for efficient repeated loading

### 2.2 Baseline Prompt Preparation (Capital Cities)
- **Status**: Partially Done (capital_data.csv exists, test.py processes it)
- **Tasks**:
  - [ ] Verify capital_data.csv has all required columns (country/state, capital, etc.)
  - [ ] Create prompt_utils.py functions for:
    - Loading prompts from CSV
    - Formatting prompts consistently
    - Extracting expected tokens
  - [ ] Test prompt formatting on sample data
  - [ ] Document prompt template and variations
  - [ ] Verify ~100-200 prompts for robust baseline

### 2.3 Baseline Output Collection
- **Status**: Partially Done (test.py collects logits)
- **Tasks**:
  - [ ] Complete baseline collection pipeline in test.py:
    - Process all prompts in batches
    - Store: input tokens, output logits, top-k predictions, correct token rank/prob
  - [ ] Save baseline results to outputs/baselines/capital_baseline.json
  - [ ] Record baseline metrics:
    - Accuracy (top-1, top-5, top-10)
    - Mean rank of correct token
    - Mean probability of correct token
  - [ ] Set random seeds (787) for reproducibility
  - [ ] Verify baseline is deterministic (same results on re-run)

### 2.4 Baseline Metrics & Analysis
- **Status**: Partially Done (score.py has partial implementation)
- **Tasks**:
  - [ ] Complete score.py with comprehensive metrics:
    - Top-1 exact match
    - Top-1 prefix match
    - Top-10 recall
    - Probability of correct token
  - [ ] Generate baseline report with figures:
    - Distribution of correct token ranks
    - Distribution of correct token probabilities
    - Accuracy breakdown (e.g., by country/continent)
  - [ ] Save metrics to outputs/baselines/capital_metrics.json

---

## Phase 3: Activation Capture & Processing

### 3.1 Activation Capture Infrastructure
- **Status**: Not Started (activations already collected in mechanistic_data/)
- **Tasks**:
  - [ ] Create activation_capture.py module with:
    - Hook registration system for capturing layer outputs
    - Batch processing of prompts
    - Memory-efficient storage of large activation tensors
    - Verification of activation shapes and dtypes
  - [ ] Document which layers are used (8, 16, 24, 32)
  - [ ] Verify activation dimensions: [num_samples, seq_len, hidden_size=2560]
  - [ ] Test on subset of data before full run

### 3.2 Activation Data Management
- **Status**: Partially Done (activations exist in mechanistic_data/)
- **Tasks**:
  - [ ] Load existing activations from mechanistic_data/activations_layer*.npy
  - [ ] Verify dimensions and data types
  - [ ] Create activation_metadata.csv with:
    - Prompt ID
    - Layer index
    - Activation shape
    - Data type
    - Collection date/seed
  - [ ] Implement activation normalization (z-score or similar if needed)
  - [ ] Create data splits:
    - Training set for SAE training
    - Validation set for SAE validation
    - Test set for intervention experiments
  - [ ] Load indices from train_indices.npy, val_indices.npy

---

## Phase 4: Sparse Autoencoder (SAE) Training

### 4.1 SAE Architecture Implementation
- **Status**: Partially Done (SparseAutoencoder class exists in train.py)
- **Tasks**:
  - [ ] Review and finalize SparseAutoencoder implementation:
    - Encoder: linear layer with ReLU activation
    - Decoder: linear layer with no bias
  - [ ] Implement architecture variants if needed
  - [ ] Add proper initialization (e.g., orthogonal for decoder)
  - [ ] Document design choices and rationale

### 4.2 SAE Training Pipeline
- **Status**: Partially Done (train loop exists but cut off)
- **Tasks**:
  - [ ] Complete train.py with:
    - Full training loop with proper logging
    - Validation loop with early stopping
    - Loss tracking and visualization
    - Checkpoint saving (best model by validation loss)
  - [ ] Implement loss function:
    - MSE reconstruction loss
    - L1 sparsity penalty (weight lambda=1e-3)
  - [ ] Set hyperparameters (can be tuned):
    - latent_dim = 8192 (bottleneck size)
    - batch_size = 64
    - epochs = 50
    - lr = 1e-3
    - l1_lambda = 1e-3
  - [ ] Train SAE for each layer independently: [8, 16, 24, 32]
  - [ ] Save trained models: mechanistic_data/sae_checkpoints/sae_layer*.pt

### 4.3 SAE Evaluation & Feature Interpretation
- **Status**: Not Started
- **Tasks**:
  - [ ] Create sae_analysis.py with:
    - SAE reconstruction accuracy metrics
    - Feature sparsity statistics
    - Dead neuron analysis
  - [ ] Load trained SAEs and compute statistics
  - [ ] Identify top latent features by activation magnitude
  - [ ] Map features to interpretable concepts:
    - Token correlations (which tokens activate each feature)
    - Behavioral correlations (when features are active)
  - [ ] Create feature importance rankings
  - [ ] Generate visualizations:
    - Feature activation distributions
    - Sparsity histograms
    - Reconstruction loss curves

### 4.4 SAE Checkpointing
- **Status**: Started (sae_checkpoints/ directory exists)
- **Tasks**:
  - [ ] Verify or create SAE checkpoint directory structure
  - [ ] Save trained models: mechanistic_data/sae_checkpoints/sae_layer8.pt, etc.
  - [ ] Create metadata file for each SAE:
    - Training date, seed, hyperparameters
    - Validation loss, sparsity achieved
    - Layer index, hidden_size, latent_dim
  - [ ] Implement checkpoint loading with error handling

---

## Phase 5: Attribution Graph Construction

### 5.1 Feature Dependency Analysis
- **Status**: Not Started
- **Tasks**:
  - [ ] Create attribution_graph.py module with:
    - Computation of activation correlations between layers
    - Identification of important features in each layer
    - Feature-to-feature dependency scoring
  - [ ] Define "importance" metric (e.g., variance, correlation with behavior)
  - [ ] Create layer-wise feature importance rankings

### 5.2 Path Tracing from Input to Output
- **Status**: Not Started
- **Tasks**:
  - [ ] Implement input → layer8 → layer16 → layer24 → layer32 → logits path tracing
  - [ ] For capital cities task:
    - Identify input tokens relevant to the behavior (e.g., "capital", country name)
    - Track which SAE features activate for these inputs
    - Trace feature activations through layers
    - Identify features that influence correct token logits
  - [ ] Create visualization of dependency paths

### 5.3 Pruned Dependency Graph
- **Status**: Not Started
- **Tasks**:
  - [ ] Create pruned graph with:
    - Only high-importance features (threshold-based filtering)
    - Only significant connections (correlation/influence threshold)
  - [ ] Implement graph reduction techniques:
    - Keep top-K features per layer
    - Keep top-K connections per feature
  - [ ] Generate graph visualizations
  - [ ] Export to formats suitable for figures (e.g., NetworkX, GraphML)

### 5.4 Output Logit Attribution
- **Status**: Not Started
- **Tasks**:
  - [ ] Identify which final logits are relevant:
    - Top-1 predicted token
    - Correct capital city token
  - [ ] Trace back from logits to decoder inputs
  - [ ] Identify SAE features that most strongly influence logits
  - [ ] Compute attribution scores for each layer

---

## Phase 6: Intervention Testing & Validation

### 6.1 Inhibition Interventions
- **Status**: Not Started
- **Tasks**:
  - [ ] Create intervention.py module with:
    - Hook insertion system for modifying activations
    - Ablation (zero-out) interventions
  - [ ] Implement inhibition testing:
    - For each identified important feature/layer combination
    - Zero out activations, observe effect on output
    - Measure: change in correct token probability, rank
    - Record results with effect sizes
  - [ ] Run on validation/test set
  - [ ] Analyze: which features are most critical?

### 6.2 Activation Swap-In Interventions
- **Status**: Not Started
- **Tasks**:
  - [ ] Implement swap-in interventions:
    - Collect activations for correct vs incorrect capital outputs
    - Swap activations from correct sequence into incorrect sequence
    - Measure: does it flip the output?
  - [ ] Conduct source-target swap experiments:
    - Source: prompts where model is correct
    - Target: prompts where model is incorrect
    - Swap layer-by-layer to identify critical layers
  - [ ] Quantify effects

### 6.3 Intervention Results Analysis
- **Status**: Not Started
- **Tasks**:
  - [ ] Aggregate intervention results:
    - Mean effect of each intervention
    - Consistency across prompts
    - Layer-wise importance ranking
  - [ ] Create summary tables and figures:
    - Ranking of important features
    - Ranking of important layers
    - Comparison to original publication's findings
  - [ ] Save results: outputs/interventions/

---

## Phase 7: Second Behavior (Extension)

### 7.1 Select Second Behavior
- **Status**: Not Started (configs mention "two_hop_reasoning" and "addition")
- **Tasks**:
  - [ ] Choose second behavior from project description or literature:
    - Options: two-hop reasoning, arithmetic, numeracy, physics (units conversion)
    - Select based on relevance and data availability
  - [ ] Document selection rationale

### 7.2 Data Preparation for Second Behavior
- **Status**: Not Started
- **Tasks**:
  - [ ] Create prompt set for second behavior
  - [ ] Run baseline collection (similar to Phase 2.3)
  - [ ] Collect activations for second behavior (similar to Phase 3)
  - [ ] If using same layers (8, 16, 24, 32): reuse SAEs
  - [ ] If using different layers: train new SAEs

### 7.3 Repeat Phases 5-6
- **Status**: Not Started
- **Tasks**:
  - [ ] Run full attribution graph construction for second behavior
  - [ ] Run intervention testing for second behavior
  - [ ] Compare mechanistic patterns across behaviors

---

## Phase 8: Reproducibility Pack & Documentation

### 8.1 Code Organization & Cleanup
- **Status**: In Progress
- **Tasks**:
  - [ ] Review all code for quality:
    - Consistent naming conventions
    - Proper error handling
    - Docstrings for all functions
    - Type hints where possible
  - [ ] Remove debug code and commented sections
  - [ ] Add comprehensive comments for complex logic
  - [ ] Organize imports (alphabetical, group by category)

### 8.2 Reproducibility Infrastructure
- **Status**: Not Started
- **Tasks**:
  - [ ] Create README.md with:
    - Project overview
    - Quick start guide
    - Detailed installation instructions
    - Usage examples for each script
  - [ ] Create requirements.txt / environment.yml with exact versions
  - [ ] Document all random seeds used (seed=787)
  - [ ] Create script to verify reproducibility:
    - Run subset of experiments
    - Compare outputs to reference results
  - [ ] Document data access and preparation

### 8.3 Experimental Tracking
- **Status**: Not Started
- **Tasks**:
  - [ ] Create experiments/ directory with:
    - Config files for each run
    - Output directories with timestamps
    - Metadata about each run
  - [ ] Implement logging system:
    - All experiments logged with date/time
    - All configs saved with results
    - All random seeds recorded
  - [ ] Create summary table of all experiments

### 8.4 Visualization & Figure Generation
- **Status**: Not Started
- **Tasks**:
  - [ ] Create figures/ directory
  - [ ] Generate all figures for report:
    - Baseline accuracy metrics
    - SAE feature visualizations
    - Attribution graphs
    - Intervention results
  - [ ] Create visualization scripts that can be re-run
  - [ ] Ensure figures are publication-quality (proper labels, legends, etc.)

### 8.5 Data Release
- **Status**: Not Started
- **Tasks**:
  - [ ] Organize outputs/:
    - baselines/ - baseline results
    - activations/ - captured activations (or pointers to them)
    - saes/ - trained SAE models
    - attributions/ - attribution graphs
    - interventions/ - intervention results
  - [ ] Create data documentation:
    - Schema for each output file
    - Definitions of all metrics
    - Units and ranges
  - [ ] Create example analysis notebook showing how to use outputs

---

## Phase 9: Report Writing

### 9.1 Executive Summary (≤1000 words)
- **Status**: Not Started
- **Tasks**:
  - [ ] Write executive summary covering:
    - Problem statement and motivation
    - Methodology overview
    - Key findings (1-2 findings)
    - Comparison to original work
    - Conclusions

### 9.2 Main Report (≤7000 words)
- **Status**: Not Started
- **Tasks**:
  - [ ] Structure report:
    - Introduction (background, problem, motivation)
    - Literature review (original paper + related work)
    - Methodology (baseline, SAE training, attribution, interventions)
    - Experiments & Results (one section per major phase)
    - Analysis & Discussion (comparison to original, limitations, insights)
    - Conclusion & Future work
  - [ ] Include figures and tables
  - [ ] Write clearly and concisely
  - [ ] Include code snippets or appendices for key implementations

### 9.3 Declaration of AI Use
- **Status**: Not Started
- **Tasks**:
  - [ ] Document any use of generative AI (ChatGPT, Copilot, etc.)
  - [ ] Specify what tasks AI was used for
  - [ ] Include in report with word count

---

## Phase 10: Final Testing & Submission

### 10.1 End-to-End Testing
- **Status**: Not Started
- **Tasks**:
  - [ ] Run all scripts from scratch on clean environment
  - [ ] Verify reproducibility on different machines/GPUs if possible
  - [ ] Test all error handling paths
  - [ ] Document any environment-specific issues

### 10.2 Code Quality Assurance
- **Status**: Not Started
- **Tasks**:
  - [ ] Run linting (pylint, flake8)
  - [ ] Check for type errors with mypy (if type hints added)
  - [ ] Verify no hardcoded paths (use relative paths or config)
  - [ ] Check for sensitive info (API keys, passwords) - should be none
  - [ ] Test all major functions with unit tests

### 10.3 Documentation Completeness
- **Status**: Not Started
- **Tasks**:
  - [ ] Verify all functions/modules have docstrings
  - [ ] Verify all config parameters are documented
  - [ ] Verify all output files are documented
  - [ ] Verify README has all necessary information

### 10.4 Repository Organization
- **Status**: Not Started
- **Tasks**:
  - [ ] Create report/ directory and place reports there
  - [ ] Create scripts/ directory for all runnable scripts
  - [ ] Verify .gitignore excludes large files (models, large data)
  - [ ] Verify all code is committed (git status clean)
  - [ ] Create git tags for milestones

### 10.5 Final Submission
- **Status**: Not Started
- **Tasks**:
  - [ ] Verify submission deadline: July 1, 23:59
  - [ ] Final git push to repository
  - [ ] Verify all required files present:
    - Code (src/, scripts/)
    - Report (report/project_report.pdf, report/executive_summary.pdf)
    - AI declaration with word count
  - [ ] Double-check word counts
  - [ ] Submit

---

## Dependencies & Resources

### Key Python Libraries
- torch, transformers - model loading and inference
- numpy - numerical computing
- pandas - data handling
- matplotlib/seaborn - visualization
- PyYAML - configuration

### Required Data
- Models: Qwen3-4B-Instruct (already downloaded to ./models/)
- Original model checkpoint (in mechanistic_data/original_model/)
- Capital cities dataset (capital_data.csv - already exists)
- Pre-collected activations (mechanistic_data/activations_*.npy)
- Pre-trained SAE checkpoints (mechanistic_data/sae_checkpoints/)

### Hardware Requirements
- GPU: 4B model fits on modern GPU (8GB VRAM sufficient)
- Memory: ~16GB RAM recommended for batch processing
- Disk: ~50GB for models + data + outputs

---

## Timeline & Priorities

### Critical Path (Must Do)
1. Phase 1 - Setup (1-2 days)
2. Phase 2 - Baseline (2-3 days)
3. Phase 4 - SAE Training (3-5 days) - can reuse existing SAEs if available
4. Phase 5 - Attribution (3-4 days)
5. Phase 6 - Interventions (3-4 days)
6. Phase 9 - Report Writing (5-7 days)

### Extension (Nice to Have)
- Phase 7 - Second Behavior (3-5 days)
- Phase 10 - Testing & Polish (2-3 days)

### Deadline
**July 1, 2025** - Submit repository with code, report, and executive summary

---

## Success Criteria

1. ✓ Baseline working and reproducible
2. ✓ SAEs trained and generating meaningful features
3. ✓ Attribution graphs show interpretable paths
4. ✓ Interventions demonstrate causal effects
5. ✓ Findings match or extend the original paper
6. ✓ Code is well-documented and reproducible
7. ✓ Report clearly explains methodology and findings
8. ✓ All submission requirements met

