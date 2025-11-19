# Starter Code and Reference Implementations

Official starter code and reference implementations for the project.

---

## ⭐ Official Starter Code: Greg Durrett's fp-dataset-artifacts

### Overview
- **Repository**: https://github.com/gregdurrett/fp-dataset-artifacts
- **Authors**: Kaj Bostrom, Jifan Chen, Greg Durrett (UT Austin)
- **Purpose**: Final project starter code for studying dataset artifacts in NLP
- **Status**: Official course material, actively maintained

### What It Provides

**Core Files**:
1. `run.py` - Main training/evaluation script
2. `helpers.py` - Preprocessing and metric computation utilities
3. `requirements.txt` - Python dependencies

**Datasets Supported**:
- SNLI (Stanford Natural Language Inference)
- SQuAD (Stanford Question Answering Dataset)

**Model**: ELECTRA-small (same as our project)

### Key Features

#### NLI Support
- Automatic SNLI/MultiNLI dataset loading
- Filters invalid labels (label=-1)
- Proper premise/hypothesis tokenization
- Accuracy metric computation

#### QA Support
- SQuAD dataset integration
- Proper answer span extraction
- F1 and Exact Match metrics
- Custom QuestionAnsweringTrainer class

#### Helper Functions (helpers.py)

1. **`prepare_dataset_nli()`**
   - Tokenizes premises and hypotheses
   - Applies truncation and padding
   - Returns tokenized inputs

2. **`compute_accuracy()`**
   - Computes classification accuracy
   - Compatible with transformers.Trainer

3. **`prepare_train_dataset_qa()`**
   - Tokenizes questions and contexts
   - Handles long contexts via chunking
   - Maps answer spans to token positions

4. **`prepare_validation_dataset_qa()`**
   - Prepares QA validation data
   - Preserves offset mappings for evaluation
   - Maintains example IDs

5. **`postprocess_qa_predictions()`**
   - Converts logits to actual answer text
   - Extracts highest-scoring spans
   - Handles edge cases

6. **`QuestionAnsweringTrainer`** (Custom class)
   - Extends transformers.Trainer
   - Specialized evaluate() method
   - Computes F1/EM metrics

### Usage Examples

**Train NLI Model**:
```bash
python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/
```

**Evaluate NLI Model**:
```bash
python3 run.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./eval_output/
```

**Train QA Model**:
```bash
python3 run.py --do_train --task qa --dataset squad --output_dir ./trained_model/
```

### Expected Performance

| Task | Dataset | Metric | Expected Score |
|------|---------|--------|----------------|
| NLI | SNLI | Accuracy | ~89% |
| QA | SQuAD | Exact Match | ~78% |
| QA | SQuAD | F1 Score | ~86% |

### Command-Line Arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `--model` | google/electra-small-discriminator | Base model |
| `--task` | Required | "nli" or "qa" |
| `--dataset` | Auto | Override default dataset |
| `--max_length` | 128 | Max sequence length |
| `--max_train_samples` | None | Limit training examples |
| `--max_eval_samples` | None | Limit eval examples |
| `--do_train` | False | Enable training |
| `--do_eval` | False | Enable evaluation |
| `--output_dir` | Required | Output directory |
| `--no_cuda` | False | Force CPU mode |

---

## 🔧 How This Project Extends the Starter Code

Our implementation builds on the starter code with additional capabilities:

### What We Added

1. **Analysis Module** (`analysis/`)
   - `error_analysis.py` - Systematic error pattern identification
   - `contrast_sets.py` - Robustness evaluation
   - `visualization.py` - Plotting utilities
   - Hypothesis-only baseline analysis
   - Lexical overlap analysis

2. **Mitigation Module** (`mitigation/`)
   - `dataset_cartography.py` - Training dynamics analysis
   - `debiasing.py` - Ensemble debiasing methods
   - `adversarial_training.py` - Data augmentation
   - Product of Experts implementation
   - Hard example identification

3. **Jupyter Notebooks** (`notebooks/`)
   - `01_baseline_training.ipynb` - Interactive training
   - `02_artifact_analysis.ipynb` - Artifact detection
   - `03_mitigation_eval.ipynb` - Debiasing evaluation

4. **Research Resources** (`resources/`)
   - Comprehensive paper collection (40+ papers)
   - GitHub repository list (17 repos)
   - Benchmark documentation (12+ benchmarks)
   - Key researcher profiles
   - SNLI-specific artifact documentation

5. **Enhanced Features**
   - Extended preprocessing for analysis
   - Confidence distribution analysis
   - Concept erasure methods (INLP, RLACE, LEACE)
   - Counterfactual generation
   - Causal inference tools

### What We Keep from Starter Code

1. **Core Architecture**
   - ELECTRA-small base model
   - HuggingFace integration
   - Trainer API usage
   - Dataset loading patterns

2. **NLI Implementation**
   - SNLI preprocessing (same approach)
   - Invalid label filtering
   - Accuracy computation

3. **QA Implementation** (Integrated from helpers.py)
   - Proper answer span extraction
   - F1/EM metric computation
   - QuestionAnsweringTrainer class

---

## 📚 Integration Notes

### Our run.py vs. Starter Code run.py

**Similarities**:
- Both use ELECTRA-small
- Both support SNLI and SQuAD
- Both use HuggingFace datasets/transformers
- Both filter SNLI invalid labels

**Differences**:
- **Ours**: More extensive argument parsing, additional parameters
- **Ours**: Integrated with analysis/mitigation modules
- **Ours**: Extended for artifact analysis workflows
- **Starter**: More concise, focused on basic training/eval
- **Starter**: Better QA postprocessing (which we now integrate)

### helpers.py Integration

We've downloaded and integrated `helpers.py` from the starter code:
- Used for proper QA postprocessing
- QuestionAnsweringTrainer for accurate F1/EM
- Compatible with our extended analysis features

---

## 🎓 Educational Context

The starter code is designed for:
- NLP course final projects
- Dataset artifact studies
- Introduction to bias detection
- Hands-on transformer fine-tuning

Our extensions are designed for:
- In-depth artifact analysis
- Advanced debiasing techniques
- Research-level investigation
- Systematic bias mitigation

---

## 📖 Citation

If using the starter code, please acknowledge:

```
Greg Durrett's fp-dataset-artifacts
https://github.com/gregdurrett/fp-dataset-artifacts
Developed by Kaj Bostrom, Jifan Chen, and Greg Durrett
University of Texas at Austin
```

For our extensions, cite the ELECTRA NLP Artifact Analysis project.

---

## 🔗 Related Resources

- **Official Repo**: https://github.com/gregdurrett/fp-dataset-artifacts
- **SNLI Dataset**: https://nlp.stanford.edu/projects/snli/
- **SQuAD Dataset**: https://rajpurkar.github.io/SQuAD-explorer/
- **ELECTRA Paper**: Clark et al., ICLR 2020
- **HuggingFace Docs**: https://huggingface.co/docs

---

**Last Updated**: November 2025
**Status**: Starter code integrated, extended with analysis/mitigation modules
