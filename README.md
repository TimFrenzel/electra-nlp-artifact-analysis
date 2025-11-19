# ELECTRA NLP Artifact Analysis

A research project investigating whether pre-trained language models genuinely solve NLP tasks or exploit spurious correlations (dataset artifacts) in training data.

## Overview

This project fine-tunes ELECTRA-small on standard NLP benchmarks to:
1. **Analyze** model behavior and identify dataset artifacts through systematic evaluation
2. **Mitigate** identified biases using debiasing techniques
3. **Evaluate** whether interventions improve model robustness and generalization

The work aims to understand if models learn genuine linguistic reasoning or merely pattern-match surface features.

## Repository Structure

```
electra-nlp-artifact-analysis/
├── README.md
├── requirements.txt
├── run.py                          # Main training/evaluation script
├── analysis/
│   ├── error_analysis.py          # Tools for analyzing model failures
│   ├── contrast_sets.py           # Contrast set evaluation
│   └── visualization.py           # Plotting utilities
├── mitigation/
│   ├── dataset_cartography.py    # Hard example identification
│   ├── debiasing.py              # Bias mitigation methods
│   └── adversarial_training.py   # Adversarial augmentation
├── notebooks/
│   ├── 01_baseline_training.ipynb
│   ├── 02_artifact_analysis.ipynb
│   └── 03_mitigation_eval.ipynb
├── results/
│   ├── baseline/
│   ├── analysis/
│   └── mitigated/
└── docs/
    └── report.pdf                 # Final technical report
```

## Prerequisites

- Python >= 3.6
- PyTorch >= 1.0
- CUDA-capable GPU (recommended) or CPU

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/electra-nlp-artifact-analysis.git
cd electra-nlp-artifact-analysis
```

### 2. Set Up Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
pip --version  # Ensure correct Python version
python -c "import torch; print(torch.__version__)"
```

## Dataset Options

Choose one of the following datasets for your analysis:

### Natural Language Inference (NLI)

**SNLI (Stanford Natural Language Inference)**
- **Size**: 570k sentence pairs
- **Task**: Classify pairs as entailment/contradiction/neutral
- **URL**: https://nlp.stanford.edu/projects/snli/
- **HuggingFace**: `stanfordnlp/snli`

**MultiNLI (Multi-Genre NLI)**
- **Size**: 433k sentence pairs
- **Task**: Cross-genre entailment classification
- **URL**: https://www.nyu.edu/projects/bowman/multinli/
- **HuggingFace**: `nyu-mll/multi_nli`

### Question Answering (QA)

**SQuAD (Stanford Question Answering Dataset)**
- **Size**: 100k+ question-answer pairs
- **Task**: Extract answer spans from Wikipedia passages
- **URL**: https://rajpurkar.github.io/SQuAD-explorer/
- **HuggingFace**: `squad`

**HotpotQA**
- **Size**: 113k question-answer pairs
- **Task**: Multi-hop reasoning over multiple documents
- **URL**: https://hotpotqa.github.io/
- **HuggingFace**: `hotpotqa/hotpot_qa`

## Quick Start

### 1. Train Baseline Model

**For NLI (SNLI):**
```bash
python run.py --do_train \
    --task nli \
    --dataset snli \
    --output_dir ./models/baseline_snli \
    --num_train_epochs 3
```

**For QA (SQuAD):**
```bash
python run.py --do_train \
    --task qa \
    --dataset squad \
    --output_dir ./models/baseline_squad \
    --num_train_epochs 3
```

**CPU-only training:**
```bash
python run.py --do_train --task nli --dataset snli --output_dir ./models/baseline --no_cuda
```

### 2. Evaluate Baseline

```bash
python run.py --do_eval \
    --task nli \
    --dataset snli \
    --model ./models/baseline_snli \
    --output_dir ./results/baseline
```

### 3. Debug on Small Subset

```bash
python run.py --do_train \
    --task nli \
    --dataset snli \
    --max_train_samples 1000 \
    --output_dir ./models/debug \
    --no_cuda
```

## Expected Baseline Performance

Using ELECTRA-small with 3 epochs of training:

| Dataset | Metric | Expected Performance |
|---------|--------|---------------------|
| SNLI | Accuracy | ~89% |
| SQuAD | Exact Match | ~78% |
| SQuAD | F1 Score | ~86% |

*Performance varies with batch size and hyperparameters*

## Project Workflow

### Phase 1: Baseline Training & Analysis

1. **Train baseline model** on chosen dataset (5-15 hours on CPU, 1-3 hours on GPU)
2. **Analyze errors** using one or more methods:
   - Contrast sets (minimal input perturbations)
   - Model ablations (hypothesis-only, question-only baselines)
   - Adversarial examples
   - Statistical tests (n-gram correlation analysis)
3. **Document findings**:
   - Specific error examples
   - General failure patterns
   - Visualizations of artifact correlations

### Phase 2: Mitigation Implementation

Select and implement one debiasing approach:

**Option 1: Dataset Cartography**
```python
# Identify hard/ambiguous examples during training
python mitigation/dataset_cartography.py --model ./models/baseline --dataset snli
```

**Option 2: Ensemble Debiasing**
```python
# Train artifact expert, then debias main model
python mitigation/debiasing.py --method ensemble --dataset snli
```

**Option 3: Adversarial Training**
```python
# Augment training with adversarial examples
python mitigation/adversarial_training.py --dataset snli
```

### Phase 3: Evaluation & Analysis

```bash
# Evaluate mitigated model
python run.py --do_eval \
    --task nli \
    --dataset snli \
    --model ./models/mitigated_snli \
    --output_dir ./results/mitigated

# Compare against baseline
python analysis/compare_models.py \
    --baseline ./results/baseline \
    --mitigated ./results/mitigated
```

## Configuration

### Key Arguments for `run.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--do_train` | Enable training mode | False |
| `--do_eval` | Enable evaluation mode | False |
| `--task` | Task type: `nli` or `qa` | Required |
| `--dataset` | Dataset name | Required |
| `--output_dir` | Directory for checkpoints/results | Required |
| `--model` | Path to pretrained model | `google/electra-small-discriminator` |
| `--num_train_epochs` | Number of training epochs | 3 |
| `--batch_size` | Training batch size | 32 |
| `--learning_rate` | Learning rate | 2e-5 |
| `--max_train_samples` | Limit training samples (for debugging) | None |
| `--no_cuda` | Force CPU-only training | False |
| `--save_steps` | Checkpoint frequency | 500 |

### Data Caching

- Models and datasets automatically cache to `~/.cache/huggingface/`
- Change cache location: `export HF_HOME=/path/to/cache`

## Analysis Tools

### Error Analysis

```python
from analysis.error_analysis import analyze_predictions

# Analyze where model fails
results = analyze_predictions(
    model_path='./models/baseline_snli',
    dataset='snli',
    split='validation'
)

# Generate error distribution plots
results.plot_error_types()
results.export_failures('errors.csv')
```

### Contrast Set Evaluation

```python
from analysis.contrast_sets import evaluate_contrast_set

# Test model on minimally-perturbed examples
contrast_results = evaluate_contrast_set(
    model_path='./models/baseline_snli',
    original_data='snli_dev.json',
    contrast_data='snli_contrast.json'
)
```

### Visualization

```python
from analysis.visualization import plot_performance_breakdown

# Visualize performance across data slices
plot_performance_breakdown(
    results_dir='./results/baseline',
    group_by='example_difficulty'  # or 'artifact_type', 'length', etc.
)
```

## Computational Resources

### Training Time Estimates

| Dataset | Examples | CPU Time | GPU Time |
|---------|----------|----------|----------|
| SNLI (full) | 550k | 10-15 hours | 2-3 hours |
| SQuAD (full) | 87k | 5-10 hours | 1-2 hours |
| Debug (1k) | 1k | 5-10 minutes | 1-2 minutes |

### Optimization Tips

1. **Debug on small data first**: Use `--max_train_samples 1000` for rapid iteration
2. **Monitor checkpoints**: Evaluate checkpoints to find optimal training time
3. **Use mixed precision**: Add `--fp16` flag for faster GPU training
4. **Batch size**: Increase if GPU memory allows (faster training)

## Free Compute Options

- **Google Colab**: Free GPU access with session limits
  - Pro tier: $9.99/month for longer sessions
  - Example notebook: `notebooks/colab_training.ipynb`
- **Google Cloud Platform**: Free credits for new accounts
  - Sufficient for full project completion
- **Kaggle Notebooks**: 30 hours/week free GPU

## Troubleshooting

### Common Issues

**ImportError: No module named 'transformers'**
```bash
pip install transformers
```

**CUDA out of memory**
```bash
# Reduce batch size or use CPU
python run.py --do_train --batch_size 8 --no_cuda
```

**Dataset download fails**
```bash
# Manually set cache directory with write permissions
export HF_HOME=/path/with/space
```

**Checkpoints not loading**
```bash
# Verify checkpoint directory structure
ls ./models/baseline_snli/checkpoint-1000/
# Should contain: config.json, pytorch_model.bin, etc.
```

## Citation

If you use this code or findings in your work, please cite:

```bibtex
@misc{electra-artifact-analysis,
  author = {Your Name},
  title = {ELECTRA NLP Artifact Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/electra-nlp-artifact-analysis}
}
```

### Key References

**ELECTRA:**
```bibtex
@inproceedings{clark2020electra,
  title={ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators},
  author={Clark, Kevin and Luong, Minh-Thang and Le, Quoc V and Manning, Christopher D},
  booktitle={ICLR},
  year={2020}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
