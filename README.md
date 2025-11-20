# ELECTRA NLP Artifact Analysis

**Professional Research Project** - Investigating whether pre-trained language models genuinely solve NLP tasks or exploit spurious correlations (dataset artifacts) in training data.

> **🚀 Optimized for Google Colab with A100 GPU** - Complete end-to-end workflow in interactive notebooks

> **Note**: This project builds on the [fp-dataset-artifacts starter code](https://github.com/gregdurrett/fp-dataset-artifacts) by Greg Durrett (UT Austin), extending it with comprehensive analysis and mitigation modules. See [`resources/starter_code.md`](resources/starter_code.md) for details.

## Overview

This project provides a complete **Google Colab-based workflow** to fine-tune ELECTRA-small on SNLI and systematically analyze dataset artifacts. Designed for researchers and practitioners with access to Google Colab GPU (A100 recommended), the project includes:

1. **Baseline Training** - Train ELECTRA-small on SNLI (1-3 hours on A100)
2. **Artifact Analysis** - Identify hypothesis-only bias, lexical overlap, and length correlations
3. **Mitigation** - Implement debiasing techniques to improve robustness
4. **Professional Reporting** - ACL-style technical report template

The work aims to understand if models learn genuine linguistic reasoning or merely pattern-match surface features.

### Why Google Colab?

- ✅ **Free A100 GPU access** - Train models in 1-3 hours vs 10+ hours on CPU
- ✅ **No local setup required** - Run directly in your browser
- ✅ **Persistent storage** - All results saved to Google Drive
- ✅ **Reproducible environment** - Consistent package versions
- ✅ **Ready-to-use notebooks** - Complete workflows with step-by-step execution

## Repository Structure

```
electra-nlp-artifact-analysis/
├── README.md
├── requirements.txt
├── run.py                          # Main training/evaluation script
├── helpers.py                      # Preprocessing & metrics (from starter code)
├── analysis/                       # Our analysis extensions
│   ├── error_analysis.py          # Tools for analyzing model failures
│   ├── contrast_sets.py           # Contrast set evaluation
│   └── visualization.py           # Plotting utilities
├── mitigation/                     # Our mitigation extensions
│   ├── dataset_cartography.py    # Hard example identification
│   ├── debiasing.py              # Bias mitigation methods
│   └── adversarial_training.py   # Adversarial augmentation
├── notebooks/                      # Interactive analysis (local)
│   ├── 01_baseline_training.ipynb
│   ├── 02_artifact_analysis.ipynb
│   └── 03_mitigation_eval.ipynb
├── colab_training.ipynb            # Google Colab: Baseline training (A100 GPU)
├── colab_analysis_part1.ipynb      # Google Colab: Part 1 - Artifact analysis
├── resources/                      # Research papers & repos (40+ resources)
│   ├── starter_code.md            # Starter code documentation
│   ├── papers/                    # Academic papers by topic
│   ├── github_repos/              # High-quality implementations
│   ├── benchmarks/                # Evaluation benchmarks
│   └── research_groups/           # Key researchers
├── report/                         # Technical report deliverables
│   ├── technical_report.md        # ACL-style report template (3-8 pages)
│   └── project_notes.md           # Project workflow and notes
├── results/
│   ├── baseline/
│   ├── analysis/
│   └── mitigated/
└── docs/
    └── report.pdf                 # Final technical report
```

## Prerequisites (Google Colab)

This project is **designed for Google Colab** and requires:

- ✅ **Google Account** - For Google Colab and Drive access
- ✅ **Google Colab** - Free tier works, Colab Pro ($9.99/month) recommended for priority A100 access
- ✅ **Google Drive** - ~5-10 GB free space for models and results
- ✅ **Web Browser** - Chrome, Firefox, or Safari

**No local installation, Python setup, or GPU required!**

### Getting Started (3 Steps)

1. **Open Google Colab** - Go to [colab.research.google.com](https://colab.research.google.com)
2. **Upload Notebooks** - Upload `colab_training.ipynb` and `colab_analysis_part1.ipynb` to your Drive
3. **Start Training** - Open `colab_training.ipynb` and run all cells

All dependencies install automatically in the notebooks.

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

## Quick Start (Google Colab Workflow)

### Step 1: Setup Your Environment (5 minutes)

1. **Download this repository**:
   ```bash
   git clone https://github.com/TimFrenzel/electra-nlp-artifact-analysis.git
   ```
   Or download as ZIP from GitHub

2. **Upload notebooks to Google Drive**:
   - Upload `colab_training.ipynb` to your Google Drive
   - Upload `colab_analysis_part1.ipynb` to your Google Drive
   - (Optional) Upload entire repository folder for easy access to code

3. **Open in Google Colab**:
   - Right-click `colab_training.ipynb` → Open with → Google Colaboratory
   - Select **GPU** runtime: Runtime → Change runtime type → GPU (T4, A100, or V100)

### Step 2: Train Baseline Model (1-3 hours with A100)

Open `colab_training.ipynb` and run all cells sequentially:

1. **GPU Verification** - Checks GPU availability and specs
2. **Mount Google Drive** - Connects to your Drive for persistent storage
3. **Clone Repository** - Downloads code and dependencies
4. **Install Dependencies** - Automatically installs requirements
5. **Configure Training** - Sets hyperparameters (SNLI, ELECTRA-small, 3 epochs)
6. **Quick Test** - Validates setup on 1000 examples (~2 minutes)
7. **Full Training** - Trains on complete SNLI dataset (1-3 hours)
8. **Evaluate Baseline** - Computes accuracy (~89% expected)

**Expected Output**:
- ✅ Baseline model saved to `Google Drive/electra-artifact-analysis/models/baseline_snli/`
- ✅ Training logs and metrics in `logs/`
- ✅ Accuracy: ~88-91%

### Step 3: Analyze Artifacts (1-2 hours with A100)

Open `colab_analysis_part1.ipynb` and run all cells:

1. **Hypothesis-Only Baseline** - Tests if model exploits hypothesis-only bias (~67% indicates severe artifacts)
2. **Lexical Overlap Analysis** - Correlation between word overlap and predictions
3. **Length Bias Analysis** - Impact of hypothesis length on predictions
4. **Error Characterization** - Systematic failure patterns
5. **Generate Visualizations** - Confusion matrices, distribution plots

**Expected Output**:
- ✅ Analysis results saved to `Google Drive/electra-artifact-analysis/analysis_results/`
- ✅ Figures saved to `figures/` for inclusion in report
- ✅ Hypothesis-only accuracy: ~65-70% (confirms artifacts)

### Step 4: Write Technical Report

Use `report/technical_report.md` as template:
- Fill in Section 4 with Part 1 analysis findings
- Include generated figures and tables
- Document baseline performance and artifact severity
- See `report/project_notes.md` for complete checklist

**Time Estimates (Google Colab A100)**:
- Setup: 5-10 minutes
- Baseline Training: 1-3 hours
- Artifact Analysis: 1-2 hours
- Report Writing: 2-4 hours
- **Total**: 4-8 hours for complete Part 1

---

---

## Alternative: Local Training (Advanced Users)

⚠️ **Not Recommended**: Local training requires significant setup and takes 10-15 hours on CPU. Google Colab is strongly recommended.

<details>
<summary><b>Click to expand local installation instructions</b></summary>

### Prerequisites for Local Setup

- Python >= 3.6
- PyTorch >= 1.0
- CUDA-capable GPU (8GB+ VRAM recommended) or CPU
- 10+ GB free disk space

### Local Installation Steps

**1. Clone the Repository**

```bash
git clone https://github.com/TimFrenzel/electra-nlp-artifact-analysis.git
cd electra-nlp-artifact-analysis
```

**2. Set Up Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Verify Installation**

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

### Local Training Commands

**Train Baseline Model (NLI/SNLI)**

```bash
python run.py --do_train \
    --task nli \
    --dataset snli \
    --output_dir ./models/baseline_snli \
    --num_train_epochs 3 \
    --fp16  # If GPU available
```

**Evaluate Baseline**

```bash
python run.py --do_eval \
    --task nli \
    --dataset snli \
    --model ./models/baseline_snli \
    --output_dir ./results/baseline
```

**Quick Debug Test (1000 examples)**

```bash
python run.py --do_train \
    --task nli \
    --dataset snli \
    --max_train_samples 1000 \
    --output_dir ./models/debug \
    --no_cuda
```

**CPU-only Training** (Very slow, 10-15 hours)

```bash
python run.py --do_train \
    --task nli \
    --dataset snli \
    --output_dir ./models/baseline \
    --no_cuda
```

</details>

## Expected Baseline Performance

Using ELECTRA-small with 3 epochs of training:

| Dataset | Metric | Expected Performance |
|---------|--------|---------------------|
| SNLI | Accuracy | ~89% |
| SQuAD | Exact Match | ~78% |
| SQuAD | F1 Score | ~86% |

*Performance varies with batch size and hyperparameters*

## Project Workflow (Google Colab)

### Phase 1: Baseline Training & Analysis (Complete ✅)

**Notebook**: `colab_training.ipynb` + `colab_analysis_part1.ipynb`

1. **Train baseline model** (1-3 hours on A100)
   - Open `colab_training.ipynb` in Google Colab
   - Run all cells to train ELECTRA-small on SNLI
   - Model saved to Google Drive automatically

2. **Analyze artifacts** using multiple methods (1-2 hours on A100):
   - Hypothesis-only baseline (tests for spurious correlations)
   - Lexical overlap analysis (word matching bias)
   - Length bias analysis (hypothesis length correlation)
   - Error characterization (systematic failure patterns)

3. **Document findings**:
   - Generated visualizations saved to `figures/`
   - Results exported to JSON for report
   - Fill in Section 4 of `report/technical_report.md`

### Phase 2: Mitigation Implementation (To Be Created)

Based on Phase 1 findings, create `colab_mitigation_part2.ipynb` with one or more debiasing approaches:

**Option 1: Product of Experts**
- Train hypothesis-only model (already done in Phase 1)
- Ensemble with full model to downweight biased examples
- Expected: Reduced hypothesis-only accuracy, better OOD performance

**Option 2: Dataset Cartography**
- Analyze training dynamics (confidence, variability)
- Identify "easy" examples that may rely on artifacts
- Focus training on "hard" examples

**Option 3: Adversarial Training**
- Generate counterfactual examples (entity swaps, negations)
- Augment training data
- Force model to use full premise-hypothesis interaction

### Phase 3: Evaluation & Comparison

Use mitigation notebook to:
- Evaluate debiased model on SNLI test set
- Test on out-of-distribution data (e.g., HANS)
- Measure hypothesis-only accuracy reduction
- Compare performance across overlap/length bins
- Document findings in Section 5 of technical report

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

## Computational Requirements

### ✅ Google Colab (Primary Platform - Strongly Recommended)

This project is **optimized for Google Colab** with GPU acceleration:

| Feature | Free Tier | Colab Pro ($9.99/month) |
|---------|-----------|------------------------|
| **GPU Access** | T4 (limited hours) | A100 (priority access) |
| **Training Time** | 3-6 hours (T4) | 1-3 hours (A100) |
| **Session Length** | ~12 hours | ~24 hours |
| **Drive Storage** | 15 GB free | 15 GB free (100 GB option available) |
| **Recommendation** | ✅ Works for project | ⭐ Best experience |

**Why Colab?**
- 🚀 **10-15× faster** than CPU training
- 💰 **Free GPU access** (no hardware purchase needed)
- 📦 **Zero setup** - runs in browser
- 💾 **Persistent storage** via Google Drive
- 🔄 **Reproducible** - same environment every time

**Notebooks Included**:
- `colab_training.ipynb` - Complete training workflow
- `colab_analysis_part1.ipynb` - Comprehensive artifact analysis

### Alternative Compute Options (Not Recommended)

<details>
<summary>Click to see alternative platforms</summary>

**Google Cloud Platform**
- Free $300 credits for new accounts
- Sufficient for project completion
- Requires manual VM setup

**Kaggle Notebooks**
- 30 hours/week free GPU (T4/P100)
- Similar to Colab but with usage limits
- Requires adapting notebooks

**Local GPU**
- Any CUDA-capable GPU with 8GB+ VRAM
- Requires significant local setup
- See "Alternative: Local Training" section

**Local CPU** ⚠️
- **Not recommended** - 10-15 hours training time
- Very slow, not practical for research

</details>

## Troubleshooting

### Google Colab Issues

**No GPU available / "Connect to a GPU runtime"**
- Solution: Runtime → Change runtime type → Hardware accelerator → GPU
- Try reconnecting if GPU is busy
- Colab Pro provides priority access to A100/V100

**Session disconnected / Runtime crashed**
- Colab free tier has 12-hour session limits
- Solution: Restart runtime and re-run cells (checkpoints saved to Drive)
- Your models/results persist in Google Drive

**Drive mounting fails**
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

**Repository clone fails**
- Solution: Check GitHub repository URL is correct
- Or manually upload repository ZIP to Drive and extract

**Out of memory during training**
- Reduce batch size in CONFIG (32 → 16 → 8)
- Use gradient accumulation: `--gradient_accumulation_steps 2`
- T4 GPU has less memory than A100

**Package installation errors**
```bash
!pip install --upgrade pip setuptools wheel
!pip install -r requirements.txt --force-reinstall
```

### Local Installation Issues

<details>
<summary>Click to expand local troubleshooting</summary>

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
ls ./models/baseline_snli/
# Should contain: config.json, pytorch_model.bin, etc.
```

</details>

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
