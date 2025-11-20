# Project Notes - ELECTRA NLP Artifact Analysis

**Professional Research Project**
**Created**: November 2025
**Last Updated**: November 2025

---

## Project Overview

This is a **professional research project** (NOT academic coursework) investigating dataset artifacts in natural language inference models. The goal is to determine whether ELECTRA-small genuinely learns language understanding or exploits spurious correlations in the SNLI dataset.

---

## Project Structure

```
electra-nlp-artifact-analysis/
├── run.py                          # Main training/evaluation script
├── helpers.py                      # Preprocessing utilities (from starter code)
├── requirements.txt                # Python dependencies
├── analysis/                       # Analysis modules
│   ├── error_analysis.py          # Error characterization
│   ├── contrast_sets.py           # Robustness evaluation
│   └── visualization.py           # Plotting utilities
├── mitigation/                     # Debiasing modules
│   ├── dataset_cartography.py     # Training dynamics
│   ├── debiasing.py               # Ensemble methods
│   └── adversarial_training.py    # Data augmentation
├── resources/                      # Research documentation
│   ├── papers/                    # 40+ annotated papers
│   ├── github_repos/              # 17 relevant repositories
│   ├── benchmarks/                # Evaluation datasets
│   └── starter_code.md            # Integration docs
├── report/                         # Technical report
│   ├── technical_report.md        # ACL-style report template
│   └── project_notes.md           # This file
├── colab_training.ipynb            # Part 0: Baseline training
├── colab_analysis_part1.ipynb      # Part 1: Artifact analysis
└── colab_mitigation_part2.ipynb    # Part 2: Mitigation (to be created)
```

---

## Workflow

### Phase 0: Setup and Baseline Training
**File**: `colab_training.ipynb`
**Time**: 1-3 hours with A100 GPU

1. Mount Google Drive for persistent storage
2. Clone repository
3. Install dependencies
4. Train baseline ELECTRA-small on SNLI
5. Evaluate baseline performance (~89% expected)
6. Quick artifact check (hypothesis-only preview)

**Outputs**:
- Trained model: `Google Drive/electra-artifact-analysis/models/baseline_snli/`
- Training logs: `Google Drive/electra-artifact-analysis/logs/`
- Baseline results: `Google Drive/electra-artifact-analysis/results/baseline/`

### Phase 1: Artifact Analysis
**File**: `colab_analysis_part1.ipynb`
**Time**: 1-2 hours with A100 GPU

1. **Hypothesis-Only Baseline**
   - Train model using only hypothesis (no premise)
   - Quantify artifact severity (expected ~67% vs. 33% random)
   - Per-class analysis

2. **Lexical Overlap Analysis**
   - Compute word overlap between premise/hypothesis
   - Stratify by overlap level
   - Test correlation with predictions

3. **Length Bias Analysis**
   - Stratify by hypothesis length
   - Analyze label distribution
   - Check for length-based shortcuts

4. **Error Characterization**
   - Identify systematic failure patterns
   - Confusion matrix analysis
   - Qualitative error examples

**Outputs**:
- Hypothesis-only model: `Google Drive/electra-artifact-analysis/models/hypothesis_only/`
- Analysis results: `Google Drive/electra-artifact-analysis/analysis_results/`
- Figures: `Google Drive/electra-artifact-analysis/figures/`
  - `hypothesis_only_confusion_matrix.png`
  - `lexical_overlap_analysis.png`
  - `length_bias_analysis.png`

**For Technical Report Section 4**: Use all findings and visualizations

### Phase 2: Mitigation and Evaluation
**File**: `colab_mitigation_part2.ipynb` (to be created)
**Time**: Variable depending on method

Based on Part 1 findings, implement debiasing method:

**Option A: Product of Experts**
- Train hypothesis-only model (done in Part 1)
- Ensemble with full model
- Downweight examples biased model gets correct

**Option B: Dataset Cartography**
- Analyze training dynamics
- Identify easy vs. hard examples
- Focus training on hard examples

**Option C: Adversarial Training**
- Generate counterfactual examples
- Data augmentation with entity swaps
- Force model to use full context

**Outputs**:
- Debiased model: `Google Drive/electra-artifact-analysis/models/debiased/`
- Mitigation results: `Google Drive/electra-artifact-analysis/results/mitigation/`
- Comparison figures

**For Technical Report Section 5**: Mitigation approach and results

---

## Key Datasets

### SNLI (Primary)
- **Size**: 570,000 sentence pairs
- **Task**: 3-way classification (entailment, neutral, contradiction)
- **URL**: https://nlp.stanford.edu/projects/snli/
- **Preprocessing**: Filter label=-1 (invalid labels, ~0.7%)

### HANS (Optional OOD Evaluation)
- **Size**: 30,000 examples
- **Purpose**: Test syntactic heuristics
- **URL**: https://github.com/tommccoy1/hans
- **Usage**: Out-of-distribution robustness test

---

## Computational Details

### Google Colab Setup
- **GPU**: A100 (recommended) or T4 (slower but functional)
- **Storage**: Google Drive for persistent models/results
- **Runtime**: Standard GPU runtime sufficient
- **Directories**:
  - `/content/drive/MyDrive/electra-artifact-analysis/` - All outputs
  - `/content/electra-nlp-artifact-analysis/` - Cloned repository

### Training Time Estimates (A100)
- Baseline training (3 epochs, full SNLI): 1-3 hours
- Hypothesis-only (3 epochs, full SNLI): 30-60 minutes
- Evaluation: 5-10 minutes
- Analysis notebooks: 1-2 hours total

### Training Time Estimates (T4)
- Baseline training: 4-6 hours
- Hypothesis-only: 1-2 hours
- Evaluation: 10-15 minutes

---

## Expected Results

### Baseline Performance
- **SNLI Accuracy**: 88-91%
- **Hypothesis-Only**: 65-70% (indicates severe artifacts)
- **HANS Accuracy** (if evaluated): 50-60%

### After Debiasing (Target)
- **SNLI Accuracy**: 87-90% (slight drop acceptable)
- **Hypothesis-Only**: 40-50% (lower is better)
- **HANS Accuracy**: 65-75% (higher is better)

### Interpretation Guidelines
- Hypothesis-only > 65%: **SEVERE** artifact exploitation
- Hypothesis-only 55-65%: **MODERATE** artifacts
- Hypothesis-only 45-55%: **MILD** artifacts
- Hypothesis-only < 45%: **MINIMAL** artifacts

---

## Technical Report Requirements

### Format
- **Style**: ACL conference format (professional technical paper)
- **Length**: 3-8 pages (excluding references)
- **Sections**:
  1. Abstract (150-200 words)
  2. Introduction (motivation, research questions, contributions)
  3. Background and Related Work
  4. Methodology (setup, baseline, analysis protocol)
  5. Part 1: Artifact Analysis (findings from analysis notebook)
  6. Part 2: Mitigation (debiasing approach and evaluation)
  7. Discussion (implications, limitations, future work)
  8. Conclusion
  9. References (comprehensive bibliography)
  10. Appendix (hyperparameters, sample predictions)

### Content Requirements
- **Figures**: Include all generated visualizations with captions
- **Tables**: Quantitative results with proper formatting
- **Citations**: Reference all key papers from resources/papers/
- **Analysis**: Both quantitative metrics and qualitative interpretation
- **Professional tone**: Research paper style, NOT homework/assignment language

---

## Key Papers to Cite

### Foundational
1. **Bowman et al. (2015)** - SNLI dataset
2. **Clark et al. (2020)** - ELECTRA model

### Artifact Detection
3. **Gururangan et al. (2018)** - Annotation artifacts (hypothesis-only bias)
4. **McCoy et al. (2019)** - HANS, syntactic heuristics

### Debiasing Methods
5. **Clark et al. (2019)** - Product of Experts
6. **Swayamdipta et al. (2020)** - Dataset Cartography

### Additional (from resources/papers/)
- See `resources/ANNOTATED_BIBLIOGRAPHY.md` for full list
- See `resources/papers/snli_specific.md` for SNLI-focused work

---

## Common Issues and Solutions

### Issue: GPU out of memory
**Solution**: Reduce batch size (32 → 16 → 8)

### Issue: Training too slow
**Solution**: Use FP16 mixed precision (--fp16 flag)

### Issue: Model not learning
**Solution**: Check learning rate, verify data preprocessing, ensure labels are correct

### Issue: Invalid labels in SNLI
**Solution**: Always filter `label != -1` (already implemented in code)

### Issue: Hypothesis-only accuracy too low
**Solution**: Ensure using only hypothesis text, not concatenated with premise

---

## Checklist

### Before Training
- [ ] Google Drive mounted
- [ ] Repository cloned
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] GPU detected (`nvidia-smi`)
- [ ] SNLI dataset accessible

### After Part 1 (Analysis)
- [ ] Baseline model trained and saved
- [ ] Baseline accuracy ~89%
- [ ] Hypothesis-only model trained
- [ ] Hypothesis-only accuracy computed (~67%)
- [ ] Lexical overlap analysis complete
- [ ] Length bias analysis complete
- [ ] Error analysis done
- [ ] All figures generated and saved
- [ ] Results JSON files saved to Google Drive

### After Part 2 (Mitigation)
- [ ] Debiasing method chosen based on Part 1 findings
- [ ] Debiased model trained
- [ ] In-distribution evaluation (SNLI test)
- [ ] Out-of-distribution evaluation (if applicable)
- [ ] Hypothesis-only accuracy reduced
- [ ] Comparison figures generated
- [ ] All results saved

### Technical Report
- [ ] Abstract written (150-200 words)
- [ ] Introduction complete (motivation, RQs, contributions)
- [ ] Related work section (6+ citations)
- [ ] Methodology described (setup, protocol)
- [ ] Part 1 results filled in (Section 4)
- [ ] Part 2 results filled in (Section 5)
- [ ] Discussion section (implications, limitations)
- [ ] Conclusion written
- [ ] All figures included with captions
- [ ] All tables formatted properly
- [ ] References complete and formatted
- [ ] Appendix with hyperparameters
- [ ] Page count: 3-8 pages
- [ ] NO homework/assignment language
- [ ] Professional research tone throughout

---

## Tips for Success

### Analysis
1. **Always visualize**: Confusion matrices, distributions, correlations
2. **Stratify analysis**: Don't just report overall accuracy
3. **Interpret quantitatively**: Use statistical tests, confidence intervals
4. **Sample errors**: Qualitative analysis complements quantitative

### Mitigation
1. **Start simple**: Implement one method well rather than multiple poorly
2. **Expect trade-offs**: In-distribution performance may decrease slightly
3. **Validate thoroughly**: Use multiple evaluation metrics
4. **Document everything**: Save all hyperparameters and results

### Report Writing
1. **Tell a story**: Introduction → Problem → Analysis → Solution → Results
2. **Be precise**: Use exact numbers, not vague descriptions
3. **Justify choices**: Explain why you chose specific methods
4. **Acknowledge limitations**: Honest assessment of what you didn't do
5. **Professional tone**: Research paper, not project writeup

---

## Resources

### Code References
- **Starter Code**: https://github.com/gregdurrett/fp-dataset-artifacts
- **HuggingFace Docs**: https://huggingface.co/docs
- **ELECTRA Model**: https://huggingface.co/google/electra-small-discriminator

### Dataset References
- **SNLI**: https://nlp.stanford.edu/projects/snli/
- **HANS**: https://github.com/tommccoy1/hans
- **MultiNLI**: https://cims.nyu.edu/~sbowman/multinli/

### Paper Collections
- See `resources/papers/` for organized collections
- See `resources/ANNOTATED_BIBLIOGRAPHY.md` for summaries

### Research Groups
- See `resources/research_groups/influential_researchers.md`

---

## Contact and Attribution

### Project Attribution
This project builds on the fp-dataset-artifacts starter code by Kaj Bostrom, Jifan Chen, and Greg Durrett (UT Austin). All extensions (analysis modules, mitigation methods, research documentation) are original work.

### Starter Code Citation
```
Greg Durrett's fp-dataset-artifacts
https://github.com/gregdurrett/fp-dataset-artifacts
Developed by Kaj Bostrom, Jifan Chen, and Greg Durrett
University of Texas at Austin
```

---

## Version History

- **v1.0** (November 2025): Initial project setup
  - Baseline implementation
  - Training notebook
  - Analysis notebook
  - Report template
  - Research resources

---

**Last Updated**: November 2025
**Project Status**: Professional Research Project
**Next Steps**: Complete Part 1 analysis → Implement Part 2 mitigation → Write technical report
