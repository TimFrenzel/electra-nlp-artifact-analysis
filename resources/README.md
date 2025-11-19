# ELECTRA NLP Artifact Analysis - Research Resources

This directory contains a curated collection of recent academic papers, GitHub repositories, benchmarks, and research groups focused on bias mitigation and artifact reduction in Natural Language Processing.

**Criteria**: All resources are from 2021-2025 (within last 4 years) and represent high-quality, peer-reviewed research or well-maintained open-source implementations.

## Directory Structure

```
resources/
├── README.md                    # This file
├── papers/                      # Academic papers organized by topic
│   ├── spurious_correlations.md
│   ├── debiasing_methods.md
│   ├── dataset_cartography.md
│   ├── adversarial_robustness.md
│   ├── counterfactual_augmentation.md
│   └── causal_inference.md
├── github_repos/                # High-quality GitHub repositories
│   ├── debiasing_tools.md
│   ├── robustness_evaluation.md
│   └── implementations.md
├── benchmarks/                  # Evaluation benchmarks and datasets
│   └── fairness_robustness.md
├── research_groups/             # Key researchers and institutions
│   └── influential_researchers.md
└── ANNOTATED_BIBLIOGRAPHY.md   # Comprehensive annotated bibliography
```

## Quick Links

### Most Important Papers (2023-2025)
1. **Mitigating Spurious Correlations** (ICLR 2025)
2. **MBIAS: Mitigating Bias in LLMs While Retaining Context** (2024)
3. **LEACE: Perfect Linear Concept Erasure** (NeurIPS 2023)
4. **Linear Adversarial Concept Erasure** (ICML 2022)

### Essential GitHub Repositories
1. [McGill-NLP/bias-bench](https://github.com/McGill-NLP/bias-bench) - ACL 2022 debiasing techniques survey
2. [allenai/cartography](https://github.com/allenai/cartography) - Dataset Cartography implementation
3. [chrisc36/debias](https://github.com/chrisc36/debias) - Ensemble-based debiasing methods
4. [shauli-ravfogel/nullspace_projection](https://github.com/shauli-ravfogel/nullspace_projection) - INLP for bias removal

### Key Benchmarks
1. **HANS** - Heuristic Analysis for NLI Systems
2. **BBQ** - Bias Benchmark for Question Answering (2022)
3. **CheckList** - Behavioral testing framework (Microsoft Research)
4. **WinoBias** - Gender bias evaluation

## Research Topics Covered

1. **Spurious Correlations & Dataset Artifacts**
   - Identification methods
   - Mitigation strategies
   - Evaluation frameworks

2. **Debiasing Methods**
   - Ensemble debiasing
   - Product of Experts
   - Adversarial debiasing
   - Concept erasure techniques

3. **Dataset Cartography**
   - Training dynamics analysis
   - Hard example identification
   - Data quality assessment

4. **Adversarial Robustness**
   - Adversarial examples generation
   - Robustness evaluation
   - Certified defenses

5. **Counterfactual Data Augmentation**
   - CDA methods for bias reduction
   - Automated generation techniques
   - Applications to fairness

6. **Causal Inference**
   - Causal reasoning in LLMs
   - Counterfactual learning
   - Intervention-based debiasing

## How to Use These Resources

### For Literature Review
1. Start with `ANNOTATED_BIBLIOGRAPHY.md` for overview
2. Dive into specific topic files in `papers/`
3. Check `research_groups/` for key authors to follow

### For Implementation
1. Browse `github_repos/` for code implementations
2. Reference papers for theoretical background
3. Use `benchmarks/` for evaluation

### For Experiments
1. Select relevant benchmarks from `benchmarks/`
2. Implement baseline methods from `github_repos/`
3. Compare against state-of-the-art from `papers/`

## Contributing

To add new resources:
1. Ensure publication/release is within last 4 years (2021+)
2. Verify quality (peer-reviewed papers, well-documented code)
3. Add to appropriate category with full citation
4. Update this README if adding new categories

## Citation

If you use these resources in your research, please cite the original authors. See individual files for specific citations.

## Last Updated

November 2025

---

**Maintained by**: ELECTRA NLP Artifact Analysis Project
**Contact**: See main repository README
