# GitHub Repositories: Debiasing Tools and Implementations

High-quality, actively maintained repositories for bias detection and mitigation in NLP.

---

## Comprehensive Frameworks

### 1. McGill-NLP/bias-bench ⭐⭐⭐⭐⭐
- **URL**: https://github.com/McGill-NLP/bias-bench
- **Paper**: An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models (ACL 2022)
- **Website**: https://mcgill-nlp.github.io/bias-bench/
- **Description**: Official benchmark for evaluating debiasing techniques across three intrinsic bias benchmarks
- **Features**:
  - Implements multiple debiasing methods (Dropout, INLP, CDA, Sentence Debias)
  - Evaluates on StereoSet, CrowS-Pairs, SEAT
  - Easy to extend with new methods
  - Comprehensive documentation
- **Language**: Python
- **Last Updated**: 2022-2023 (stable, well-maintained)
- **Stars**: 100+
- **Use Case**: Benchmarking and comparing debiasing methods

### 2. chrisc36/debias ⭐⭐⭐⭐⭐
- **URL**: https://github.com/chrisc36/debias
- **Paper**: Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases
- **Description**: Ensemble-based debiasing methods to avoid dataset biases
- **Features**:
  - Product of Experts implementation
  - Learned-mixin adaptive variant
  - Hypothesis-only and question-only baselines
  - Multiple task implementations (SQuAD, MNLI, VQA, TriviaQA-CP)
  - Both TensorFlow and PyTorch versions
- **Language**: Python (TensorFlow, PyTorch)
- **Last Updated**: Active through 2023
- **Stars**: 200+
- **Use Case**: Ensemble debiasing, avoiding spurious correlations

### 3. VectorInstitute/bias-mitigation-unlearning ⭐⭐⭐⭐
- **URL**: https://github.com/VectorInstitute/bias-mitigation-unlearning
- **Paper**: Mitigating Social Biases in Language Models through Unlearning (EMNLP 2024)
- **Description**: Machine unlearning methods for bias mitigation
- **Features**:
  - Negation via Task Vectors (25.5% bias reduction)
  - PCGU implementation
  - Evaluation on multiple LLMs (LLaMA-2, OPT)
  - Minimal perplexity degradation
- **Language**: Python
- **Last Updated**: 2024 (very recent)
- **Stars**: Growing (new repo)
- **Use Case**: LLM debiasing through unlearning

---

## Concept Erasure and Information Removal

### 4. shauli-ravfogel/nullspace_projection (INLP) ⭐⭐⭐⭐⭐
- **URL**: https://github.com/shauli-ravfogel/nullspace_projection
- **Paper**: Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (ACL 2020)
- **Author**: Shauli Ravfogel, Yoav Goldberg
- **Description**: Remove information from neural representations via iterative nullspace projection
- **Features**:
  - Iterative training of linear classifiers
  - Projection onto null-space
  - Makes representations oblivious to target properties
  - Well-documented API
- **Language**: Python
- **Last Updated**: Maintained through 2024
- **Stars**: 300+
- **Citations**: 800+ (highly influential)
- **Use Case**: Removing gender, race, other protected attributes from embeddings

### 5. shauli-ravfogel/rlace-icml (RLACE) ⭐⭐⭐⭐⭐
- **URL**: https://github.com/shauli-ravfogel/rlace-icml
- **Paper**: Linear Adversarial Concept Erasure (ICML 2022)
- **Website**: https://shauli-ravfogel.netlify.app/publication/rlace/
- **Description**: Adversarial concept erasure using minimax game
- **Features**:
  - Identifies rank-k subspace for concept neutralization
  - Prevents linear classifiers from recovering concepts
  - Minimal representation change
  - Closed-form solution variants
- **Language**: Python
- **Last Updated**: 2023-2024
- **Stars**: 150+
- **Citations**: 500+
- **Use Case**: Advanced concept erasure, theoretical guarantees

### 6. EleutherAI/concept-erasure (LEACE) ⭐⭐⭐⭐
- **URL**: https://github.com/EleutherAI/concept-erasure
- **Paper**: LEACE: Perfect Linear Concept Erasure in Closed Form (NeurIPS 2023)
- **Description**: Provably perfect linear concept erasure
- **Features**:
  - Closed-form solution
  - Prevents ALL linear classifiers from detecting concept
  - Minimal representation change (least-squares)
  - Concept scrubbing for LLMs
- **Language**: Python
- **Last Updated**: 2024 (actively maintained by EleutherAI)
- **Stars**: 200+
- **Use Case**: Strongest guarantees for linear concept removal

---

## Fairness Benchmarks and Evaluation

### 7. uclanlp/awesome-fairness-papers ⭐⭐⭐⭐
- **URL**: https://github.com/uclanlp/awesome-fairness-papers
- **Description**: Curated list of fairness papers in NLP (2022-2024)
- **Features**:
  - Papers from ACL, EMNLP, NeurIPS, etc.
  - Organized by conference and year
  - Includes links to code when available
  - Regularly updated
- **Language**: Markdown (paper collection)
- **Last Updated**: 2024 (active)
- **Stars**: 500+
- **Use Case**: Literature review, staying current with fairness research

### 8. i-gallegos/Fair-LLM-Benchmark ⭐⭐⭐
- **URL**: https://github.com/i-gallegos/Fair-LLM-Benchmark
- **Description**: Fairness benchmarking for large language models
- **Features**:
  - Multiple fairness metrics
  - Evaluation across demographic groups
  - Compatible with HuggingFace models
  - Includes recent LLMs (Gemini, Claude, GPT-4, LLaMA)
- **Language**: Python
- **Last Updated**: 2024
- **Stars**: 50+ (growing)
- **Use Case**: Evaluating LLM fairness

---

## Adversarial Attack and Defense

### 9. thunlp/TAADpapers ⭐⭐⭐⭐⭐
- **URL**: https://github.com/thunlp/TAADpapers
- **Description**: Must-read papers on Textual Adversarial Attack and Defense
- **Features**:
  - Comprehensive paper list (2022-2024)
  - Organized by attack type and defense method
  - Links to code implementations
  - Regular updates
- **Recent Papers**:
  - SemRoDe (NAACL 2024)
  - DSRM (ACL 2023)
  - Generative Adversarial Training (EMNLP 2023)
- **Language**: Markdown (paper collection)
- **Last Updated**: 2024 (very active)
- **Stars**: 1000+
- **Use Case**: Adversarial robustness research

---

## Industry Tools and Frameworks

### 10. Trusted-AI/AIF360 (IBM) ⭐⭐⭐⭐⭐
- **URL**: https://github.com/Trusted-AI/AIF360
- **Website**: https://aif360.res.ibm.com/
- **Description**: Comprehensive fairness metrics and bias mitigation algorithms
- **Features**:
  - 70+ fairness metrics
  - 10+ bias mitigation algorithms
  - Pre-processing, in-processing, post-processing methods
  - Extensive documentation and tutorials
  - R and Python APIs
- **Language**: Python, R
- **Maintained by**: IBM Research
- **Last Updated**: 2024 (actively maintained)
- **Stars**: 2000+
- **Use Case**: Production fairness evaluation and mitigation

### 11. marcotcr/checklist (Microsoft Research) ⭐⭐⭐⭐⭐
- **URL**: https://github.com/marcotcr/checklist
- **Paper**: Beyond Accuracy: Behavioral Testing of NLP Models with CheckList (ACL 2020)
- **Website**: https://www.microsoft.com/en-us/research/publication/beyond-accuracy-behavioral-testing-of-nlp-models-with-checklist/
- **Description**: Behavioral testing framework for NLP models
- **Features**:
  - Task-agnostic testing methodology
  - Matrix of linguistic capabilities and test types
  - Software tool for generating test cases
  - Perturbation capabilities (name changes, typos, etc.)
- **Language**: Python
- **Last Updated**: 2023-2024 (stable)
- **Stars**: 1500+
- **Citations**: 1000+
- **Use Case**: Comprehensive model testing, robustness evaluation

---

## AllenAI Repositories

### 12. allenai/allennlp-models ⭐⭐⭐⭐
- **URL**: https://github.com/allenai/allennlp-models
- **Description**: Pre-trained models including bias-mitigated versions
- **Features**:
  - RoBERTa fine-tuned on SNLI with adversarial bias mitigation
  - Binary gender bias mitigation models
  - Well-documented model cards
- **Language**: Python
- **Last Updated**: 2022 (maintenance mode)
- **Stars**: 1000+
- **Use Case**: Pre-trained debiased models

### 13. allenai/persona-bias ⭐⭐⭐⭐
- **URL**: https://github.com/allenai/persona-bias
- **Paper**: Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs (ICLR 2024)
- **Dataset**: https://huggingface.co/datasets/allenai/persona-bias
- **Description**: Code and dataset for persona-based bias analysis
- **Features**:
  - Outputs from 4 LLMs (ChatGPT, GPT-4, LLaMA-2)
  - Implicit reasoning bias evaluation
  - June'23 and Nov'23 model versions
- **Language**: Python
- **Last Updated**: 2024 (recent)
- **Stars**: 50+ (growing)
- **Use Case**: LLM bias analysis, persona effects

---

## Causal Inference for Debiasing

### 14. zhijing-jin/CausalNLP_Papers ⭐⭐⭐⭐
- **URL**: https://github.com/zhijing-jin/CausalNLP_Papers
- **Description**: Reading list for causality in NLP
- **Features**:
  - Papers on causal inference for NLP
  - Counterfactual reasoning
  - Intervention-based debiasing
  - Organized by topic
- **Language**: Markdown (paper collection)
- **Last Updated**: 2024 (active)
- **Stars**: 300+
- **Use Case**: Causal approaches to debiasing

### 15. anpwu/Awesome-Causal-LLM ⭐⭐⭐
- **URL**: https://github.com/anpwu/Awesome-Causal-LLM
- **Description**: Causality and large language models
- **Features**:
  - Recent papers on causal LLMs (2023-2024)
  - Causal reasoning evaluation
  - Causal intervention methods
- **Language**: Markdown (paper collection)
- **Last Updated**: 2024
- **Stars**: 100+ (growing)
- **Use Case**: Causal inference in modern LLMs

---

## HuggingFace Models

### 16. d4data/bias-detection-model
- **URL**: https://huggingface.co/d4data/bias-detection-model
- **Description**: Pre-trained bias detection model
- **Features**:
  - Ready-to-use bias classifier
  - HuggingFace Transformers compatible
  - Fine-tuned for bias detection
- **Last Updated**: 2023-2024
- **Downloads**: 1000+
- **Use Case**: Quick bias detection in production

### 17. newsmediabias/MBIAS
- **URL**: https://huggingface.co/newsmediabias/MBIAS
- **Paper**: MBIAS: Mitigating Bias in LLMs While Retaining Context (2024)
- **Description**: Fine-tuned LLM with bias mitigation
- **Performance**: 30%+ bias reduction, >90% in specific demographics
- **Features**:
  - Retains contextual accuracy
  - Reduced toxicity
  - Ready for inference
- **Last Updated**: 2024 (recent)
- **Use Case**: Debiased text generation

---

## Quick Reference

### For Getting Started
1. **McGill-NLP/bias-bench** - Comprehensive evaluation framework
2. **chrisc36/debias** - Ensemble methods
3. **Trusted-AI/AIF360** - Industry-standard toolkit

### For Research
1. **shauli-ravfogel/nullspace_projection** (INLP) - Foundational method
2. **shauli-ravfogel/rlace-icml** (RLACE) - Advanced concept erasure
3. **uclanlp/awesome-fairness-papers** - Stay current with literature

### For Production
1. **Trusted-AI/AIF360** - Production-ready fairness toolkit
2. **marcotcr/checklist** - Behavioral testing
3. **newsmediabias/MBIAS** - Pre-debiased LLM

### For Adversarial Robustness
1. **thunlp/TAADpapers** - Paper collection
2. **marcotcr/checklist** - Testing framework

---

## Installation Examples

### INLP
```bash
git clone https://github.com/shauli-ravfogel/nullspace_projection
cd nullspace_projection
pip install -r requirements.txt
```

### Bias Bench
```bash
git clone https://github.com/McGill-NLP/bias-bench
cd bias-bench
pip install -e .
```

### AIF360
```bash
pip install aif360
```

### CheckList
```bash
pip install checklist
```

---

**Last Updated**: November 2025
**Maintained by**: ELECTRA NLP Artifact Analysis Project
