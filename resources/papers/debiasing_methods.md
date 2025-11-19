# Debiasing Methods for Language Models

Research papers on bias detection and mitigation techniques in NLP.

---

## 2024

### MBIAS: Mitigating Bias in Large Language Models While Retaining Context
- **Venue**: arXiv 2024
- **Link**: https://arxiv.org/html/2405.11290v2
- **Key Contribution**: Fine-tuned LLM designed to enhance safety while retaining contextual accuracy
- **Performance**: >30% reduction in bias overall, >90% in specific demographics
- **Model**: Available on HuggingFace as `newsmediabias/MBIAS`
- **Application**: Contextual text generation with reduced toxicity
- **Code**: HuggingFace model card includes usage examples

### Towards Trustworthy LLMs: A Review on Debiasing and Dehallucinating
- **Venue**: Artificial Intelligence Review (Springer) 2024
- **Link**: https://link.springer.com/article/10.1007/s10462-024-10896-y
- **Key Contribution**: Comprehensive taxonomy of mitigation methods for both bias and hallucination
- **Scope**: Reviews 100+ papers, covers latest techniques through 2024
- **Topics**:
  - Stand-alone debiasing components
  - Guardrail models for AI safety
  - Adapter-based approaches
- **Citation Count**: Rapidly growing (published 2024)

### Bias and De-Biasing in Large Language Models
- **Venue**: *SEM 2024
- **Link**: https://aclanthology.org/2024.starsem-1.30.pdf
- **Key Contribution**: Survey of bias sources and debiasing approaches in LLMs
- **Coverage**: Gender bias, racial bias, representational bias
- **Methods Reviewed**: Pre-training, fine-tuning, and inference-time debiasing

### Open-DeBias: Toward Mitigating Open-Set Bias in Language Models
- **Venue**: arXiv 2024
- **Link**: https://arxiv.org/html/2509.23805
- **Key Contribution**: Addresses biases beyond pre-defined categories (open-set bias)
- **Innovation**: Generalizable debiasing without category-specific data
- **Application**: Real-world deployment where all bias categories aren't known

### Mitigating Social Biases in Language Models through Unlearning
- **Venue**: EMNLP 2024
- **Link**: https://arxiv.org/html/2406.13551v1
- **Authors**: Vector Institute
- **Key Contribution**: Machine unlearning approaches for bias mitigation
- **Methods**:
  - Negation via Task Vectors (25.5% bias reduction for LLaMA-2)
  - PCGU approach (up to 40% for OPT models)
- **Code**: https://github.com/VectorInstitute/bias-mitigation-unlearning
- **Key Finding**: Task vectors comparable to DPO with minimal perplexity increase

---

## 2023

### Saliency Guided Debiasing: Detecting and Mitigating Biases Using Feature Attribution
- **Venue**: Neurocomputing (ScienceDirect) 2023
- **Link**: https://www.sciencedirect.com/science/article/abs/pii/S0925231223009748
- **Key Contribution**: Uses saliency techniques to identify and down-weight biased features
- **Methods**: Data augmentation + saliency-based feature attribution
- **Application**: Natural language generation
- **Code**: Available upon request from authors

### From Measurement to Mitigation: Exploring Transferability of Debiasing Approaches
- **Venue**: arXiv 2023 (Maltese LMs)
- **Link**: https://arxiv.org/html/2507.03142
- **Key Contribution**: Studies transferability of debiasing methods to low-resource languages
- **Application**: Gender bias in Maltese language models
- **Key Finding**: Not all debiasing methods transfer across languages

### Continual Debiasing: A Bias Mitigation Framework for NLU Systems
- **Venue**: Expert Systems with Applications (ScienceDirect) 2024/2025
- **Link**: https://www.sciencedirect.com/science/article/abs/pii/S0957417425002155
- **Key Contribution**: Framework for continuous bias mitigation in deployed systems
- **Innovation**: Handles evolving biases over time
- **Application**: Production NLU systems

---

## 2022

### An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models
- **Venue**: ACL 2022
- **Authors**: Nicholas Meade, Elinor Poole-Dayan, Siva Reddy (McGill-NLP)
- **Link**: https://mcgill-nlp.github.io/bias-bench/
- **Code**: https://github.com/McGill-NLP/bias-bench
- **Key Contribution**: Systematic evaluation of debiasing techniques across 3 benchmarks
- **Benchmarks**: StereoSet, CrowS-Pairs, SEAT
- **Methods Evaluated**:
  - Dropout
  - INLP
  - CDA
  - Sentence debias
- **Key Finding**: No single method consistently outperforms across all benchmarks
- **Impact**: Gold standard for evaluating new debiasing methods

### Linear Adversarial Concept Erasure (RLACE)
- **Venue**: ICML 2022
- **Authors**: Shauli Ravfogel, Yoav Goldberg, et al.
- **Link**: https://arxiv.org/abs/2201.12091
- **Code**: https://github.com/shauli-ravfogel/rlace-icml
- **Key Contribution**: Identifies rank-k subspace for concept neutralization
- **Method**: Minimax game between predictor and projection matrix
- **Application**: Gender bias removal, POS information removal
- **Impact**: 500+ citations, foundation for later work

---

## 2021

### Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (INLP)
- **Venue**: ACL 2020 (Heavily cited 2021-2024)
- **Authors**: Shauli Ravfogel, Yoav Goldberg, et al.
- **Link**: https://aclanthology.org/2020.acl-main.647/
- **Code**: https://github.com/shauli-ravfogel/nullspace_projection
- **Key Contribution**: Iterative removal of information via nullspace projection
- **Method**: Train classifiers, project to null space repeatedly
- **Application**: Gender, race, other protected attributes
- **Impact**: 800+ citations, widely used baseline

### Mind the Trade-off: Debiasing NLU Models without Degrading Performance
- **Venue**: ACL 2020 (Highly relevant through 2024)
- **Link**: https://aclanthology.org/2020.acl-main.770.pdf
- **Key Contribution**: Maintains in-distribution performance while debiasing
- **Methods**: Learned-mixin, confidence regularization
- **Application**: NLI, reading comprehension
- **Key Finding**: Trade-off between bias mitigation and task performance is addressable

---

## Ensemble and Product-of-Experts Methods

### Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases
- **Authors**: Christopher Clark, et al. (University of Washington)
- **Link**: Paper details in chrisc36/debias repository
- **Code**: https://github.com/chrisc36/debias
- **Key Contribution**: Train biased model, ensemble with main model
- **Tasks**: SQuAD, MNLI, VQA, TriviaQA-CP
- **Methods**:
  - Hypothesis-only baseline for NLI
  - Question-only baseline for QA
  - Product of Experts
  - Learned-mixin adaptive variant
- **Implementation**: Both TensorFlow and PyTorch
- **Impact**: Foundation for ensemble debiasing approaches

---

## Key Techniques Summary

### Data-Level Debiasing
1. **Counterfactual Data Augmentation (CDA)**: Flip protected attributes
2. **Balanced Sampling**: Equal representation across groups
3. **Hard Negative Mining**: Focus on challenging examples

### Model-Level Debiasing
1. **INLP**: Iterative nullspace projection
2. **RLACE/LEACE**: Adversarial concept erasure
3. **Adapter-Based**: Add debiasing adapters to frozen models
4. **Ensemble Methods**: Product of Experts, Learned-mixin

### Inference-Time Debiasing
1. **Guardrail Models**: Monitor inputs/outputs
2. **Prompt Engineering**: Debiasing prompts
3. **Output Filtering**: Post-process predictions

### Evaluation Methods
1. **Intrinsic Metrics**: StereoSet, CrowS-Pairs, SEAT
2. **Extrinsic Metrics**: Downstream task fairness
3. **Trade-off Analysis**: Bias reduction vs. performance

---

## Comparison of Methods

| Method | Type | Pros | Cons | Best For |
|--------|------|------|------|----------|
| INLP | Model-level | Simple, effective | May degrade performance | Protected attributes |
| RLACE/LEACE | Model-level | Theoretically grounded | Requires target concept | Concept erasure |
| CDA | Data-level | Improves robustness | May introduce new biases | Data augmentation |
| Ensemble | Model-level | Maintains performance | Requires training biased model | Known artifacts |
| Adapters | Model-level | Doesn't modify base model | Additional parameters | Production systems |
| Unlearning | Model-level | Removes learned bias | Computational cost | Post-training |

---

## Research Trends (2024-2025)

1. **LLM-Specific Debiasing**: Methods for billion-parameter models
2. **Open-Set Bias**: Handling unknown bias categories
3. **Multilingual**: Transferability across languages
4. **Continual Learning**: Adapting to evolving biases
5. **Efficient Methods**: Low-compute debiasing (adapters, LoRA)

---

## Related Topics
- See `spurious_correlations.md` for artifact identification
- See `counterfactual_augmentation.md` for CDA methods
- See `causal_inference.md` for causal debiasing

---

**Last Updated**: November 2025
