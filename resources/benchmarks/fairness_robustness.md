# Fairness and Robustness Benchmarks for NLP

Evaluation datasets and frameworks for testing bias, fairness, and robustness in NLP models.

---

## Robustness Benchmarks

### 1. HANS (Heuristic Analysis for NLI Systems) ⭐⭐⭐⭐⭐
- **Year**: 2019 (Still widely used 2021-2024)
- **Paper**: Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in NLI (ACL 2019)
- **GitHub**: https://github.com/tommccoy1/hans
- **Dataset**: https://paperswithcode.com/dataset/hans
- **HuggingFace**: Available on datasets
- **Description**: 30,000 synthetically generated examples testing three fallible syntactic heuristics
- **Heuristics Tested**:
  1. Lexical overlap heuristic
  2. Subsequence heuristic
  3. Constituent heuristic
- **Task**: Natural Language Inference (NLI)
- **Key Feature**: Models relying on heuristics fail dramatically (near 0% accuracy)
- **Recent Usage**: Still primary benchmark for NLI robustness in 2024
- **Citation Count**: 1000+
- **Use Case**: Diagnosing spurious correlation reliance in NLI models

### 2. CheckList ⭐⭐⭐⭐⭐
- **Year**: 2020 (Extended through 2024)
- **Paper**: Beyond Accuracy: Behavioral Testing of NLP Models (ACL 2020)
- **GitHub**: https://github.com/marcotcr/checklist
- **Description**: Task-agnostic behavioral testing framework
- **Features**:
  - Matrix of linguistic capabilities × test types
  - Automatic test case generation
  - Perturbation capabilities
  - Templates for various tests
- **Capabilities Tested**:
  - Vocabulary (typos, synonyms)
  - Taxonomy (named entities)
  - Robustness (negation, paraphrase)
  - NER capability
  - Temporal understanding
  - Coref, SRL, Logic
- **Tasks**: Sentiment, QA, textual entailment, etc.
- **Citation Count**: 1000+
- **Extensions**: AutoTestForge (2025) builds on CheckList
- **Use Case**: Comprehensive behavioral testing across capabilities

### 3. Contrast Sets
- **Year**: 2020 (Concept still active 2021-2024)
- **Papers**: Multiple papers implementing contrast sets for different tasks
- **Description**: Minimal perturbations that should change model predictions
- **Tasks**: NLI, Reading Comprehension, Sentiment Analysis
- **Key Idea**: If model is robust, should handle minimal meaning-changing edits
- **Use Case**: Testing if models rely on spurious cues vs. true understanding
- **Related**: Counterfactual evaluation

---

## Fairness Benchmarks

### 4. BBQ (Bias Benchmark for Question Answering) ⭐⭐⭐⭐⭐
- **Year**: 2022 (Current standard through 2024)
- **Paper**: BBQ: A Hand-Built Bias Benchmark for Question Answering
- **Link**: https://arxiv.org/abs/2110.08193
- **Semantic Scholar**: https://www.semanticscholar.org/paper/7d5c661fa9a4255ee087e861f820564ea2e2bd6b
- **Description**: Hand-built bias benchmark across 9 social dimensions
- **Social Dimensions**:
  1. Age
  2. Disability
  3. Gender identity
  4. Nationality
  5. Physical appearance
  6. Race/ethnicity
  7. Religion
  8. Socioeconomic status
  9. Sexual orientation
- **Evaluation Levels**:
  1. Under-informative context: Tests bias strength
  2. Adequately informative context: Tests if bias overrides correct answer
- **Context**: U.S. English-speaking contexts
- **Citation Count**: 200+
- **Recent Usage**: Standard in 2024 LLM fairness evaluations
- **Use Case**: Measuring social biases in QA systems

### 5. WinoBias ⭐⭐⭐
- **Year**: 2018 (Still relevant but aging)
- **Paper**: Gender Bias in Coreference Resolution
- **Link**: https://www.catalyzex.com/s/Winobias
- **Description**: Tests gender bias in coreference resolution
- **Format**: Stereotypical and anti-stereotypical contexts with occupations
- **Task**: Associating gender pronouns with occupations
- **Limitation (2024)**: Modern LLMs rely more on semantics than syntax
- **Status**: Being superseded by newer benchmarks but still cited
- **Citation Count**: 500+
- **Use Case**: Gender bias in coreference

### 6. BOLD (Bias in Open-Ended Language Generation Dataset) ⭐⭐⭐⭐
- **Year**: 2021
- **Paper**: BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation (FAccT 2021)
- **Description**: Tests bias in open-ended language generation
- **Domains**: Profession, gender, race, religion, political ideology
- **Format**: Wikipedia prompts across demographic groups
- **Limitation**: Template-based, limited syntactic diversity
- **Citation Count**: 300+
- **Use Case**: Bias in generation tasks

### 7. StereoSet ⭐⭐⭐⭐
- **Year**: 2020 (Widely used through 2024)
- **Paper**: StereoSet: Measuring stereotypical bias in pretrained language models
- **Description**: Measures stereotypical biases across 4 domains
- **Domains**: Gender, profession, race, religion
- **Format**: Intrasentence and intersentence contexts
- **Metrics**: Language Modeling Score (LMS) and Stereotypical Score (SS)
- **Citation Count**: 500+
- **Used In**: bias-bench benchmark suite
- **Use Case**: Intrinsic bias measurement

### 8. CrowS-Pairs ⭐⭐⭐
- **Year**: 2020
- **Paper**: CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked LMs
- **Description**: 1,508 examples covering 9 bias types
- **Bias Types**: Race, gender, sexual orientation, religion, age, nationality, disability, physical appearance, socioeconomic status
- **Format**: Sentence pairs (stereotypical vs. anti-stereotypical)
- **Citation Count**: 400+
- **Used In**: bias-bench benchmark suite
- **Use Case**: Social bias measurement in MLMs

---

## Adversarial Benchmarks

### 9. Adversarial SQuAD ⭐⭐⭐⭐
- **Year**: 2017 (Foundational, still influential)
- **Paper**: Adversarial Examples for Evaluating Reading Comprehension Systems (EMNLP 2017)
- **Author**: Robin Jia
- **GitHub**: https://github.com/robinjia/adversarial-squad
- **Description**: Adversarial examples for reading comprehension
- **Method**: Add distracting sentences to passages
- **Impact**: Showed vulnerability of state-of-the-art models
- **Citation Count**: 1000+
- **Use Case**: Adversarial robustness for QA

### 10. ANLI (Adversarial NLI) ⭐⭐⭐⭐
- **Year**: 2020
- **Paper**: Adversarial NLI: A New Benchmark for Natural Language Understanding
- **Description**: Human-in-the-loop adversarial NLI dataset
- **Rounds**: Three rounds (R1, R2, R3) with increasing difficulty
- **Size**: 162,865 examples
- **Key Feature**: Humans create examples models get wrong
- **Citation Count**: 400+
- **Use Case**: Challenging NLI evaluation

---

## Compound Benchmarks and Suites

### 11. GLUE and SuperGLUE
- **Years**: 2018, 2019 (Baseline reference)
- **Status**: Saturated by 2022, but still used for comparison
- **Tasks**: Multiple NLU tasks
- **Limitations**: Models achieve near-human performance
- **Use Case**: General NLU baseline

### 12. Fair-LLM-Benchmark (2024) ⭐⭐⭐
- **Year**: 2024
- **GitHub**: https://github.com/i-gallegos/Fair-LLM-Benchmark
- **Description**: Fairness evaluation for modern LLMs
- **Models Tested**: Gemini 1.5 Pro, LLaMA 3 70B, Claude 3 Opus, GPT-4o
- **Scenarios**:
  - Gender bias in occupations
  - Gender, age, racial bias in crime scenarios
- **Features**: Comprehensive fairness metrics
- **Use Case**: LLM fairness evaluation (2024 standard)

---

## Evaluation Metrics and Frameworks

### Fairness Metrics
1. **Demographic Parity**: Equal positive rate across groups
2. **Equalized Odds**: Equal TPR and FPR across groups
3. **Calibration**: Predicted probabilities match actual frequencies
4. **Individual Fairness**: Similar individuals treated similarly

### Robustness Metrics
1. **Accuracy on OOD Data**: Performance on out-of-distribution examples
2. **Certified Accuracy**: Provably robust predictions
3. **Attack Success Rate**: Percentage of successful adversarial examples
4. **Consistency**: Agreement on paraphrases/perturbations

### Available Tools
- **AIF360** (IBM): 70+ fairness metrics
- **Fairlearn** (Microsoft): Fairness assessment and mitigation
- **bias-bench**: Standardized evaluation across StereoSet, CrowS-Pairs, SEAT
- **CheckList**: Behavioral testing framework

---

## Recent Benchmark Trends (2023-2024)

### LLM-Specific Benchmarks
1. **Persona-based evaluation**: Testing persona-induced biases
2. **Uncertainty-aware fairness**: Combining fairness with calibration
3. **Open-set bias**: Evaluating beyond pre-defined categories
4. **Multilingual fairness**: Cross-lingual bias evaluation

### Emerging Areas
1. **Causal evaluation**: Using causal inference for bias measurement
2. **Intersectional fairness**: Multiple protected attributes
3. **Dynamic evaluation**: Adapting to model improvements
4. **Safety benchmarks**: Combining fairness, toxicity, safety

---

## Benchmark Comparison

| Benchmark | Task | Year | Size | Bias Type | Status (2024) |
|-----------|------|------|------|-----------|---------------|
| HANS | NLI | 2019 | 30k | Syntactic | Active |
| BBQ | QA | 2022 | - | Social (9 dims) | Standard |
| WinoBias | Coref | 2018 | 3k | Gender | Aging |
| BOLD | Generation | 2021 | - | Multi-domain | Active |
| StereoSet | MLM | 2020 | 16k | 4 domains | Active |
| CrowS-Pairs | MLM | 2020 | 1.5k | 9 types | Active |
| CheckList | Multi-task | 2020 | Generated | Behavioral | Active |
| Adversarial SQuAD | QA | 2017 | - | Adversarial | Reference |

---

## Recommended Benchmark Suites

### For NLI Models
1. **HANS** - Syntactic heuristics
2. **ANLI** - Adversarial examples
3. **Contrast sets** - Minimal perturbations
4. **BBQ** - Social bias

### For QA Models
1. **Adversarial SQuAD** - Robustness
2. **BBQ** - Social bias
3. **CheckList** - Behavioral capabilities

### For Language Models
1. **StereoSet** - Stereotypical bias
2. **CrowS-Pairs** - Social bias
3. **BOLD** - Generation bias
4. **BBQ** - QA bias

### For Production Systems
1. **CheckList** - Comprehensive testing
2. **AIF360 metrics** - Fairness evaluation
3. **Custom adversarial tests** - Domain-specific robustness

---

## Accessing Benchmarks

### HuggingFace Datasets
```python
from datasets import load_dataset

# HANS
hans = load_dataset("hans")

# WinoBias
winobias = load_dataset("wino_bias")

# StereoSet
stereoset = load_dataset("stereoset")
```

### Direct Downloads
- **BBQ**: Via paper repository
- **CheckList**: Via GitHub (marcotcr/checklist)
- **HANS**: Via GitHub (tommccoy1/hans)

---

**Last Updated**: November 2025
