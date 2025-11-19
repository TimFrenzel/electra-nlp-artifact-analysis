# SNLI-Specific Artifact and Bias Research

Research specifically focused on artifacts and biases in the SNLI (Stanford Natural Language Inference) dataset.

---

## Dataset Overview

**SNLI (Stanford Natural Language Inference)**
- **Published**: 2015 (Bowman et al.)
- **Official URL**: https://nlp.stanford.edu/projects/snli/
- **Size**: 570,000 sentence pairs
- **Task**: 3-way classification (entailment, contradiction, neutral)
- **Source**: Image captions from Flickr30k
- **Labels**: Crowdsourced annotations
- **Known Issues**: Contains ~0.7% invalid labels (label=-1)

---

## Known Artifacts in SNLI

### 1. Hypothesis-Only Bias ⭐⭐⭐⭐⭐

**Finding**: Models can achieve 67% accuracy using ONLY the hypothesis (should be 33% random baseline)

**Key Papers**:

#### Annotation Artifacts in Natural Language Inference Data (NAACL 2018)
- **Authors**: Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel Bowman, Noah A. Smith
- **Link**: https://aclanthology.org/N18-2017/
- **Key Finding**: Identified strong hypothesis-only baseline performance
- **Citations**: 600+
- **Impact**: Seminal paper exposing SNLI artifacts

**Artifacts Identified**:
- **Negation**: "never", "nobody", "no" → contradiction
- **Vague words**: "some", "various" → neutral
- **Specific details**: Names, dates → neutral
- **Generic statements**: → entailment

---

### 2. Lexical Overlap Bias

**Finding**: High word overlap between premise and hypothesis correlates with entailment

**Papers**:

#### Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in NLI (ACL 2019)
- **Authors**: R. Thomas McCoy, Ellie Pavlick, Tal Linzen
- **Link**: https://aclanthology.org/P19-1334/
- **HANS Benchmark**: Specifically tests SNLI-trained models
- **Citations**: 1000+
- **Key Finding**: Models trained on SNLI fail dramatically on HANS (near 0% on some subsets)

**Heuristics**:
1. **Lexical overlap**: High overlap → entailment
2. **Subsequence**: If hypothesis is subsequence of premise → entailment
3. **Constituent**: If premise constituent matches hypothesis → entailment

---

### 3. Length Bias

**Finding**: Shorter hypotheses tend to be labeled as entailment

**Research**:
- Controlled length experiments show correlation between length and label distribution
- Models learn this spurious signal

---

### 4. Contradiction-Word Bias

**Papers**:

#### Towards Robustifying NLI Models Against Lexical Dataset Biases (ACL 2020)
- **Authors**: Xiang Zhou, Mohit Bansal
- **Link**: https://www.semanticscholar.org/paper/8cbb254809749fbe00cbf224daba059e345891eb
- **Citations**: 150+
- **Focus**: Contradiction words and lexical biases in SNLI

**Contradiction Indicators**:
- Negation words: "not", "never", "nobody"
- Contradiction verbs: "deny", "reject"
- Opposite adjectives: "different", "opposite"

---

## Debiasing SNLI Models

### Product of Experts Approach

#### Don't Take the Easy Way Out (University of Washington)
- **Authors**: Christopher Clark, Mark Yatskar, Luke Zettlemoyer
- **Approach**: Train hypothesis-only model, ensemble with full model
- **Code**: https://github.com/chrisc36/debias
- **Application**: Specifically tested on MNLI (similar to SNLI)
- **Result**: Improved out-of-distribution performance

### Learned-Mixin Variant
- **Paper**: Clark et al. 2019
- **Method**: Adaptive product of experts
- **SNLI Application**: Downweight examples hypothesis-only model gets correct

---

### Data Augmentation

#### Counterfactual Data Augmentation for SNLI
- **Method**: Flip entities, negations to create counterfactuals
- **Goal**: Reduce reliance on spurious correlations
- **Example**:
  - Original: "A man is playing guitar" → "A man is making music" (entailment)
  - Counterfactual: "A woman is playing guitar" → "A woman is making music" (entailment)

---

### Adversarial Training

#### Adversarial NLI (ANLI)
- **Year**: 2020
- **Authors**: Nie et al.
- **Approach**: Human-in-the-loop adversarial examples
- **Relationship to SNLI**: Specifically designed to be harder than SNLI
- **Result**: Models trained on SNLI perform poorly on ANLI (~20% accuracy)

---

## SNLI Evaluation Benchmarks

### 1. HANS (Heuristic Analysis for NLI)
- **Purpose**: Test syntactic heuristics learned from SNLI
- **Size**: 30,000 examples
- **Format**: Templates targeting specific heuristics
- **Key Result**: BERT trained on SNLI gets near 0% on some HANS subsets
- **Link**: https://github.com/tommccoy1/hans

### 2. SNLI Hard Subset
- **Description**: Adversarially filtered examples where hypothesis-only baseline fails
- **Purpose**: More challenging evaluation
- **Source**: Various papers create custom hard subsets

### 3. Stress Tests
- **Approach**: Minimal edits to SNLI examples
- **Types**:
  - Negation addition/removal
  - Entity swapping
  - Paraphrasing
- **Goal**: Test model robustness to surface changes

---

## SNLI-Specific Implementation Tips

### 1. Always Filter Invalid Labels
```python
# SNLI has ~0.7% examples with label=-1 (no consensus)
dataset = dataset.filter(lambda x: x["label"] != -1)
```

### 2. Hypothesis-Only Baseline
```python
# Test if your model relies on hypothesis-only bias
def hypothesis_only_baseline(examples):
    # Train model on only hypothesis text
    return tokenizer(examples["hypothesis"], ...)
```

Expected hypothesis-only accuracy:
- Random: 33.3%
- Biased baseline: ~67%
- Good model on full data: ~90%
- **Your debiased model**: Should reduce hypothesis-only to <50%

### 3. Stratified Evaluation
```python
# Evaluate separately on different premise lengths
short_premises = dataset.filter(lambda x: len(x["premise"].split()) < 10)
long_premises = dataset.filter(lambda x: len(x["premise"].split()) >= 20)
```

### 4. Lexical Overlap Analysis
```python
def compute_overlap(premise, hypothesis):
    p_words = set(premise.lower().split())
    h_words = set(hypothesis.lower().split())
    return len(p_words & h_words) / len(h_words)

# Group by overlap and check accuracy
high_overlap = dataset.filter(lambda x: compute_overlap(...) > 0.5)
low_overlap = dataset.filter(lambda x: compute_overlap(...) < 0.2)
```

---

## Expected ELECTRA Performance on SNLI

### Baseline (No Debiasing)
- **Accuracy**: 89-91%
- **Hypothesis-only**: 65-70%
- **HANS accuracy**: 50-60%

### After Debiasing (Target)
- **Accuracy**: 88-90% (slight drop acceptable)
- **Hypothesis-only**: 40-50% (lower is better)
- **HANS accuracy**: 65-75% (higher is better)

---

## Recommended Experimental Protocol for ELECTRA-SNLI

1. **Train Baseline**:
   - Full SNLI training set
   - Standard hyperparameters
   - Track validation accuracy

2. **Analyze Artifacts**:
   - Hypothesis-only accuracy
   - Lexical overlap correlation
   - Error analysis on HANS
   - Length bias analysis

3. **Apply Mitigation**:
   - Option A: Product of Experts (hypothesis-only + full model)
   - Option B: Dataset Cartography (focus on hard examples)
   - Option C: Adversarial training (add HANS examples)

4. **Evaluate**:
   - In-distribution: SNLI test set
   - Out-of-distribution: HANS
   - Hypothesis-only baseline
   - Contrast sets

---

## Key Papers Timeline

- **2015**: SNLI dataset released (Bowman et al.)
- **2018**: Annotation artifacts identified (Gururangan et al.)
- **2019**: HANS benchmark (McCoy et al.)
- **2019**: Product of Experts debiasing (Clark et al.)
- **2020**: Dataset Cartography (Swayamdipta et al.)
- **2020**: Robustifying NLI (Zhou & Bansal)
- **2022**: Continued bias analysis in LLMs

---

## Citation

### SNLI Dataset
```bibtex
@inproceedings{bowman2015snli,
  title={A large annotated corpus for learning natural language inference},
  author={Bowman, Samuel R and Angeli, Gabor and Potts, Christopher and Manning, Christopher D},
  booktitle={EMNLP},
  year={2015}
}
```

### Annotation Artifacts
```bibtex
@inproceedings{gururangan2018annotation,
  title={Annotation Artifacts in Natural Language Inference Data},
  author={Gururangan, Suchin and Swayamdipta, Swabha and Levy, Omer and Schwartz, Roy and Bowman, Samuel and Smith, Noah A},
  booktitle={NAACL},
  year={2018}
}
```

---

## Related Datasets

- **MultiNLI (MNLI)**: Similar to SNLI but multi-genre (433k pairs)
- **ANLI**: Adversarially collected NLI (162k pairs, harder)
- **HANS**: Diagnostic dataset for heuristics (30k examples)
- **SNLI-hard**: Filtered hard subset

---

**Last Updated**: November 2025
**Relevance**: Critical for ELECTRA NLP Artifact Analysis project
