# Dataset Artifacts in Natural Language Inference: An Investigation Using ELECTRA

**Professional Research Project - Technical Report**

**Author**: [Your Name]
**Date**: November 2025
**Project**: ELECTRA NLP Artifact Analysis

---

## Abstract

*[150-200 words summarizing the entire project]*

Natural Language Inference (NLI) models have achieved high performance on benchmarks like SNLI, but recent research suggests these models may exploit spurious correlations (dataset artifacts) rather than learning robust linguistic reasoning. This study investigates whether ELECTRA-small genuinely understands natural language inference or relies on superficial patterns in the SNLI dataset. We conduct a systematic analysis identifying hypothesis-only bias, lexical overlap correlations, and length biases. Our baseline ELECTRA model achieves [XX]% accuracy on SNLI, but a hypothesis-only baseline reaches [XX]% (vs. 33% random), indicating severe artifact exploitation. We implement [mitigation method] to reduce reliance on spurious features, resulting in [XX]% improvement on out-of-distribution evaluation while maintaining [XX]% in-distribution performance. Our findings demonstrate that [key conclusion] and highlight the importance of artifact-aware evaluation for robust NLP systems.

**Keywords**: Dataset artifacts, natural language inference, ELECTRA, bias mitigation, spurious correlations

---

## 1. Introduction

### 1.1 Motivation

Natural Language Inference (NLI) is a fundamental task in natural language understanding, requiring models to determine whether a hypothesis logically follows from, contradicts, or is neutral with respect to a given premise. The Stanford Natural Language Inference (SNLI) dataset [1] has become a standard benchmark for evaluating NLI systems, with modern pre-trained models like BERT and ELECTRA achieving human-level accuracy (~90%).

However, recent research [2-4] has revealed that NLI datasets contain systematic biases and spurious correlations—dataset artifacts—that allow models to achieve high performance without genuine language understanding. For instance, Gururangan et al. [2] demonstrated that models trained only on hypotheses (ignoring premises entirely) can achieve 67% accuracy on SNLI, far exceeding the 33% random baseline. This suggests models exploit superficial patterns rather than learning robust reasoning.

### 1.2 Research Questions

This study investigates the following research questions:

1. **RQ1**: To what extent does ELECTRA-small exploit dataset artifacts in SNLI?
2. **RQ2**: What types of spurious correlations does the model rely on (hypothesis-only bias, lexical overlap, length bias)?
3. **RQ3**: Can we mitigate these artifacts while maintaining competitive in-distribution performance?
4. **RQ4**: How do debiasing interventions affect out-of-distribution generalization?

### 1.3 Contributions

This work makes the following contributions:

1. **Systematic artifact analysis** of ELECTRA-small on SNLI, quantifying hypothesis-only bias, lexical overlap effects, and length correlations
2. **Comprehensive error characterization** identifying patterns in model failures
3. **Implementation and evaluation** of [mitigation method] for reducing artifact reliance
4. **Empirical demonstration** that artifact mitigation improves robustness to distribution shift

### 1.4 Organization

The remainder of this report is organized as follows: Section 2 reviews related work on dataset artifacts and debiasing methods. Section 3 describes our experimental methodology and baseline model setup. Section 4 presents our artifact analysis findings (Part 1). Section 5 details our mitigation approach and evaluation (Part 2). Section 6 discusses implications and limitations. Section 7 concludes.

---

## 2. Background and Related Work

### 2.1 Natural Language Inference

Natural Language Inference (NLI), also known as Recognizing Textual Entailment (RTE), is a three-way classification task:
- **Entailment**: Hypothesis follows logically from premise
- **Contradiction**: Hypothesis contradicts premise
- **Neutral**: Hypothesis neither follows nor contradicts premise

**Example**:
- Premise: "A man plays guitar in the park."
- Hypothesis (Entailment): "A person is making music outdoors."
- Hypothesis (Contradiction): "Nobody is playing an instrument."
- Hypothesis (Neutral): "The man is a professional musician."

### 2.2 SNLI Dataset

The Stanford Natural Language Inference (SNLI) dataset [1] contains 570,000 sentence pairs labeled by crowdworkers. Premises are derived from Flickr30k image captions, and workers wrote corresponding hypotheses for each label class. The dataset has become the standard NLI benchmark, with models achieving 89-91% accuracy.

**Known Issues**:
- ~0.7% invalid labels (label=-1, no consensus)
- Annotation artifacts from crowdsourcing process
- Hypothesis-only biases enabling 67% accuracy [2]

### 2.3 Dataset Artifacts

**Definition**: Dataset artifacts are spurious correlations in training data that allow models to achieve high performance through shortcut learning rather than genuine task understanding.

**Key Research**:

1. **Annotation Artifacts in NLI** [2]
   - Gururangan et al. (NAACL 2018)
   - Discovered hypothesis-only baseline achieves 67% on SNLI
   - Identified lexical cues: negation words → contradiction, vague words → neutral

2. **Syntactic Heuristics** [3]
   - McCoy et al. (ACL 2019)
   - Created HANS diagnostic dataset
   - BERT trained on SNLI fails catastrophically on HANS (near 0% on some subsets)
   - Heuristics: lexical overlap → entailment, subsequence → entailment

3. **Right for the Wrong Reasons** [4]
   - Models rely on surface patterns rather than compositional reasoning
   - High performance on in-distribution data masks poor generalization

### 2.4 Debiasing Methods

**Product of Experts (PoE)** [5]:
- Train biased model (e.g., hypothesis-only) alongside full model
- Ensemble predictions to downweight artifact-reliant examples
- Improves out-of-distribution performance

**Dataset Cartography** [6]:
- Analyze training dynamics (confidence, variability, correctness)
- Identify "easy" examples that may rely on artifacts
- Focus training on "hard" examples for robust learning

**Adversarial Training**:
- Augment data with counterfactual examples
- Force model to rely on premise-hypothesis interaction
- Examples: entity swapping, negation insertion

**Confidence Regularization**:
- Penalize overconfident predictions on biased examples
- Encourage model to distribute probability mass more uniformly

### 2.5 ELECTRA Architecture

ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) [7] is a pre-trained transformer model using replaced token detection instead of masked language modeling. ELECTRA-small has 14M parameters and achieves competitive performance with significantly less compute than BERT.

**Advantages for this study**:
- Efficient fine-tuning (suitable for limited compute)
- Strong baseline performance on SNLI (~89%)
- Well-studied in debiasing literature

---

## 3. Methodology

### 3.1 Experimental Setup

**Model**: ELECTRA-small-discriminator (google/electra-small-discriminator)
- Parameters: 14M
- Hidden size: 256
- Attention heads: 4
- Layers: 12

**Dataset**: SNLI (Stanford Natural Language Inference)
- Training: ~550,000 pairs (after filtering label=-1)
- Validation: ~10,000 pairs
- Test: ~10,000 pairs
- Preprocessing: Filter invalid labels, truncate to 128 tokens

**Computational Resources**:
- Platform: Google Colab with A100 GPU
- Training time: ~1-3 hours for baseline (3 epochs)
- Batch size: 32
- Learning rate: 2e-5
- Mixed precision (FP16) for efficiency

**Evaluation Metrics**:
- **In-distribution**: Accuracy on SNLI test set
- **Out-of-distribution**: [HANS, contrast sets, or custom evaluation]
- **Artifact measures**: Hypothesis-only accuracy, performance stratified by overlap/length

### 3.2 Baseline Training

We fine-tune ELECTRA-small on SNLI for 3 epochs using standard hyperparameters from prior work [starter code citation]. Training employs:
- AdamW optimizer
- Linear warmup (10% of steps)
- Cross-entropy loss
- Gradient clipping (max norm 1.0)

**Implementation**: We build on the fp-dataset-artifacts starter code [8], which provides proper SNLI preprocessing, invalid label filtering, and metric computation.

### 3.3 Analysis Protocol (Part 1)

We conduct systematic artifact analysis:

1. **Hypothesis-Only Baseline**
   - Train ELECTRA-small using only hypothesis text (no premise)
   - Compare to random baseline (33%) and expected biased baseline (~67%)
   - Quantify artifact severity

2. **Lexical Overlap Analysis**
   - Compute Jaccard similarity between premise and hypothesis
   - Stratify validation set by overlap level (0-20%, 20-40%, etc.)
   - Test correlation: high overlap → entailment prediction

3. **Length Bias Analysis**
   - Stratify by hypothesis length (words)
   - Analyze label distribution and accuracy across length bins
   - Test if shorter hypotheses correlate with specific labels

4. **Error Characterization**
   - Identify systematic failure patterns
   - Analyze confusion matrix
   - Sample representative errors for qualitative analysis

### 3.4 Mitigation Strategy (Part 2)

*[Fill in after implementing Part 2]*

Based on Part 1 findings, we implement [chosen mitigation method]:
- Rationale: [why this method addresses identified artifacts]
- Implementation details: [architecture, hyperparameters, training procedure]
- Expected outcomes: [hypotheses about performance changes]

### 3.5 Evaluation

We evaluate on:
1. **SNLI test set** (in-distribution)
2. **[OOD dataset]** (out-of-distribution generalization)
3. **Hypothesis-only baseline** (artifact reliance measure)
4. **Stratified subsets** (overlap, length bins)

---

## 4. Part 1: Artifact Analysis

### 4.1 Baseline Performance

*[Fill in with actual results from colab_training.ipynb]*

Our baseline ELECTRA-small model achieves the following performance on SNLI:

| Metric | Value |
|--------|-------|
| Validation Accuracy | XX.X% |
| Test Accuracy | XX.X% |
| Entailment F1 | XX.X% |
| Neutral F1 | XX.X% |
| Contradiction F1 | XX.X% |

This performance is consistent with prior work [citation] and confirms successful baseline training.

**Table 1**: Baseline ELECTRA-small performance on SNLI test set.

### 4.2 Hypothesis-Only Bias

*[Fill in with results from colab_analysis_part1.ipynb]*

We train a hypothesis-only model using identical architecture but providing only hypothesis text as input. Results:

| Model | Accuracy |
|-------|----------|
| Random Baseline | 33.3% |
| Hypothesis-Only | XX.X% |
| Full Model (Baseline) | XX.X% |

**Analysis**: The hypothesis-only model achieves [XX]% accuracy, indicating [SEVERE/MODERATE/MILD] artifact exploitation. This [XX]pp improvement over random demonstrates that SNLI hypotheses contain strong label-predictive signals independent of premises.

**Per-Class Analysis**:
- Entailment: [XX]% precision (hypothesis-only)
- Neutral: [XX]% precision (hypothesis-only)
- Contradiction: [XX]% precision (hypothesis-only)

**Interpretation**: [Which class is most predictable from hypothesis alone? What does this suggest about annotation artifacts?]

**Figure 1**: Hypothesis-only confusion matrix *(include figure)*

### 4.3 Lexical Overlap Correlation

*[Fill in with overlap analysis results]*

We compute lexical overlap (Jaccard similarity) between premises and hypotheses, stratifying the validation set by overlap level:

| Overlap Level | Accuracy | Entailment % | Neutral % | Contradiction % |
|---------------|----------|--------------|-----------|-----------------|
| 0-20% | XX.X% | XX% | XX% | XX% |
| 20-40% | XX.X% | XX% | XX% | XX% |
| 40-60% | XX.X% | XX% | XX% | XX% |
| 60-80% | XX.X% | XX% | XX% | XX% |
| 80-100% | XX.X% | XX% | XX% | XX% |

**Table 2**: Model accuracy and label distribution across lexical overlap bins.

**Findings**:
- Accuracy [increases/remains stable/decreases] with overlap
- High overlap examples are [X]× more likely to be entailment
- This [confirms/refutes] hypothesis that model exploits superficial word matching

**Figure 2**: Lexical overlap distribution and accuracy correlation *(include figure)*

### 4.4 Length Bias

*[Fill in with length analysis results]*

| Hypothesis Length | Accuracy | Entailment % | Neutral % | Contradiction % |
|-------------------|----------|--------------|-----------|-----------------|
| 1-5 words | XX.X% | XX% | XX% | XX% |
| 6-10 words | XX.X% | XX% | XX% | XX% |
| 11-15 words | XX.X% | XX% | XX% | XX% |
| 16+ words | XX.X% | XX% | XX% | XX% |

**Table 3**: Model accuracy and label distribution across hypothesis length bins.

**Findings**:
- [Shorter/Longer] hypotheses are more likely to be [label]
- Accuracy [varies/is stable] across length bins
- This suggests [presence/absence] of length-based shortcuts

### 4.5 Error Analysis

*[Fill in with error characterization]*

We analyze [XXX] prediction errors, identifying the following patterns:

**Most Common Error Types**:
1. Entailment → Neutral: [XX] errors ([XX]%)
2. Contradiction → Neutral: [XX] errors ([XX]%)
3. Neutral → Entailment: [XX] errors ([XX]%)

**Qualitative Patterns**:
- [Pattern 1]: Examples with [characteristic] are systematically misclassified
- [Pattern 2]: Model struggles with [linguistic phenomenon]
- [Pattern 3]: [Additional pattern]

**Representative Examples**:
*[Include 2-3 illustrative error examples with analysis]*

**Figure 3**: Error distribution by type *(include confusion matrix or bar chart)*

### 4.6 Summary of Findings

Based on our systematic analysis, we identify the following artifacts in our ELECTRA-SNLI baseline:

1. **Hypothesis-only bias**: [SEVERITY] - Model achieves [XX]% using only hypothesis
2. **Lexical overlap**: [STRONG/MODERATE/WEAK] correlation with entailment predictions
3. **Length bias**: [PRESENT/ABSENT] - [Description]
4. **Failure modes**: Model struggles with [categories]

These findings motivate our mitigation strategy in Part 2.

---

## 5. Part 2: Mitigation and Evaluation

*[Fill in after implementing mitigation]*

### 5.1 Mitigation Approach

Based on Part 1 findings, we implement [chosen method]:

**Method Overview**: [Brief description]

**Why This Method**: This approach addresses the identified artifacts because [rationale]

**Implementation Details**:
- [Architecture/algorithm specifics]
- [Hyperparameters]
- [Training procedure]

### 5.2 Results

#### 5.2.1 In-Distribution Performance

| Model | SNLI Test Accuracy | Entailment F1 | Neutral F1 | Contradiction F1 |
|-------|-------------------|---------------|------------|------------------|
| Baseline | XX.X% | XX.X% | XX.X% | XX.X% |
| Debiased | XX.X% | XX.X% | XX.X% | XX.X% |

**Table 4**: In-distribution performance comparison.

**Analysis**: The debiased model achieves [XX]% accuracy, a [gain/loss] of [XX]pp compared to baseline. This [acceptable/concerning] trade-off is [expected/unexpected] because [explanation].

#### 5.2.2 Out-of-Distribution Performance

| Model | [OOD Dataset] Accuracy | Hypothesis-Only Accuracy |
|-------|------------------------|-------------------------|
| Baseline | XX.X% | XX.X% |
| Debiased | XX.X% | XX.X% |

**Table 5**: Out-of-distribution generalization.

**Analysis**: The debiased model shows [XX]pp improvement on [OOD dataset], demonstrating [better/worse] generalization. Hypothesis-only accuracy decreases from [XX]% to [XX]%, indicating reduced artifact reliance.

#### 5.2.3 Stratified Analysis

Recomputing lexical overlap and length analyses:

| Overlap Level | Baseline Acc | Debiased Acc | Δ |
|---------------|--------------|--------------|---|
| 0-20% | XX.X% | XX.X% | +/-XX.Xpp |
| 20-40% | XX.X% | XX.X% | +/-XX.Xpp |
| 40-60% | XX.X% | XX.X% | +/-XX.Xpp |
| 60-80% | XX.X% | XX.X% | +/-XX.Xpp |
| 80-100% | XX.X% | XX.X% | +/-XX.Xpp |

**Table 6**: Performance across overlap bins before/after debiasing.

**Interpretation**: The debiased model shows [more/less] consistent performance across overlap levels, suggesting [reduced/persistent] reliance on lexical matching.

### 5.3 Ablation Studies

*[If applicable]*

We conduct ablation studies to understand which components of our approach contribute most to debiasing:

[Ablation results and analysis]

---

## 6. Discussion

### 6.1 Key Takeaways

1. **Dataset artifacts are pervasive**: Our ELECTRA baseline achieves [XX]% hypothesis-only accuracy, confirming that high SNLI performance can stem from spurious correlations rather than robust reasoning.

2. **[Mitigation method] is effective for [aspect]**: We demonstrate [XX]pp improvement on [metric], showing that [conclusion about effectiveness].

3. **Trade-offs are necessary**: Debiasing results in [small/large] in-distribution performance loss ([XX]pp) but improves robustness, highlighting the tension between fitting training distribution and generalizing broadly.

4. **Evaluation matters**: Standard accuracy on SNLI is insufficient for assessing genuine language understanding. Hypothesis-only baselines and out-of-distribution tests are essential.

### 6.2 Implications

**For NLP Practitioners**:
- Always evaluate hypothesis-only or premise-only baselines to detect artifacts
- Use out-of-distribution benchmarks (HANS, contrast sets) alongside in-distribution metrics
- Consider debiasing methods when deploying models in real-world applications

**For Dataset Construction**:
- Annotation protocols should actively avoid systematic biases
- Adversarial filtering can help create more challenging benchmarks
- Regular artifact analysis should accompany dataset releases

**For Research**:
- High benchmark performance does not guarantee robust understanding
- Shortcut learning is a fundamental challenge requiring continued attention
- Debiasing methods improve generalization but have not solved the problem entirely

### 6.3 Limitations

1. **Computational constraints**: A100 GPU allows training small models but limits exploration of larger architectures (ELECTRA-base, ELECTRA-large)

2. **Single dataset focus**: This study focuses on SNLI; findings may not generalize to other NLI datasets (MultiNLI, ANLI) or tasks

3. **Mitigation scope**: We implement [one/two] debiasing method(s); other approaches (confidence regularization, causal intervention) may yield different results

4. **Evaluation coverage**: [Specify if you didn't evaluate on all proposed benchmarks]

### 6.4 Future Work

1. **Extend to larger models**: Investigate whether artifacts persist in larger pre-trained models (ELECTRA-large, RoBERTa, LLMs)

2. **Multi-dataset debiasing**: Train on multiple NLI datasets simultaneously to reduce dataset-specific artifacts

3. **Causal analysis**: Apply causal inference methods (do-calculus, counterfactual generation) for deeper understanding of learned representations

4. **Real-world evaluation**: Test debiased models on downstream applications to assess practical impact

---

## 7. Conclusion

This study investigated dataset artifacts in ELECTRA-small trained on SNLI and evaluated debiasing interventions. Our systematic analysis revealed [SEVERE/MODERATE/MILD] hypothesis-only bias ([XX]% vs. 33% random), [STRONG/WEAK] lexical overlap correlation, and [presence/absence of length bias]. These findings demonstrate that high SNLI accuracy can stem from spurious pattern matching rather than genuine language understanding.

We implemented [mitigation method] to address these artifacts, achieving [XX]pp improvement in out-of-distribution generalization while maintaining competitive in-distribution performance ([XX]% → [XX]%). The reduced hypothesis-only accuracy ([XX]% → [XX]%) indicates decreased artifact reliance, confirming the effectiveness of our approach.

Our work highlights the importance of artifact-aware evaluation for NLP systems. Standard benchmark accuracy is necessary but insufficient for assessing model robustness. We recommend that practitioners routinely employ hypothesis-only baselines, stratified analysis, and out-of-distribution testing to ensure models learn genuine task understanding rather than dataset-specific shortcuts.

Dataset artifacts remain a fundamental challenge in NLP. While debiasing methods show promise, continued research into robust learning, causal reasoning, and artifact-resistant evaluation is essential for developing truly reliable language understanding systems.

---

## References

[1] Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). A large annotated corpus for learning natural language inference. *EMNLP*.

[2] Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S., & Smith, N. A. (2018). Annotation artifacts in natural language inference data. *NAACL*.

[3] McCoy, R. T., Pavlick, E., & Linzen, T. (2019). Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. *ACL*.

[4] [Additional key papers from your literature review]

[5] Clark, C., Yatskar, M., & Zettlemoyer, L. (2019). Don't take the easy way out: Ensemble based methods for avoiding known dataset biases. *EMNLP*.

[6] Swayamdipta, S., Schwartz, R., Lourie, N., Choi, Y., & Smith, N. A. (2020). Dataset cartography: Mapping and diagnosing datasets with training dynamics. *EMNLP*.

[7] Clark, K., Luong, M. T., Le, Q. V., & Manning, C. D. (2020). ELECTRA: Pre-training text encoders as discriminators rather than generators. *ICLR*.

[8] Bostrom, K., Chen, J., & Durrett, G. (2021). fp-dataset-artifacts starter code. University of Texas at Austin. https://github.com/gregdurrett/fp-dataset-artifacts

[Additional references from your literature review]

---

## Appendix A: Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | google/electra-small-discriminator |
| Max Sequence Length | 128 |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| Mixed Precision | FP16 |

**Table A1**: Training hyperparameters for baseline model.

---

## Appendix B: Sample Predictions

*[Optional: Include sample predictions showing where model succeeds/fails]*

---

## Appendix C: Code and Data Availability

All code, trained models, and analysis results are available at:
- **Repository**: https://github.com/TimFrenzel/electra-nlp-artifact-analysis
- **Google Drive**: [Link to your Drive folder with models/results]
- **Colab Notebooks**:
  - Training: `colab_training.ipynb`
  - Analysis: `colab_analysis_part1.ipynb`
  - Mitigation: `colab_mitigation_part2.ipynb` (if implemented)

---

**Document Length**: [X pages] (excluding references)
**Word Count**: ~[XXXX] words
**Status**: Professional research project report
