# Dataset Cartography and Training Dynamics

Papers on using training dynamics to identify data quality issues and bias.

---

## Foundational Work

### Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics
- **Authors**: Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yejin Choi, Noah A. Smith
- **Venue**: EMNLP 2020 (Highly influential 2021-2024)
- **Link**: https://arxiv.org/abs/2009.10795
- **ACL Anthology**: https://aclanthology.org/2020.emnlp-main.746/
- **GitHub**: https://github.com/allenai/cartography
- **Citations**: 1000+

**Key Contributions**:
- Introduces Data Maps for dataset characterization
- Three intuitive measures from training dynamics:
  1. **Confidence**: Mean predicted probability for true label
  2. **Variability**: Standard deviation of confidence across epochs
  3. **Correctness**: Fraction of epochs predicted correctly
- Categorizes examples as easy, hard, or ambiguous
- Obtained in single training run (efficient)

**Applications**:
- Data pruning for efficiency
- Hard example identification
- Bias mitigation (focus on hard examples)
- Active learning
- Dataset diagnosis

**Key Findings**:
- Easy examples: High confidence, low variability
- Hard examples: Low confidence, high variability
- Ambiguous examples: Medium confidence, high variability
- Shift from quantity to quality improves OOD generalization

---

## Recent Applications (2022-2024)

### Sampling Optimization for Bias Mitigation
- **Citation**: Swayamdipta et al. (2020), Yao et al. (2022), Kye et al. (2023)
- **Application**: Refining hard sample selection for bias reduction
- **Method**: Use training dynamics to identify examples contributing to bias

### Easy vs. Hard Data Selection
- **Authors**: Li et al. (2024)
- **Finding**: Selecting easy-to-learn data over hard-to-learn can be more beneficial for certain tasks
- **Implication**: Dataset Cartography informs selective training strategies

### GPT-3 for Challenging Reasoning
- **Application**: Using Dataset Cartography to identify examples with challenging reasoning patterns
- **Method**: Automatically identify hard examples, instruct GPT-3 to compose similar examples
- **Impact**: Targeted data augmentation for difficult cases

---

## Differential Data Maps

### Differential Dataset Cartography
- **Venue**: Springer 2023
- **Link**: https://link.springer.com/chapter/10.1007/978-3-031-35995-8_11
- **Application**: Explainable AI in personalized sentiment analysis
- **Key Contribution**: Visual comparison of different models using data maps
- **Use Case**: Understanding model differences through training dynamics

---

## Extensions and Variants

### Confidence-Based Selection
- **Method**: Select examples based on confidence thresholds
- **Application**: Data pruning, curriculum learning
- **Benefit**: Reduce dataset size while maintaining performance

### Variability-Based Identification
- **Method**: High variability indicates ambiguous or mislabeled examples
- **Application**: Data cleaning, label verification
- **Benefit**: Improve dataset quality

### Correctness-Based Filtering
- **Method**: Filter consistently misclassified examples
- **Application**: Identifying systematic biases or difficult cases
- **Benefit**: Targeted intervention for problematic examples

---

## Practical Implementation

### Basic Workflow
1. Train model for N epochs (typically 10-20)
2. Record predictions at each epoch for each example
3. Compute confidence, variability, correctness
4. Create 2D scatter plot (confidence × variability)
5. Analyze patterns and categorize examples

### Computational Cost
- **Training**: Standard training with prediction logging
- **Analysis**: Minimal (post-processing of predictions)
- **Benefit**: Single training run provides rich insights

### Integration with Debiasing
1. **Identify**: Use cartography to find hard/ambiguous examples
2. **Analyze**: Determine if these correlate with biased features
3. **Mitigate**: Focus training on hard examples, prune easy ones, or reweight

---

## Code Example

```python
from cartography import compute_train_dynamics

# During training, log predictions
dynamics = compute_train_dynamics(
    model=model,
    dataset=train_dataset,
    num_epochs=10
)

# Analyze results
easy_examples = dynamics[
    (dynamics['confidence'] > 0.7) &
    (dynamics['variability'] < 0.1)
]

hard_examples = dynamics[
    (dynamics['confidence'] < 0.5) &
    (dynamics['variability'] > 0.2)
]

ambiguous_examples = dynamics[
    (dynamics['confidence'].between(0.4, 0.6)) &
    (dynamics['variability'] > 0.2)
]
```

---

## Related Work

### Curriculum Learning
- Dataset Cartography provides data-driven curriculum
- Train on easy examples first, gradually add harder ones

### Active Learning
- Use variability to identify informative examples
- Query labels for high-variability examples

### Data Pruning
- Remove easy examples to reduce training time
- Focus compute on hard/ambiguous cases

---

## Limitations

1. **Task-Dependent**: Maps depend on model and task
2. **Requires Full Training**: Need multiple epochs to compute dynamics
3. **Linear Separation**: Assumes categories are somewhat separable
4. **Annotation Needed**: Doesn't replace human judgment for ambiguous cases

---

## Future Directions (2024-2025)

1. **LLM Adaptation**: Applying to billion-parameter models
2. **Few-Shot Learning**: Cartography for low-resource settings
3. **Multi-Task**: Joint cartography across tasks
4. **Online Learning**: Continuous cartography during deployment

---

## Key Takeaway

Dataset Cartography shifts focus from "more data" to "better data" by providing actionable insights into example difficulty and quality, enabling targeted interventions for bias mitigation and improved generalization.

---

**Last Updated**: November 2025
