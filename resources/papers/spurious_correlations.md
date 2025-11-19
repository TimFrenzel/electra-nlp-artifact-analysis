# Spurious Correlations and Dataset Artifacts in NLP

Research papers on identifying and mitigating spurious correlations in NLP models.

---

## 2025

### Mitigating Spurious Correlations
- **Venue**: ICLR 2025
- **Link**: https://openreview.net/pdf/de385dc93e29d5a5b0fd224a1344c0015b5f894a.pdf
- **Key Contribution**: Addresses spurious correlations in Vision-Language Models (VLMs), showing these models still suffer from predictions based on irrelevant features
- **Application**: Vision-language models, multimodal learning
- **Methods**: Novel mitigation strategies for VLMs

### Assessing Robustness to Spurious Correlations in Post-Training Language Models
- **Venue**: arXiv 2025
- **Link**: https://arxiv.org/html/2505.05704v1
- **Key Contribution**: Investigates whether LLMs learn to rely on spurious features rather than true correctness during post-training on noisy data
- **Application**: Post-training evaluation, LLM fine-tuning
- **Methods**: Systematic evaluation of post-training robustness

---

## 2024

### Explore Spurious Correlations at the Concept Level in Language Models
- **Venue**: ACL 2024
- **Link**: https://aclanthology.org/2024.acl-long.28.pdf
- **Key Contribution**: Examines how LLMs (LLAMA2, GPT3.5) handle spurious correlations at the concept level
- **Application**: Large language models, concept-level analysis
- **Methods**: Concept-based evaluation framework
- **Code**: Check ACL proceedings for implementation

### Understanding and Mitigating Spurious Correlations in Text Classification
- **Venue**: EACL 2024 Findings
- **Link**: https://aclanthology.org/2024.findings-eacl.68.pdf
- **Key Contribution**: Proposes that last layer re-training is sufficient for robustness to spurious correlations
- **Application**: Text classification
- **Methods**: NFL (Neighborhood Feature Learning), last-layer retraining
- **Key Finding**: Significant robustness improvements with minimal computational cost

### Spurious Correlations in Machine Learning: A Survey
- **Venue**: arXiv 2024
- **Link**: https://arxiv.org/html/2402.12715v2
- **Key Contribution**: Comprehensive survey noting spurious correlations studied under various names: shortcuts, dataset biases, group robustness, simplicity bias
- **Domains**: Computer vision, NLP, healthcare
- **Scope**: 150+ papers reviewed
- **Taxonomy**: Organizes mitigation methods by approach

### Spurious Correlations and Beyond in SDOH Extraction with LLMs
- **Venue**: arXiv 2024
- **Link**: https://arxiv.org/html/2506.00134
- **Key Contribution**: Understanding and mitigating shortcut learning in social determinants of health extraction
- **Application**: Clinical NLP, healthcare AI
- **Methods**: Domain-specific mitigation strategies

---

## 2022-2023

### Towards Robustifying NLI Models Against Lexical Dataset Biases
- **Venue**: ACL 2020 (Still highly relevant, widely cited in 2022-2024)
- **Link**: https://www.semanticscholar.org/paper/Towards-Robustifying-NLI-Models-Against-Lexical-Zhou-Bansal/8cbb254809749fbe00cbf224daba059e345891eb
- **Key Contribution**: Addresses lexical dataset biases in NLI
- **Application**: Natural Language Inference
- **Methods**: Debiasing through data augmentation and reweighting
- **Citations**: 150+ (as of 2024)

### Identifying and Mitigating Spurious Correlations for Improving Robustness in NLP Models
- **Venue**: arXiv 2021 (Updated 2022)
- **Link**: https://arxiv.org/abs/2110.07736
- **Key Contribution**: Automatically identifies spurious correlations by leveraging interpretability methods to extract tokens affecting model decisions
- **Application**: General NLP tasks
- **Methods**: Saliency-based identification, token attribution
- **Code**: Available on request from authors

### Towards Mitigating Spurious Correlations in the Wild
- **Venue**: ResearchGate 2023
- **Link**: https://www.researchgate.net/publication/371758556_Towards_Mitigating_Spurious_Correlations_in_the_Wild_A_Benchmark_a_more_Realistic_Dataset
- **Key Contribution**: New benchmark and realistic dataset for spurious correlations
- **Application**: Real-world deployment scenarios
- **Dataset**: Available for research use

---

## Key Insights from Recent Literature

### Common Findings
1. **Last-layer retraining** is surprisingly effective for mitigating learned shortcuts
2. **Concept-level analysis** provides deeper understanding than token-level
3. **Domain-specific artifacts** require tailored mitigation strategies
4. **LLMs still susceptible** despite scale and pre-training

### Effective Mitigation Strategies
1. **Data-level**:
   - Counterfactual augmentation
   - Hard negative mining
   - Balanced sampling

2. **Model-level**:
   - Last-layer retraining
   - Ensemble debiasing
   - Adversarial training

3. **Evaluation-level**:
   - Out-of-distribution testing
   - Concept-based evaluation
   - Stress tests (HANS, etc.)

### Open Challenges
- Scalability to new domains
- Computational efficiency
- Preservation of in-distribution performance
- Detection of unknown artifacts

---

## Related Topics
- See `debiasing_methods.md` for mitigation techniques
- See `adversarial_robustness.md` for robustness evaluation
- See `dataset_cartography.md` for data quality analysis

## Citation Template

```bibtex
@inproceedings{spurious-nlp-2024,
  title={Understanding and Mitigating Spurious Correlations in Text Classification},
  booktitle={Findings of EACL},
  year={2024},
  url={https://aclanthology.org/2024.findings-eacl.68.pdf}
}
```

---

**Last Updated**: November 2025
