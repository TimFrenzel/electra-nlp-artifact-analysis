"""
Mitigation module for bias and artifact reduction.

Contains implementations for:
- Dataset Cartography (hard example identification)
- Debiasing methods
- Adversarial training
"""

from .dataset_cartography import DatasetCartographer, analyze_training_dynamics
from .debiasing import EnsembleDebiaser, train_biased_model
from .adversarial_training import AdversarialTrainer, generate_adversarial_examples

__all__ = [
    "DatasetCartographer",
    "analyze_training_dynamics",
    "EnsembleDebiaser",
    "train_biased_model",
    "AdversarialTrainer",
    "generate_adversarial_examples",
]
