"""
Debiasing methods for mitigating dataset artifacts.

Implements:
- Hypothesis-only baseline debiasing
- Ensemble debiasing
- Learned-Mixin+H0 (product of experts)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class BiasedModel(nn.Module):
    """
    Biased model that sees only hypothesis (for NLI) or question (for QA).
    This model learns to exploit dataset artifacts.
    """

    def __init__(self, base_model_path: str, num_labels: int = 3):
        """
        Initialize biased model.

        Args:
            base_model_path: Path to base model
            num_labels: Number of output labels
        """
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass using only hypothesis."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs


class EnsembleDebiaser:
    """
    Ensemble debiasing using product of experts.

    Trains a biased model on artifacts, then debiases main model by
    downweighting examples the biased model gets correct.
    """

    def __init__(
        self,
        main_model_path: str = "google/electra-small-discriminator",
        num_labels: int = 3,
    ):
        """
        Initialize ensemble debiaser.

        Args:
            main_model_path: Path to main model
            num_labels: Number of output labels
        """
        self.main_model_path = main_model_path
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(main_model_path)

        self.biased_model = None
        self.main_model = None

    def train_biased_model(
        self,
        dataset: Dataset,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 32,
    ):
        """
        Train biased model on hypothesis/question only.

        Args:
            dataset: Training dataset
            output_dir: Output directory for biased model
            num_epochs: Number of training epochs
            batch_size: Training batch size
        """
        logger.info("Training biased model (hypothesis-only)...")

        # Preprocess dataset to use only hypothesis
        def preprocess_hypothesis_only(examples):
            # For NLI: use only hypothesis
            return self.tokenizer(
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=128,  # Shorter for hypothesis-only
            )

        processed_dataset = dataset.map(
            preprocess_hypothesis_only,
            batched=True,
            remove_columns=dataset.column_names,
        )

        # Initialize biased model
        self.biased_model = AutoModelForSequenceClassification.from_pretrained(
            self.main_model_path,
            num_labels=self.num_labels,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=2e-5,
            save_strategy="epoch",
            logging_steps=100,
        )

        # Train
        trainer = Trainer(
            model=self.biased_model,
            args=training_args,
            train_dataset=processed_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(output_dir)

        logger.info(f"Biased model saved to {output_dir}")

    def compute_example_weights(
        self, dataset: Dataset, temperature: float = 1.0
    ) -> np.ndarray:
        """
        Compute example weights based on biased model predictions.

        Args:
            dataset: Dataset to weight
            temperature: Temperature for softening weights

        Returns:
            Array of example weights
        """
        if self.biased_model is None:
            raise ValueError("Must train biased model first")

        logger.info("Computing example weights from biased model...")

        self.biased_model.eval()
        device = next(self.biased_model.parameters()).device

        weights = []

        for example in dataset:
            # Get biased model prediction
            inputs = self.tokenizer(
                example["hypothesis"],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.biased_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()[0]

            label = example["label"]
            bias_score = probs[label]  # Probability biased model assigns to true label

            # Weight inversely to bias score
            # High bias_score = low weight (biased model confident)
            weight = 1.0 - bias_score

            weights.append(weight)

        return np.array(weights)

    def train_debiased_model(
        self,
        dataset: Dataset,
        output_dir: str,
        example_weights: Optional[np.ndarray] = None,
        num_epochs: int = 3,
        batch_size: int = 32,
    ):
        """
        Train debiased main model with reweighted examples.

        Args:
            dataset: Training dataset
            output_dir: Output directory
            example_weights: Per-example weights (computed if not provided)
            num_epochs: Number of training epochs
            batch_size: Training batch size
        """
        logger.info("Training debiased main model...")

        # Compute weights if not provided
        if example_weights is None:
            example_weights = self.compute_example_weights(dataset)

        # Preprocess dataset (full premise + hypothesis)
        def preprocess_full(examples):
            return self.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=512,
            )

        processed_dataset = dataset.map(
            preprocess_full,
            batched=True,
            remove_columns=dataset.column_names,
        )

        # Add weights to dataset
        processed_dataset = processed_dataset.add_column("weight", example_weights.tolist())

        # Initialize main model
        self.main_model = AutoModelForSequenceClassification.from_pretrained(
            self.main_model_path,
            num_labels=self.num_labels,
        )

        # Custom trainer with weighted loss
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                weights = inputs.pop("weight", None)
                outputs = model(**inputs)
                loss = outputs.loss

                if weights is not None:
                    # Apply weights to loss
                    weights_tensor = torch.tensor(weights, device=loss.device)
                    loss = (loss * weights_tensor).mean()

                return (loss, outputs) if return_outputs else loss

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=2e-5,
            save_strategy="epoch",
            logging_steps=100,
        )

        # Train
        trainer = WeightedTrainer(
            model=self.main_model,
            args=training_args,
            train_dataset=processed_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(output_dir)

        logger.info(f"Debiased model saved to {output_dir}")


def train_biased_model(
    dataset_name: str,
    output_dir: str,
    base_model: str = "google/electra-small-discriminator",
    num_epochs: int = 3,
):
    """
    Convenience function to train hypothesis-only biased model.

    Args:
        dataset_name: Dataset to train on
        output_dir: Output directory
        base_model: Base model path
        num_epochs: Number of epochs
    """
    # Load dataset
    if dataset_name.lower() == "snli":
        dataset = load_dataset("snli")["train"]
    else:
        dataset = load_dataset(dataset_name)["train"]

    # Filter invalid labels
    dataset = dataset.filter(lambda x: x["label"] != -1)

    # Initialize debiaser
    debiaser = EnsembleDebiaser(base_model, num_labels=3)

    # Train biased model
    debiaser.train_biased_model(dataset, output_dir, num_epochs)

    return debiaser


class ProductOfExperts(nn.Module):
    """
    Product of Experts combining main model and biased model.

    Computes: P(y|x) ∝ P_main(y|x,h) / P_bias(y|h)^α
    """

    def __init__(
        self,
        main_model: nn.Module,
        bias_model: nn.Module,
        alpha: float = 1.0,
    ):
        """
        Initialize Product of Experts.

        Args:
            main_model: Main model (sees full input)
            bias_model: Biased model (sees only hypothesis)
            alpha: Weight for bias model (higher = more debiasing)
        """
        super().__init__()
        self.main_model = main_model
        self.bias_model = bias_model
        self.alpha = alpha

    def forward(self, main_inputs, bias_inputs):
        """
        Forward pass combining both models.

        Args:
            main_inputs: Inputs for main model (premise + hypothesis)
            bias_inputs: Inputs for bias model (hypothesis only)

        Returns:
            Combined logits
        """
        # Get logits from both models
        main_logits = self.main_model(**main_inputs).logits
        bias_logits = self.bias_model(**bias_inputs).logits

        # Compute product of experts
        # log P(y|x) = log P_main(y|x) - α * log P_bias(y|h)
        combined_logits = main_logits - self.alpha * bias_logits

        return combined_logits


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python debiasing.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]
    output_dir = "./models/biased_model"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    debiaser = train_biased_model(dataset, output_dir)
    print(f"Biased model trained and saved to {output_dir}")
