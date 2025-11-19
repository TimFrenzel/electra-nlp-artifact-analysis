"""
Error analysis tools for identifying systematic model failures and artifacts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Analyzes model errors to identify potential dataset artifacts."""

    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        """
        Initialize ErrorAnalyzer.

        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer (defaults to model_path)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Store predictions and errors
        self.predictions = []
        self.errors = []
        self.results = {}

    def predict(self, premise: str, hypothesis: str) -> Tuple[int, np.ndarray]:
        """
        Make prediction on a single example.

        Args:
            premise: Premise text
            hypothesis: Hypothesis text

        Returns:
            Tuple of (predicted_label, probabilities)
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = torch.argmax(logits, dim=-1).item()

        return pred, probs

    def analyze_dataset(
        self, dataset_name: str, split: str = "validation", max_samples: Optional[int] = None
    ) -> Dict:
        """
        Analyze model performance on a dataset.

        Args:
            dataset_name: Name of dataset (e.g., "snli")
            split: Dataset split to analyze
            max_samples: Maximum number of samples to analyze

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Loading dataset: {dataset_name}, split: {split}")

        # Load dataset
        if dataset_name.lower() == "snli":
            dataset = load_dataset("snli")[split]
        elif dataset_name.lower() in ["multi_nli", "mnli"]:
            dataset = load_dataset("multi_nli")["validation_matched"]
        else:
            dataset = load_dataset(dataset_name)[split]

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # Run predictions
        correct = 0
        total = 0
        self.predictions = []
        self.errors = []

        logger.info(f"Analyzing {len(dataset)} examples...")

        for idx, example in enumerate(tqdm(dataset)):
            # Skip invalid examples
            if example["label"] == -1:
                continue

            pred, probs = self.predict(example["premise"], example["hypothesis"])
            label = example["label"]

            prediction_data = {
                "idx": idx,
                "premise": example["premise"],
                "hypothesis": example["hypothesis"],
                "true_label": label,
                "pred_label": pred,
                "probabilities": probs.tolist(),
                "confidence": float(probs[pred]),
                "correct": pred == label,
            }

            self.predictions.append(prediction_data)

            if pred == label:
                correct += 1
            else:
                self.errors.append(prediction_data)

            total += 1

        accuracy = correct / total if total > 0 else 0

        # Analyze error patterns
        error_patterns = self._analyze_error_patterns()

        self.results = {
            "accuracy": accuracy,
            "total_examples": total,
            "correct": correct,
            "errors": len(self.errors),
            "error_rate": 1 - accuracy,
            "error_patterns": error_patterns,
        }

        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Errors: {len(self.errors)}/{total}")

        return self.results

    def _analyze_error_patterns(self) -> Dict:
        """Analyze patterns in errors to identify potential artifacts."""
        if not self.errors:
            return {}

        patterns = {
            "by_true_label": defaultdict(int),
            "by_pred_label": defaultdict(int),
            "confusion_matrix": defaultdict(lambda: defaultdict(int)),
            "low_confidence_errors": [],
            "high_confidence_errors": [],
        }

        for error in self.errors:
            true_label = error["true_label"]
            pred_label = error["pred_label"]
            confidence = error["confidence"]

            patterns["by_true_label"][true_label] += 1
            patterns["by_pred_label"][pred_label] += 1
            patterns["confusion_matrix"][true_label][pred_label] += 1

            if confidence < 0.6:
                patterns["low_confidence_errors"].append(error)
            elif confidence > 0.9:
                patterns["high_confidence_errors"].append(error)

        return patterns

    def analyze_hypothesis_only(self, max_samples: int = 1000) -> Dict:
        """
        Analyze hypothesis-only baseline to detect label biases.

        Args:
            max_samples: Maximum samples to analyze

        Returns:
            Dictionary with hypothesis-only results
        """
        logger.info("Running hypothesis-only analysis...")

        correct = 0
        total = 0

        for pred_data in self.predictions[:max_samples]:
            # Predict using only hypothesis
            hypothesis = pred_data["hypothesis"]
            pred, _ = self.predict("", hypothesis)

            if pred == pred_data["true_label"]:
                correct += 1
            total += 1

        hypothesis_only_acc = correct / total if total > 0 else 0

        logger.info(f"Hypothesis-only accuracy: {hypothesis_only_acc:.2%}")

        return {
            "hypothesis_only_accuracy": hypothesis_only_acc,
            "samples_analyzed": total,
        }

    def export_errors(self, output_path: str):
        """Export errors to CSV file."""
        if not self.errors:
            logger.warning("No errors to export")
            return

        df = pd.DataFrame(self.errors)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(self.errors)} errors to {output_path}")

    def plot_error_types(self):
        """Plot error distribution by type."""
        from .visualization import plot_error_types

        plot_error_types(self.results)


def analyze_predictions(
    model_path: str,
    dataset: str,
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> ErrorAnalyzer:
    """
    Convenience function for analyzing model predictions.

    Args:
        model_path: Path to trained model
        dataset: Dataset name
        split: Dataset split
        max_samples: Maximum samples to analyze

    Returns:
        ErrorAnalyzer instance with results
    """
    analyzer = ErrorAnalyzer(model_path)
    analyzer.analyze_dataset(dataset, split, max_samples)
    return analyzer


def find_lexical_overlaps(premise: str, hypothesis: str) -> Dict:
    """
    Find lexical overlaps between premise and hypothesis.

    Args:
        premise: Premise text
        hypothesis: Hypothesis text

    Returns:
        Dictionary with overlap statistics
    """
    premise_words = set(premise.lower().split())
    hypothesis_words = set(hypothesis.lower().split())

    overlap = premise_words & hypothesis_words
    overlap_ratio = len(overlap) / len(hypothesis_words) if hypothesis_words else 0

    return {
        "overlap_count": len(overlap),
        "overlap_ratio": overlap_ratio,
        "overlapping_words": list(overlap),
    }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python error_analysis.py <model_path> <dataset>")
        sys.exit(1)

    model_path = sys.argv[1]
    dataset = sys.argv[2]

    analyzer = analyze_predictions(model_path, dataset, max_samples=1000)
    analyzer.export_errors("errors.csv")
    print(f"Analysis complete. Accuracy: {analyzer.results['accuracy']:.2%}")
