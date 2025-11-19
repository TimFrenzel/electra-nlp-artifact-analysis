"""
Contrast set evaluation for testing model robustness.

Contrast sets are minimal perturbations of test examples that should
change the label, helping identify spurious correlations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ContrastSetEvaluator:
    """Evaluates model on contrast sets."""

    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        """
        Initialize ContrastSetEvaluator.

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

    def evaluate_contrast_pair(
        self, original: Dict, contrast: Dict
    ) -> Dict:
        """
        Evaluate a contrast pair.

        Args:
            original: Original example dict with premise, hypothesis, label
            contrast: Contrasting example dict with premise, hypothesis, label

        Returns:
            Dictionary with evaluation results
        """
        orig_pred, orig_probs = self.predict(
            original["premise"], original["hypothesis"]
        )
        contrast_pred, contrast_probs = self.predict(
            contrast["premise"], contrast["hypothesis"]
        )

        result = {
            "original": {
                "premise": original["premise"],
                "hypothesis": original["hypothesis"],
                "true_label": original["label"],
                "pred_label": orig_pred,
                "probabilities": orig_probs.tolist(),
                "correct": orig_pred == original["label"],
            },
            "contrast": {
                "premise": contrast["premise"],
                "hypothesis": contrast["hypothesis"],
                "true_label": contrast["label"],
                "pred_label": contrast_pred,
                "probabilities": contrast_probs.tolist(),
                "correct": contrast_pred == contrast["label"],
            },
            "both_correct": (orig_pred == original["label"])
            and (contrast_pred == contrast["label"]),
            "prediction_changed": orig_pred != contrast_pred,
        }

        return result

    def evaluate_contrast_set(
        self, original_data: List[Dict], contrast_data: List[Dict]
    ) -> Dict:
        """
        Evaluate full contrast set.

        Args:
            original_data: List of original examples
            contrast_data: List of contrast examples (same length)

        Returns:
            Dictionary with aggregated results
        """
        if len(original_data) != len(contrast_data):
            raise ValueError("Original and contrast data must have same length")

        logger.info(f"Evaluating {len(original_data)} contrast pairs...")

        pair_results = []
        both_correct = 0
        prediction_changed = 0

        for orig, contrast in tqdm(zip(original_data, contrast_data), total=len(original_data)):
            result = self.evaluate_contrast_pair(orig, contrast)
            pair_results.append(result)

            if result["both_correct"]:
                both_correct += 1
            if result["prediction_changed"]:
                prediction_changed += 1

        total = len(pair_results)
        original_acc = sum(r["original"]["correct"] for r in pair_results) / total
        contrast_acc = sum(r["contrast"]["correct"] for r in pair_results) / total
        consistency = both_correct / total
        sensitivity = prediction_changed / total

        self.results = {
            "total_pairs": total,
            "original_accuracy": original_acc,
            "contrast_accuracy": contrast_acc,
            "consistency": consistency,  # Both original and contrast correct
            "sensitivity": sensitivity,  # Prediction changed between pair
            "pair_results": pair_results,
        }

        logger.info(f"Original accuracy: {original_acc:.2%}")
        logger.info(f"Contrast accuracy: {contrast_acc:.2%}")
        logger.info(f"Consistency (both correct): {consistency:.2%}")
        logger.info(f"Sensitivity (prediction changed): {sensitivity:.2%}")

        return self.results

    def create_negation_contrasts(self, examples: List[Dict]) -> List[Dict]:
        """
        Create simple negation-based contrasts.

        Args:
            examples: List of examples with hypothesis field

        Returns:
            List of contrast examples with negated hypotheses
        """
        contrasts = []

        for example in examples:
            hypothesis = example["hypothesis"]
            premise = example["premise"]
            label = example["label"]

            # Simple negation (this is a basic heuristic)
            if " not " in hypothesis:
                negated = hypothesis.replace(" not ", " ")
            elif " no " in hypothesis:
                negated = hypothesis.replace(" no ", " ")
            else:
                negated = f"It is not the case that {hypothesis}"

            # Flip label (simple heuristic)
            if label == 0:  # entailment -> contradiction
                new_label = 2
            elif label == 2:  # contradiction -> entailment
                new_label = 0
            else:  # neutral stays neutral
                new_label = 1

            contrasts.append({
                "premise": premise,
                "hypothesis": negated,
                "label": new_label,
            })

        return contrasts

    def export_results(self, output_path: str):
        """Export results to JSON file."""
        if not self.results:
            logger.warning("No results to export")
            return

        output_data = {
            "summary": {
                "total_pairs": self.results["total_pairs"],
                "original_accuracy": self.results["original_accuracy"],
                "contrast_accuracy": self.results["contrast_accuracy"],
                "consistency": self.results["consistency"],
                "sensitivity": self.results["sensitivity"],
            },
            "pairs": self.results["pair_results"],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Exported results to {output_path}")


def evaluate_contrast_set(
    model_path: str,
    original_data: str,
    contrast_data: str,
) -> ContrastSetEvaluator:
    """
    Convenience function for contrast set evaluation.

    Args:
        model_path: Path to trained model
        original_data: Path to JSON file with original examples
        contrast_data: Path to JSON file with contrast examples

    Returns:
        ContrastSetEvaluator instance with results
    """
    # Load data
    with open(original_data) as f:
        original = json.load(f)
    with open(contrast_data) as f:
        contrast = json.load(f)

    evaluator = ContrastSetEvaluator(model_path)
    evaluator.evaluate_contrast_set(original, contrast)

    return evaluator


def generate_minimal_pairs(examples: List[Dict], num_pairs: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate minimal pairs by making small perturbations.

    Args:
        examples: Source examples
        num_pairs: Number of pairs to generate

    Returns:
        Tuple of (original_examples, contrast_examples)
    """
    import random

    random.seed(42)

    # Select random examples
    selected = random.sample(examples, min(num_pairs, len(examples)))

    original = []
    contrasts = []

    for ex in selected:
        original.append(ex)

        # Create contrast by swapping words (simple heuristic)
        hypothesis = ex["hypothesis"]
        words = hypothesis.split()

        if len(words) > 3:
            # Swap random adjacent words
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
            perturbed = " ".join(words)

            contrasts.append({
                "premise": ex["premise"],
                "hypothesis": perturbed,
                "label": 1,  # Likely neutral after perturbation
            })
        else:
            # Just negate
            contrasts.append({
                "premise": ex["premise"],
                "hypothesis": f"Not {hypothesis}",
                "label": 2 if ex["label"] == 0 else 0,
            })

    return original, contrasts


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 4:
        print("Usage: python contrast_sets.py <model_path> <original_data.json> <contrast_data.json>")
        sys.exit(1)

    model_path = sys.argv[1]
    original_data = sys.argv[2]
    contrast_data = sys.argv[3]

    evaluator = evaluate_contrast_set(model_path, original_data, contrast_data)
    evaluator.export_results("contrast_results.json")
    print(f"Consistency: {evaluator.results['consistency']:.2%}")
