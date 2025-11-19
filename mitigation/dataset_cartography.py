"""
Dataset Cartography for identifying hard, ambiguous, and easy examples.

Based on "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics"
(Swayamdipta et al., 2020)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetCartographer:
    """
    Analyzes training dynamics to identify data quality.

    Tracks:
    - Confidence: Mean predicted probability for true label
    - Variability: Std of predicted probability for true label
    - Correctness: Fraction of epochs example was predicted correctly
    """

    def __init__(self, model_path: str, dataset_name: str):
        """
        Initialize DatasetCartographer.

        Args:
            model_path: Base model path
            dataset_name: Dataset to analyze
        """
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Track training dynamics
        self.training_dynamics = defaultdict(lambda: {
            "probabilities": [],
            "predictions": [],
            "labels": [],
        })

        self.results = {}

    def record_training_dynamics(
        self,
        model: torch.nn.Module,
        dataset,
        epoch: int,
    ):
        """
        Record model predictions during training.

        Args:
            model: Current model
            dataset: Training dataset
            epoch: Current epoch number
        """
        model.eval()
        device = next(model.parameters()).device

        logger.info(f"Recording dynamics for epoch {epoch}")

        with torch.no_grad():
            for idx, example in enumerate(tqdm(dataset)):
                inputs = self.tokenizer(
                    example["premise"],
                    example["hypothesis"],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred = torch.argmax(logits, dim=-1).item()

                label = example["label"]

                # Record dynamics
                self.training_dynamics[idx]["probabilities"].append(probs[label])
                self.training_dynamics[idx]["predictions"].append(pred)
                self.training_dynamics[idx]["labels"].append(label)

        model.train()

    def compute_cartography_metrics(self) -> pd.DataFrame:
        """
        Compute cartography metrics from recorded dynamics.

        Returns:
            DataFrame with metrics for each example
        """
        logger.info("Computing cartography metrics...")

        metrics = []

        for idx, dynamics in self.training_dynamics.items():
            probs = np.array(dynamics["probabilities"])
            preds = np.array(dynamics["predictions"])
            labels = np.array(dynamics["labels"])

            # Compute metrics
            confidence = np.mean(probs)
            variability = np.std(probs)
            correctness = np.mean(preds == labels)

            metrics.append({
                "idx": idx,
                "confidence": confidence,
                "variability": variability,
                "correctness": correctness,
            })

        df = pd.DataFrame(metrics)

        # Categorize examples
        df["category"] = df.apply(self._categorize_example, axis=1)

        self.results = df
        return df

    def _categorize_example(self, row: pd.Series) -> str:
        """
        Categorize example based on confidence and variability.

        Args:
            row: Row with confidence and variability

        Returns:
            Category string
        """
        conf = row["confidence"]
        var = row["variability"]

        # Thresholds (can be tuned)
        high_conf = 0.7
        low_var = 0.1

        if conf > high_conf and var < low_var:
            return "easy"
        elif conf < high_conf and var > low_var:
            return "hard"
        else:
            return "ambiguous"

    def identify_hard_examples(
        self, top_k: Optional[int] = None, percentile: float = 20
    ) -> List[int]:
        """
        Identify hardest examples.

        Args:
            top_k: Return top k hardest examples
            percentile: Return bottom percentile by confidence

        Returns:
            List of example indices
        """
        if self.results is None or len(self.results) == 0:
            logger.warning("No results available. Run compute_cartography_metrics first.")
            return []

        # Sort by confidence (ascending) and variability (descending)
        sorted_df = self.results.sort_values(["confidence", "variability"], ascending=[True, False])

        if top_k:
            hard_examples = sorted_df.head(top_k)["idx"].tolist()
        else:
            threshold = np.percentile(self.results["confidence"], percentile)
            hard_examples = self.results[self.results["confidence"] < threshold]["idx"].tolist()

        logger.info(f"Identified {len(hard_examples)} hard examples")
        return hard_examples

    def export_results(self, output_path: str):
        """Export cartography results to CSV."""
        if self.results is None or len(self.results) == 0:
            logger.warning("No results to export")
            return

        self.results.to_csv(output_path, index=False)
        logger.info(f"Exported results to {output_path}")

    def plot_cartography_map(self, output_path: Optional[str] = None):
        """
        Plot dataset cartography map (confidence vs variability).

        Args:
            output_path: Path to save figure (optional)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.results is None or len(self.results) == 0:
            logger.warning("No results to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot by category
        categories = self.results["category"].unique()
        colors = {"easy": "green", "hard": "red", "ambiguous": "orange"}

        for cat in categories:
            data = self.results[self.results["category"] == cat]
            ax.scatter(
                data["confidence"],
                data["variability"],
                c=colors.get(cat, "gray"),
                label=cat,
                alpha=0.6,
                s=20,
            )

        ax.set_xlabel("Confidence", fontsize=12)
        ax.set_ylabel("Variability", fontsize=12)
        ax.set_title("Dataset Cartography Map", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {output_path}")
        else:
            plt.show()


def analyze_training_dynamics(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    num_epochs: int = 3,
) -> DatasetCartographer:
    """
    Convenience function to analyze training dynamics.

    Args:
        model_path: Path to base model
        dataset_name: Dataset to analyze
        output_dir: Output directory for results
        num_epochs: Number of epochs to train

    Returns:
        DatasetCartographer instance with results
    """
    cartographer = DatasetCartographer(model_path, dataset_name)

    # Load dataset
    if dataset_name.lower() == "snli":
        dataset = load_dataset("snli")["train"]
    else:
        dataset = load_dataset(dataset_name)["train"]

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)

    # Training loop (simplified)
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Record dynamics before training
        cartographer.record_training_dynamics(model, dataset, epoch)

        # Train for one epoch (simplified - actual training would go here)
        # For full implementation, integrate with Trainer

    # Compute metrics
    cartographer.compute_cartography_metrics()
    cartographer.export_results(f"{output_dir}/cartography_metrics.csv")
    cartographer.plot_cartography_map(f"{output_dir}/cartography_map.png")

    return cartographer


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python dataset_cartography.py <model_path> <dataset>")
        sys.exit(1)

    model_path = sys.argv[1]
    dataset = sys.argv[2]
    output_dir = "./results/cartography"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cartographer = analyze_training_dynamics(model_path, dataset, output_dir)
    hard_examples = cartographer.identify_hard_examples(top_k=100)
    print(f"Identified {len(hard_examples)} hard examples")
