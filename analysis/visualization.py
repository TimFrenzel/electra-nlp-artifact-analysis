"""
Visualization utilities for analysis results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def plot_error_types(results: Dict, output_path: Optional[str] = None):
    """
    Plot error distribution by type.

    Args:
        results: Results dictionary from ErrorAnalyzer
        output_path: Path to save figure (optional)
    """
    if "error_patterns" not in results:
        logger.warning("No error patterns in results")
        return

    patterns = results["error_patterns"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot errors by true label
    if "by_true_label" in patterns:
        labels = list(patterns["by_true_label"].keys())
        counts = list(patterns["by_true_label"].values())

        label_names = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
        labels_str = [label_names.get(l, str(l)) for l in labels]

        axes[0].bar(labels_str, counts, color="coral")
        axes[0].set_xlabel("True Label")
        axes[0].set_ylabel("Number of Errors")
        axes[0].set_title("Errors by True Label")
        axes[0].tick_params(axis="x", rotation=45)

    # Plot confusion matrix
    if "confusion_matrix" in patterns:
        cm = patterns["confusion_matrix"]
        label_names = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

        # Create matrix
        all_labels = sorted(set(list(cm.keys()) + [k for v in cm.values() for k in v.keys()]))
        matrix = np.zeros((len(all_labels), len(all_labels)))

        for i, true_label in enumerate(all_labels):
            for j, pred_label in enumerate(all_labels):
                matrix[i, j] = cm.get(true_label, {}).get(pred_label, 0)

        # Plot heatmap
        label_strs = [label_names.get(l, str(l)) for l in all_labels]
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            xticklabels=label_strs,
            yticklabels=label_strs,
            ax=axes[1],
        )
        axes[1].set_xlabel("Predicted Label")
        axes[1].set_ylabel("True Label")
        axes[1].set_title("Confusion Matrix (Errors Only)")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_performance_breakdown(
    results_dir: str,
    group_by: str = "example_difficulty",
    output_path: Optional[str] = None,
):
    """
    Plot performance breakdown across different slices.

    Args:
        results_dir: Directory with results
        group_by: How to group results (e.g., 'example_difficulty', 'length')
        output_path: Path to save figure (optional)
    """
    # This is a placeholder - actual implementation would depend on result structure
    logger.info(f"Plotting performance breakdown by {group_by}")

    # Example visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sample data (replace with actual data)
    categories = ["Easy", "Medium", "Hard"]
    baseline = [0.92, 0.85, 0.71]
    mitigated = [0.93, 0.87, 0.78]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width / 2, baseline, width, label="Baseline", color="skyblue")
    ax.bar(x + width / 2, mitigated, width, label="Mitigated", color="lightcoral")

    ax.set_xlabel("Example Difficulty")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Performance Breakdown by {group_by}")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_confidence_distribution(
    predictions: List[Dict], output_path: Optional[str] = None
):
    """
    Plot distribution of model confidence scores.

    Args:
        predictions: List of prediction dictionaries with 'confidence' field
        output_path: Path to save figure (optional)
    """
    if not predictions:
        logger.warning("No predictions provided")
        return

    confidences_correct = [
        p["confidence"] for p in predictions if p.get("correct", False)
    ]
    confidences_incorrect = [
        p["confidence"] for p in predictions if not p.get("correct", True)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        confidences_correct,
        bins=30,
        alpha=0.6,
        label="Correct",
        color="green",
        density=True,
    )
    ax.hist(
        confidences_incorrect,
        bins=30,
        alpha=0.6,
        label="Incorrect",
        color="red",
        density=True,
    )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Model Confidence Distribution")
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_contrast_set_results(
    results: Dict, output_path: Optional[str] = None
):
    """
    Plot contrast set evaluation results.

    Args:
        results: Results dictionary from ContrastSetEvaluator
        output_path: Path to save figure (optional)
    """
    if not results:
        logger.warning("No results provided")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    metrics = ["Original", "Contrast"]
    accuracies = [results["original_accuracy"], results["contrast_accuracy"]]

    axes[0].bar(metrics, accuracies, color=["skyblue", "lightcoral"])
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Original vs Contrast Accuracy")
    axes[0].set_ylim([0, 1])

    # Add value labels
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f"{v:.2%}", ha="center", fontweight="bold")

    # Consistency and sensitivity
    metrics2 = ["Consistency\n(Both Correct)", "Sensitivity\n(Pred Changed)"]
    values = [results["consistency"], results["sensitivity"]]

    axes[1].bar(metrics2, values, color=["lightgreen", "orange"])
    axes[1].set_ylabel("Proportion")
    axes[1].set_title("Consistency and Sensitivity")
    axes[1].set_ylim([0, 1])

    # Add value labels
    for i, v in enumerate(values):
        axes[1].text(i, v + 0.02, f"{v:.2%}", ha="center", fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_training_curves(
    log_file: str, output_path: Optional[str] = None
):
    """
    Plot training curves from logs.

    Args:
        log_file: Path to training log file (JSON or CSV)
        output_path: Path to save figure (optional)
    """
    # Load logs
    if log_file.endswith(".csv"):
        df = pd.read_csv(log_file)
    elif log_file.endswith(".json"):
        df = pd.read_json(log_file)
    else:
        logger.error(f"Unsupported log file format: {log_file}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    if "loss" in df.columns:
        axes[0].plot(df["step"], df["loss"], label="Training Loss", color="blue")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()

    # Plot accuracy
    if "accuracy" in df.columns:
        axes[1].plot(df["step"], df["accuracy"], label="Accuracy", color="green")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()


def create_comparison_table(
    baseline_results: Dict, mitigated_results: Dict
) -> pd.DataFrame:
    """
    Create comparison table between baseline and mitigated models.

    Args:
        baseline_results: Results from baseline model
        mitigated_results: Results from mitigated model

    Returns:
        DataFrame with comparison
    """
    data = {
        "Metric": [
            "Accuracy",
            "Error Rate",
            "Total Examples",
            "Correct",
            "Errors",
        ],
        "Baseline": [
            f"{baseline_results.get('accuracy', 0):.2%}",
            f"{baseline_results.get('error_rate', 0):.2%}",
            baseline_results.get("total_examples", 0),
            baseline_results.get("correct", 0),
            baseline_results.get("errors", 0),
        ],
        "Mitigated": [
            f"{mitigated_results.get('accuracy', 0):.2%}",
            f"{mitigated_results.get('error_rate', 0):.2%}",
            mitigated_results.get("total_examples", 0),
            mitigated_results.get("correct", 0),
            mitigated_results.get("errors", 0),
        ],
    }

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Example usage
    logger.info("Visualization module loaded")
