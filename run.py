#!/usr/bin/env python3
"""
Main training and evaluation script for ELECTRA NLP Artifact Analysis.

Supports:
- Natural Language Inference (NLI): SNLI, MultiNLI
- Question Answering (QA): SQuAD, HotpotQA
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate

# Import helpers from starter code (Greg Durrett's fp-dataset-artifacts)
from helpers import (
    prepare_dataset_nli,
    compute_accuracy,
    prepare_train_dataset_qa,
    prepare_validation_dataset_qa,
    QuestionAnsweringTrainer,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate ELECTRA on NLP benchmarks"
    )

    # Core arguments
    parser.add_argument(
        "--do_train", action="store_true", help="Run training"
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Run evaluation"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["nli", "qa"],
        help="Task type: nli or qa",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., snli, multi_nli, squad, hotpot_qa)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for checkpoints and results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/electra-small-discriminator",
        help="Model name or path",
    )

    # Training parameters
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Warmup steps"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )

    # Debug and optimization
    parser.add_argument(
        "--max_train_samples", type=int, default=None, help="Limit training samples"
    )
    parser.add_argument(
        "--max_eval_samples", type=int, default=None, help="Limit evaluation samples"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Force CPU-only training"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision training"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Checkpoint save frequency"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Evaluation frequency"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint"
    )

    return parser.parse_args()


def load_nli_dataset(dataset_name, max_train_samples=None, max_eval_samples=None):
    """Load NLI dataset (SNLI or MultiNLI)."""
    logger.info(f"Loading NLI dataset: {dataset_name}")

    dataset_mapping = {
        "snli": "snli",
        "multi_nli": "multi_nli",
        "mnli": "multi_nli",
    }

    hf_dataset = dataset_mapping.get(dataset_name.lower(), dataset_name)
    dataset = load_dataset(hf_dataset)

    # Handle dataset splits
    if "train" in dataset:
        train_dataset = dataset["train"]
        # Filter out invalid labels (SNLI has label=-1 for unlabeled examples)
        train_dataset = train_dataset.filter(lambda x: x["label"] != -1)
        if max_train_samples:
            train_dataset = train_dataset.select(range(max_train_samples))
    else:
        train_dataset = None

    if "validation" in dataset:
        eval_dataset = dataset["validation"]
    elif "validation_matched" in dataset:  # MultiNLI
        eval_dataset = dataset["validation_matched"]
    else:
        eval_dataset = None

    if eval_dataset:
        # Filter out invalid labels
        eval_dataset = eval_dataset.filter(lambda x: x["label"] != -1)
        if max_eval_samples:
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    return train_dataset, eval_dataset


def load_qa_dataset(dataset_name, max_train_samples=None, max_eval_samples=None):
    """Load QA dataset (SQuAD or HotpotQA)."""
    logger.info(f"Loading QA dataset: {dataset_name}")

    dataset_mapping = {
        "squad": "squad",
        "squad_v2": "squad_v2",
        "hotpot_qa": "hotpot_qa",
        "hotpotqa": "hotpot_qa",
    }

    hf_dataset = dataset_mapping.get(dataset_name.lower(), dataset_name)

    if hf_dataset == "hotpot_qa":
        dataset = load_dataset(hf_dataset, "fullwiki")
    else:
        dataset = load_dataset(hf_dataset)

    train_dataset = dataset["train"] if "train" in dataset else None
    eval_dataset = dataset["validation"] if "validation" in dataset else None

    if train_dataset and max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
    if eval_dataset and max_eval_samples:
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    return train_dataset, eval_dataset


# NOTE: preprocess_nli and preprocess_qa functions are now provided by helpers.py
# from the starter code (Greg Durrett's fp-dataset-artifacts)
# See: prepare_dataset_nli, prepare_train_dataset_qa, prepare_validation_dataset_qa

def _legacy_preprocess_qa(examples, tokenizer, max_seq_length=512):
    """Preprocess QA examples."""
    # SQuAD format: question, context -> answer
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        padding="max_length",
        max_length=max_seq_length,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
    )

    # Map answers to token positions
    offset_mapping = tokenized.pop("offset_mapping")
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]

        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find token positions
            sequence_ids = tokenized.sequence_ids(i)
            context_start = sequence_ids.index(1) if 1 in sequence_ids else 0
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1) if 1 in sequence_ids else len(sequence_ids) - 1

            # Check if answer is in context
            if offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char:
                idx = context_start
                while idx <= context_end and offsets[idx][0] <= start_char:
                    idx += 1
                start_position = idx - 1

                idx = context_end
                while idx >= context_start and offsets[idx][1] >= end_char:
                    idx -= 1
                end_position = idx + 1

                tokenized["start_positions"].append(start_position)
                tokenized["end_positions"].append(end_position)
            else:
                tokenized["start_positions"].append(0)
                tokenized["end_positions"].append(0)

    return tokenized


# NOTE: compute_nli_metrics and compute_qa_metrics are now provided by helpers.py
# NLI: Use compute_accuracy from helpers
# QA: Use QuestionAnsweringTrainer which handles postprocessing and metrics automatically

def _legacy_compute_qa_metrics(eval_pred: EvalPrediction):
    """LEGACY: Simplified QA metrics (replaced by helpers.py implementation).

    This was a simplified version that only compared token positions.
    The proper implementation from helpers.py:
    - Extracts actual answer text from context
    - Computes real F1 and Exact Match scores
    - Handles edge cases properly
    """
    start_logits, end_logits = eval_pred.predictions
    start_positions, end_positions = eval_pred.label_ids

    start_preds = np.argmax(start_logits, axis=-1)
    end_preds = np.argmax(end_logits, axis=-1)

    exact_match = ((start_preds == start_positions) & (end_preds == end_positions)).mean()

    return {"exact_match": exact_match}


def main():
    """Main training and evaluation function."""
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    if args.task == "nli":
        train_dataset, eval_dataset = load_nli_dataset(
            args.dataset, args.max_train_samples, args.max_eval_samples
        )
        num_labels = 3  # entailment, contradiction, neutral
    elif args.task == "qa":
        train_dataset, eval_dataset = load_qa_dataset(
            args.dataset, args.max_train_samples, args.max_eval_samples
        )
        num_labels = None  # QA doesn't use num_labels
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Preprocess datasets
    # Keep reference to raw eval data for QA postprocessing
    raw_eval_dataset = eval_dataset

    if args.task == "nli":
        if train_dataset:
            train_dataset = train_dataset.map(
                lambda x: prepare_dataset_nli(x, tokenizer, args.max_seq_length),
                batched=True,
                remove_columns=train_dataset.column_names,
            )
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                lambda x: prepare_dataset_nli(x, tokenizer, args.max_seq_length),
                batched=True,
                remove_columns=eval_dataset.column_names,
            )
    elif args.task == "qa":
        if train_dataset:
            train_dataset = train_dataset.map(
                lambda x: prepare_train_dataset_qa(x, tokenizer, args.max_seq_length),
                batched=True,
                remove_columns=train_dataset.column_names,
            )
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                lambda x: prepare_validation_dataset_qa(x, tokenizer),
                batched=True,
                remove_columns=eval_dataset.column_names,
            )

    # Load model and setup metrics
    if args.task == "nli":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=num_labels
        )
        compute_metrics = compute_accuracy
    elif args.task == "qa":
        model = AutoModelForQuestionAnswering.from_pretrained(args.model)
        # Load SQuAD metric for QA
        squad_metric = evaluate.load("squad")

        def compute_qa_metrics_squad(eval_preds):
            """Compute SQuAD F1 and EM metrics."""
            return squad_metric.compute(
                predictions=eval_preds.predictions,
                references=eval_preds.label_ids
            )

        compute_metrics = compute_qa_metrics_squad

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if args.do_eval else "no",
        save_total_limit=3,
        load_best_model_at_end=True if args.do_eval else False,
        metric_for_best_model="accuracy" if args.task == "nli" else "exact_match",
        fp16=args.fp16,
        no_cuda=args.no_cuda,
        seed=args.seed,
        report_to=["tensorboard"],
    )

    # Initialize trainer
    if args.task == "qa":
        # Use QuestionAnsweringTrainer for proper QA evaluation
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            eval_examples=raw_eval_dataset if args.do_eval else None,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    else:
        # Use standard Trainer for NLI
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if args.do_train else None,
            eval_dataset=eval_dataset if args.do_eval else None,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

    # Training
    if args.do_train:
        logger.info("*** Training ***")
        checkpoint = None
        if args.resume_from_checkpoint:
            checkpoint = args.resume_from_checkpoint
        elif os.path.isdir(args.output_dir):
            checkpoint = get_last_checkpoint(args.output_dir)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluating ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Done!")


if __name__ == "__main__":
    main()
