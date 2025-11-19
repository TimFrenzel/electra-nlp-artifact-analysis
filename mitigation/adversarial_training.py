"""
Adversarial training for improving model robustness.

Implements:
- Word substitution attacks
- Paraphrase generation
- Adversarial data augmentation
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class AdversarialTrainer:
    """
    Adversarial training using data augmentation.
    """

    def __init__(
        self,
        model_path: str = "google/electra-small-discriminator",
        num_labels: int = 3,
    ):
        """
        Initialize adversarial trainer.

        Args:
            model_path: Path to base model
            num_labels: Number of output labels
        """
        self.model_path = model_path
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = None

    def generate_word_swap_adversarial(
        self,
        text: str,
        num_swaps: int = 1,
    ) -> str:
        """
        Generate adversarial example by swapping words.

        Args:
            text: Original text
            num_swaps: Number of word swaps

        Returns:
            Adversarial text
        """
        words = text.split()

        if len(words) < 2:
            return text

        # Perform random swaps
        for _ in range(num_swaps):
            if len(words) < 2:
                break

            idx1 = random.randint(0, len(words) - 1)
            idx2 = random.randint(0, len(words) - 1)

            # Swap
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)

    def generate_synonym_replacement(
        self,
        text: str,
        num_replacements: int = 1,
    ) -> str:
        """
        Generate adversarial example by replacing words with synonyms.

        Note: This is a simplified version. For production, use WordNet or similar.

        Args:
            text: Original text
            num_replacements: Number of replacements

        Returns:
            Adversarial text
        """
        # Simple synonym dictionary (extend this for better results)
        synonyms = {
            "good": ["great", "excellent", "fine"],
            "bad": ["poor", "terrible", "awful"],
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "miniature"],
            "happy": ["joyful", "pleased", "delighted"],
            "sad": ["unhappy", "sorrowful", "dejected"],
        }

        words = text.split()
        replaced = 0

        for i, word in enumerate(words):
            if replaced >= num_replacements:
                break

            word_lower = word.lower()
            if word_lower in synonyms:
                synonym = random.choice(synonyms[word_lower])
                # Preserve case
                if word[0].isupper():
                    synonym = synonym.capitalize()
                words[i] = synonym
                replaced += 1

        return " ".join(words)

    def generate_negation_adversarial(self, text: str) -> str:
        """
        Generate adversarial example by adding/removing negation.

        Args:
            text: Original text

        Returns:
            Adversarial text with negation
        """
        # Simple negation rules
        if " not " in text:
            return text.replace(" not ", " ")
        elif " no " in text:
            return text.replace(" no ", " ")
        else:
            # Add negation
            words = text.split()
            if len(words) > 2:
                insert_pos = random.randint(1, len(words) - 1)
                words.insert(insert_pos, "not")
            else:
                return f"It is not the case that {text}"

            return " ".join(words)

    def augment_dataset(
        self,
        dataset: Dataset,
        augmentation_ratio: float = 0.5,
        methods: Optional[List[str]] = None,
    ) -> Dataset:
        """
        Augment dataset with adversarial examples.

        Args:
            dataset: Original dataset
            augmentation_ratio: Fraction of examples to augment
            methods: List of augmentation methods to use

        Returns:
            Augmented dataset
        """
        if methods is None:
            methods = ["word_swap", "synonym", "negation"]

        logger.info(f"Augmenting dataset with {augmentation_ratio:.0%} additional examples...")

        augmented_examples = []
        num_to_augment = int(len(dataset) * augmentation_ratio)

        # Randomly select examples to augment
        indices = random.sample(range(len(dataset)), min(num_to_augment, len(dataset)))

        for idx in indices:
            example = dataset[idx]
            method = random.choice(methods)

            # Generate adversarial hypothesis
            if method == "word_swap":
                adv_hypothesis = self.generate_word_swap_adversarial(
                    example["hypothesis"], num_swaps=1
                )
            elif method == "synonym":
                adv_hypothesis = self.generate_synonym_replacement(
                    example["hypothesis"], num_replacements=1
                )
            elif method == "negation":
                adv_hypothesis = self.generate_negation_adversarial(
                    example["hypothesis"]
                )
            else:
                continue

            # Create augmented example
            # Note: Label might need adjustment based on perturbation
            augmented_example = {
                "premise": example["premise"],
                "hypothesis": adv_hypothesis,
                "label": example["label"],  # Simplified - should be relabeled
            }

            augmented_examples.append(augmented_example)

        logger.info(f"Generated {len(augmented_examples)} adversarial examples")

        # Combine original and augmented
        if augmented_examples:
            # Convert to Dataset
            augmented_dataset = Dataset.from_dict({
                "premise": [ex["premise"] for ex in augmented_examples],
                "hypothesis": [ex["hypothesis"] for ex in augmented_examples],
                "label": [ex["label"] for ex in augmented_examples],
            })

            # Concatenate
            from datasets import concatenate_datasets
            combined_dataset = concatenate_datasets([dataset, augmented_dataset])
            logger.info(f"Combined dataset size: {len(combined_dataset)}")

            return combined_dataset
        else:
            return dataset

    def train_with_adversarial_examples(
        self,
        dataset: Dataset,
        output_dir: str,
        augmentation_ratio: float = 0.5,
        num_epochs: int = 3,
        batch_size: int = 32,
    ):
        """
        Train model with adversarial data augmentation.

        Args:
            dataset: Training dataset
            output_dir: Output directory
            augmentation_ratio: Ratio of adversarial examples to add
            num_epochs: Number of training epochs
            batch_size: Training batch size
        """
        logger.info("Training with adversarial examples...")

        # Augment dataset
        augmented_dataset = self.augment_dataset(dataset, augmentation_ratio)

        # Preprocess
        def preprocess(examples):
            return self.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=512,
            )

        processed_dataset = augmented_dataset.map(
            preprocess,
            batched=True,
            remove_columns=augmented_dataset.column_names,
        )

        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
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
            evaluation_strategy="no",
        )

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(output_dir)

        logger.info(f"Adversarially trained model saved to {output_dir}")


def generate_adversarial_examples(
    dataset: Dataset,
    method: str = "word_swap",
    num_examples: int = 100,
) -> List[Dict]:
    """
    Generate adversarial examples from dataset.

    Args:
        dataset: Source dataset
        method: Augmentation method
        num_examples: Number of examples to generate

    Returns:
        List of adversarial examples
    """
    trainer = AdversarialTrainer()

    adversarial = []
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))

    for idx in indices:
        example = dataset[idx]

        if method == "word_swap":
            adv_text = trainer.generate_word_swap_adversarial(example["hypothesis"])
        elif method == "synonym":
            adv_text = trainer.generate_synonym_replacement(example["hypothesis"])
        elif method == "negation":
            adv_text = trainer.generate_negation_adversarial(example["hypothesis"])
        else:
            continue

        adversarial.append({
            "premise": example["premise"],
            "hypothesis": adv_text,
            "label": example["label"],
            "original_hypothesis": example["hypothesis"],
            "method": method,
        })

    return adversarial


class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) attack for generating adversarial examples.
    """

    def __init__(self, model, tokenizer, epsilon: float = 0.01):
        """
        Initialize FGSM attack.

        Args:
            model: Target model
            tokenizer: Tokenizer
            epsilon: Perturbation magnitude
        """
        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.model.eval()

    def generate_adversarial(
        self,
        premise: str,
        hypothesis: str,
        label: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate adversarial example using FGSM.

        Args:
            premise: Premise text
            hypothesis: Hypothesis text
            label: True label

        Returns:
            Tuple of (original_inputs, adversarial_inputs)
        """
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        label_tensor = torch.tensor([label])

        # Get embeddings
        embeddings = self.model.get_input_embeddings()
        input_embeds = embeddings(inputs["input_ids"])
        input_embeds.requires_grad = True

        # Forward pass
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=inputs["attention_mask"],
            labels=label_tensor,
        )

        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Generate adversarial perturbation
        perturbation = self.epsilon * input_embeds.grad.sign()
        adversarial_embeds = input_embeds + perturbation

        return input_embeds.detach(), adversarial_embeds.detach()


if __name__ == "__main__":
    # Example usage
    import sys
    from datasets import load_dataset

    if len(sys.argv) < 2:
        print("Usage: python adversarial_training.py <dataset>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    output_dir = "./models/adversarial_model"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    if dataset_name.lower() == "snli":
        dataset = load_dataset("snli")["train"]
    else:
        dataset = load_dataset(dataset_name)["train"]

    # Filter invalid labels
    dataset = dataset.filter(lambda x: x["label"] != -1)

    # Limit for demo
    dataset = dataset.select(range(min(10000, len(dataset))))

    # Train with adversarial augmentation
    trainer = AdversarialTrainer()
    trainer.train_with_adversarial_examples(
        dataset,
        output_dir,
        augmentation_ratio=0.3,
        num_epochs=3,
    )

    print(f"Adversarial training complete. Model saved to {output_dir}")
