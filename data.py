"""Data generation utilities following Google Python Style."""

import json
import random
from typing import Dict, List, Union

import torch
from config import config
from tokenizer import Tokenizer


def generate_train_data(dataset_size: int) -> List[str]:
    """Generates training samples as simple addition equations.

    Args:
        dataset_size: Number of samples to generate.

    Returns:
        A list of strings like "a+b=c".
    """
    dataset: List[str] = []
    for _ in range(dataset_size):
        a = random.randint(0, 1000)
        b = random.randint(0, 1000)
        c = a + b
        dataset.append(f"{a}+{b}={c}")
    return dataset


train_dataset = generate_train_data(config.train_size)


def generate_test_data(dataset_size: int) -> List[Dict[str, str]]:
    """Generates test samples with input/output split.

    Args:
        dataset_size: Number of samples to generate.

    Returns:
        A list of dicts with keys "input" and "output".
    """
    dataset: List[Dict[str, str]] = []
    for _ in range(dataset_size):
        a = random.randint(0, 1000)
        b = random.randint(0, 1000)
        c = a + b
        dataset.append({"input": f"{a}+{b}=", "output": f"{c}"})
    return dataset


# 验证/测试数据集
val_dataset = generate_test_data(config.val_size)
test_dataset = generate_test_data(config.test_size)


class EquationDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that tokenizes arithmetic samples and creates targets.

    Each sample is converted to token ids with a leading <sop> and trailing
    <eop>. The inputs are the sequence without the last token, and the targets
    are the sequence without the first token (next-token prediction). Both are
    padded to a fixed length of ``config.max_seq_len - 1`` using ``pad_token_id``.

    Args:
        samples: List of training strings like "a+b=c" or dicts with
            keys ``input`` and ``output``.
        config: Global configuration providing vocab and max lengths.
    """

    def __init__(self, samples: List[Union[str, Dict[str, str]]], config):
        self.samples = samples
        self.config = config
        self.tokenizer = Tokenizer()

        # Precompute fixed output length (inputs/targets are one shorter than max_seq_len)
        self.fixed_length = max(1, self.config.max_seq_len - 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        if isinstance(sample, str):
            text = sample
        else:
            # Concatenate prompt and expected output for supervised next-token prediction
            text = f"{sample['input']}{sample['output']}"

        # Tokenize with special tokens
        ids = [self.tokenizer.sop_token_id] + self.tokenizer.encode(text) + [
            self.tokenizer.eop_token_id
        ]

        # Truncate to max_seq_len
        ids = ids[: self.config.max_seq_len]

        # Build input and target sequences (shift by one)
        input_ids = ids[:-1]
        target_ids = ids[1:]

        # Pad to fixed length
        pad_id = self.tokenizer.pad_token_id
        if len(input_ids) < self.fixed_length:
            pad_amount = self.fixed_length - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_amount
            target_ids = target_ids + [pad_id] * pad_amount
        else:
            # Ensure exact length
            input_ids = input_ids[: self.fixed_length]
            target_ids = target_ids[: self.fixed_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


if __name__ == "__main__":
    with open("train_dataset.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset, f)
    with open("test_dataset.json", "w", encoding="utf-8") as f:
        json.dump(test_dataset, f)