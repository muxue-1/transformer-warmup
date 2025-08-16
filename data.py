"""Data generation utilities following Google Python Style."""

import json
import random
from typing import Dict, List


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


train_dataset = generate_train_data(10000)


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


test_dataset = generate_test_data(1000)


if __name__ == "__main__":
    with open("train_dataset.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset, f)
    with open("test_dataset.json", "w", encoding="utf-8") as f:
        json.dump(test_dataset, f)