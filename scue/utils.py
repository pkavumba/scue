from __future__ import annotations

import random
from typing import List, Tuple
import numpy as np

from .scue import (
    get_vocab_size_total_counts,
    compute_pmi,
    count_label_occurrences,
    count_ngram_occurrences,
    count_ngram_cooccurrences,
    fast_count_ngram_cooccurrences,
    get_all_ngrams,
    get_n_from_ngram_list,
    get_n_from_ngrams,
    get_ngrams,
    count_all_ngram_instance_occurrences,
    count_unique_ngrams_in_correct_and_wrong_answers,
    get_nivens_applicability,
    get_vocab_size_total_counts,
    compute_pmi_for_all_ngrams,
    compute_pmi_for_all_keys,
)

CORRECT_TOKEN = "<CORRECT>"
INCORRECT_TOKEN = "<INCORRECT>"


def train_test_split(
    data: List[dict], test_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """
    Randomly splits a list of JSON instances into train and test sets.

    Args:
        data (List[dict]): A list of JSON instances.
        test_ratio (float, optional): The proportion of instances to include in the test set. Default is 0.2.
        seed (int, optional): Seed for the random number generator. Default is 42.

    Returns:
        Tuple[List[dict], List[dict]]: A tuple containing the train and test sets.
    """
    random.seed(seed)
    random.shuffle(data)

    test_size = int(len(data) * test_ratio)

    test_data = data[:test_size]
    train_data = data[test_size:]

    return train_data, test_data


def compute_accuracy(labels: list[int], predictions: list[int]):
    """
    Computes the accuracy of a list of predictions.

    Args:
        predictions (list[int]): A list of predictions.
        labels (list[int]): A list of labels.

    Returns:
        float: The accuracy of the predictions.
    """
    preds, answers = np.array(predictions), np.array(labels)

    return 100 * (preds == answers).mean()
