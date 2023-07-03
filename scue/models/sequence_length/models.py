from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from ...utils import compute_accuracy
from ..abc import BaseModel


@dataclass
class SequenceLengthModelForMultipleChoice(BaseModel):
    training_data: list[dict[Any, Any]]
    n: int = 1
    id_field: str = "id"
    label_field: str = "label"
    choices_field: str = "choices"
    context_field: str = "context"
    question_field: str | None = None

    def __post_init__(self) -> None:
        pass

    def _extract_ngrams(self, data):
        for item in tqdm(data, desc="Extracting ngrams"):
            for answer in item[self.choices_field]:
                answer["ngrams"] = self._get_ngrams(answer["tokens"], self.n)

    def _get_choices_lengths(self, choices) -> list[int]:
        return [len(choice["tokens"]) for choice in choices]

    def _format_data(self, data):
        formatted_data = []
        for item in data:
            formatted_data.append(
                {
                    "id": item[self.id_field] if self.id_field else uuid.uuid4(),
                    "label": item[self.label_field],
                    "choices": self._get_choices_lengths(item[self.choices_field]),
                }
            )
        return formatted_data

    def _get_correct_rank(self, data_point: dict[str, Any]) -> int:
        choices = [(choice, idx) for idx, choice in enumerate(data_point["choices"])]
        sorted_choices = sorted(choices, key=lambda x: x[0], reverse=True)

        for rank, (choice, idx) in enumerate(sorted_choices):
            if idx == data_point["label"]:
                return rank + 1

    def _get_correct_ranks(self, data):
        ranks = []
        for item in tqdm(data):
            ranks.append(self._get_correct_rank(item))
        return ranks

    def _predict(self, choice_lengths):
        common_rank = self.most_common_rank
        choices = [(choice_len, idx) for idx, choice_len in enumerate(choice_lengths)]
        sorted_choices = sorted(choices, key=lambda x: x[0], reverse=True)
        _, idx = sorted_choices[common_rank - 1]
        return idx

    def fit(self):
        self._extract_ngrams(self.training_data)
        data = self._format_data(self.training_data)
        self.ranks = self._get_correct_ranks(data)
        self.most_common_rank = Counter(self.ranks).most_common(1)[0][0]

        return self

    def predict(self, data):
        predictions = []
        self._extract_ngrams(data)
        data = self._format_data(data)
        for item in tqdm(data, desc="Predicting"):
            predicton = self._predict(item[self.choices_field])
            predictions.append(predicton)
        return predictions

    def evaluate(self, data: list[Any]) -> dict[str, float]:
        predictions = self.predict(data)
        labels = [item[self.label_field] for item in data]

        return {"acc": compute_accuracy(labels, predictions)}

    def load(self):
        pass

    @property
    def important_features(self):
        return Counter(self.ranks)

    def save(self):
        pass
