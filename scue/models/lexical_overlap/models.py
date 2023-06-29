from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import gensim.downloader as gensim_api
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from ...models.abc import BaseModel
from ...token import Token


@dataclass
class LexicalOverlapModelForMultipleChoice(BaseModel):
    training_data: list[dict[Any, Any]]
    id_field: str = "id"
    label_field: str = "label"
    choices_field: str = "choices"
    context_field: str = "context"
    question_field: str | None = None
    n: int = 1
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    lowercase: bool = True
    matching_method: str = "exact"

    def __post_init__(self, *args, **kwargs):
        print("Loading word2vec model...")
        self.model = gensim_api.load("word2vec-google-news-300")

    def _extract_ngrams(self, data):
        for item in tqdm(data, desc="Extracting ngrams"):
            context = item[self.choices_field]
            context["ngrams"] = self._get_ngrams(context["tokens"], self.n)

            for answer in item[self.choices_field]:
                answer["ngrams"] = self._get_ngrams(answer["tokens"], self.n)

    def _tokens_to_text(self, tokens: list[str]) -> str:
        return " ".join(token for token in tokens)

    def _get_distances(self, context, choices):
        distances: list[float] = []

        context = self._tokens_to_text(context["tokens"])

        for choice in choices:
            distance = self.model.wmdistance(
                context, self._tokens_to_text(choice["tokens"])
            )
            if distance == np.inf:
                distances.append(10)
            else:
                distances.append(distance)
        return distances

    def _get_training_distances(self):
        distances: dict[str, list[float]] = defaultdict(list)

        for item in tqdm(self.training_data):
            choice_distances = self._get_distances(
                item[self.context_field], item[self.choices_field]
            )
            for idx, choice in enumerate(item[self.choices_field]):
                if idx == item[self.label_field]:
                    distances["correct_choices"].append(choice_distances[idx])
                else:
                    distances["incorrect_choices"].append(choice_distances[idx])
        return distances

    def _get_correct_rank(self, data_points: dict[str, Any]) -> int:
        sorted_choices = sorted(data_points["choices"], reverse=True)
        correct_choice = data_points["choices"][data_points["label"]]
        rank = sorted_choices.index(correct_choice) + 1

        return rank

    def _get_correct_ranks(self, data):
        ranks = []
        for item in tqdm(data):
            ranks.append(self._get_correct_rank(item))
        return ranks

    def _format_data(self, data):
        formatted_data = []
        for item in data:
            formatted_data.append(
                {
                    "id": item[self.id_field] if self.id_field else uuid.uuid4(),
                    "label": item[self.label_field],
                    "choices": self._get_distances(
                        item[self.context_field], item[self.choices_field]
                    ),
                }
            )
        return formatted_data

    def fit(self):
        data = self._format_data(self.training_data)
        self.ranks = self._get_correct_ranks(data)
        self.most_common_rank = Counter(self.ranks).most_common(1)[0][0]
        return self

    def important_features(self):
        return Counter(self.ranks).most_common()

    def _predict(self, choice_lengths):
        common_rank = self.most_common_rank
        sorted_choices = sorted(choice_lengths, reverse=True)
        correct_score = sorted_choices[common_rank - 1]
        return choice_lengths.index(correct_score)

    def predict(self, data):
        predictions = []
        data = self._format_data(data)
        for item in tqdm(data, desc="Predicting"):
            predicton = self._predict(item[self.choices_field])
            predictions.append(predicton)
        return predictions

    def evaluate(self, data: list[Any]) -> dict[str, float]:
        predictions = self.predict(data)
        labels = [item[self.label_field] for item in data]

        return {
            "acc": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="macro"),
            "precision": precision_score(labels, predictions, average="macro"),
            "recall": recall_score(labels, predictions, average="macro"),
        }

    def load(self):
        pass

    def save(self):
        pass


@dataclass
class LexicalOverlapDecisionModelForMultipleChoice(BaseModel):
    training_data: list[dict[Any, Any]]
    id_field: str = "ind"
    label_field: str = "label"
    choices_field: str = "choices"
    context_field: str = "context"
    question_field: str | None = None
    n: int = 1
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    lowercase: bool = True
    matching_method: str = "exact"

    def __post_init__(self, *args, **kwargs):
        self.model = DecisionTreeClassifier()

    def _extract_ngrams(
        self,
        data: dict,
    ):
        for row in data:
            row["context"]["ngrams"] = self._get_ngrams(
                row["context"]["tokens"], self.n
            )
            for choice in row["choices"]:
                choice["ngrams"] = self._get_ngrams(choice["tokens"], self.n)

    def _extract_ngram_overlap_features(
        self,
        data: dict,
    ) -> tuple[list[list[int]], list[int]]:
        features = []
        for choice in data["choices"]:
            overlap_features: list[int] = []
            for n in range(1, self.n + 1):
                context_ngram_counter = Counter(
                    ngram for ngram in data["context"]["ngrams"] if len(ngram) == n
                )
                choice_ngrams = Counter(
                    ngram for ngram in choice["ngrams"] if len(ngram) == n
                )

                overlaps = context_ngram_counter & choice_ngrams
                overlap_features.append(sum(overlaps.values()))
            features.append(overlap_features)

        return features, data["label"]

    def _extract_features(self, data: dict) -> tuple[list[list[int]], list[int]]:
        features = []
        labels = []
        for row in data:
            feature, label = self._extract_ngram_overlap_features(row)
            feature = list(chain(*feature))
            features.append(feature)
            labels.append(label)
        return features, labels

    def fit(self):
        self._extract_ngrams(self.training_data)
        features, labels = self._extract_features(self.training_data)
        self.model.fit(features, labels)
        return self

    def important_features(self):
        pass

    def evaluate(self, data):
        self._extract_ngrams(data)
        features, labels = self._extract_features(data)
        predictions = self.model.predict(features)
        results = {
            "acc": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions),
        }
        return results

    def predict(self, data):
        self._extract_ngrams(data)
        features, labels = self._extract_features(data)
        return self.model.predict(features)

    def load(self):
        pass

    def save(self):
        pass
