from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from ...utils import (
    CORRECT_TOKEN,
    INCORRECT_TOKEN,
    compute_pmi,
    compute_pmi_for_all_keys,
    compute_pmi_for_all_ngrams,
    count_unique_ngrams_in_correct_and_wrong_answers,
    fast_count_ngram_cooccurrences,
    get_nivens_applicability,
    get_vocab_size_total_counts,
)
from ..abc import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class UnbalancedNgramModelForMultipleChoice(BaseModel):
    training_data: list[dict[Any, Any]]
    id_field: str = "ind"
    label_field: str = "label"
    choices_field: str = "choices"
    context_field: str = "context"
    question_field: str | None = None
    metric: str = "applicability"
    n: int = 1
    smoothing: float = 3.0

    def __post_init__(self, *args, **kwargs):
        if self.metric == "applicability":
            self.model = ApplicabilityNgramModelForMultipleChoice(
                training_data=self.training_data,
                id_field=self.id_field,
                label_field=self.label_field,
                choices_field=self.choices_field,
                context_field=self.context_field,
                question_field=self.question_field,
                n=self.n,
            )
        elif self.metric == "pmi":
            self.model = PPMINgramModelForMultipleChoice(
                training_data=self.training_data,
                id_field=self.id_field,
                label_field=self.label_field,
                choices_field=self.choices_field,
                context_field=self.context_field,
                question_field=self.question_field,
                n=self.n,
                smoothing=self.smoothing,
            )
        if self.metric not in {"pmi", "applicability"}:
            raise ValueError("Invalid metric")

    def fit(self):
        self.model.fit()
        return self

    def important_features(self):
        return self.model.important_features()

    def evaluate(self, data):
        return self.model.evaluate(data)

    def predict(self, data):
        return self.model.predict(data)

    def load(self):
        pass

    def save(self):
        pass


@dataclass(frozen=True)
class NgramApplicability:
    ngram: str
    applicability: int
    productivity: float
    coverage: float

    def __hash__(self) -> int:
        return hash(self.ngram)

    def __eq__(self, other: Any) -> bool:
        return self.ngram == other.ngram

    def __lt__(self, other: Any) -> bool:
        return self.coverage < other.coverage

    def __gt__(self, other: Any) -> bool:
        return self.coverage > other.coverage


@dataclass
class ApplicabilityNgramModelForMultipleChoice(BaseModel):
    training_data: list[dict[Any, Any]]
    id_field: str = "ind"
    label_field: str = "label"
    choices_field: str = "choices"
    context_field: str = "context"
    question_field: str | None = None
    metric: str = "applicability"
    n: int = 1
    smoothing: float = 3.0

    def __post_init__(self, *args, **kwargs):
        if self.metric != "applicability":
            raise ValueError("Invalid metric")

    def _extract_ngrams(self, data):
        for item in tqdm(data, desc="Extracting ngrams"):
            for answer in item[self.choices_field]:
                answer["ngrams"] = self._get_ngrams(answer["tokens"], self.n)

    def _analyze_training_set_applicability(self):
        logger.info("Analyzing dataset applicability stats")
        print("Analyzing dataset applicability stats")
        self.ngram_applicability = get_nivens_applicability(
            json.dumps(self.training_data)
        )

        (
            self.applicable_correct_ngram_counter,
            self.applicable_wrong_ngram_counter,
        ) = count_unique_ngrams_in_correct_and_wrong_answers(
            json.dumps(self.training_data)
        )

    def _compute_productivities_statistics(self):
        self.correct_applicability_statistics = OrderedDict()
        self.wrong_applicability_statistics = OrderedDict()

        for ngram, count in sorted(
            self.applicable_correct_ngram_counter.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            self.correct_applicability_statistics[ngram] = NgramApplicability(
                ngram=ngram,
                applicability=count,
                productivity=100 * count / self.ngram_applicability[ngram],
                coverage=100
                * self.ngram_applicability[ngram]
                / len(self.training_data),
            )

        for ngram, count in sorted(
            self.applicable_wrong_ngram_counter.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            self.wrong_applicability_statistics[ngram] = NgramApplicability(
                ngram=ngram,
                applicability=count,
                productivity=100 * count / self.ngram_applicability[ngram],
                coverage=100
                * self.ngram_applicability[ngram]
                / len(self.training_data),
            )

    def most_productivity_ngrams(
        self, n: int = 10
    ) -> dict[str, list[NgramApplicability]]:
        correct_ngrams = []
        for stats_obj in sorted(
            self.correct_applicability_statistics.values(),
            key=lambda x: x.productivity,
            reverse=True,
        )[:n]:
            correct_ngrams.append(stats_obj)

        wrong_ngrams = []
        for stats_obj in sorted(
            self.wrong_applicability_statistics.values(),
            key=lambda x: x.productivity,
            reverse=True,
        )[:n]:
            wrong_ngrams.append(stats_obj)

        return {"correct": correct_ngrams, "wrong": wrong_ngrams}

    def widest_coverage_ngrams(
        self, n: int = 10
    ) -> dict[str, list[NgramApplicability]]:
        correct_ngrams = []
        for stats_obj in sorted(
            self.correct_applicability_statistics.values(),
            key=lambda x: x.coverage,
            reverse=True,
        )[:n]:
            correct_ngrams.append(stats_obj)

        wrong_ngrams = []
        for stats_obj in sorted(
            self.wrong_applicability_statistics.values(),
            key=lambda x: x.coverage,
            reverse=True,
        )[:n]:
            wrong_ngrams.append(stats_obj)

        return {"correct": correct_ngrams, "wrong": wrong_ngrams}

    def estimate_correct_answer(self, test_instance):
        choice_applicability_values = []
        UNK_TOKEN = NgramApplicability(
            ngram="UNK",
            applicability=0,
            productivity=0,
            coverage=0,
        )

        for choice in test_instance["choices"]:
            choice_applicability = sum(
                self.correct_applicability_statistics.get(ngram, UNK_TOKEN).productivity
                for ngram in choice["ngrams"]
            ) - sum(
                self.wrong_applicability_statistics.get(ngram, UNK_TOKEN).productivity
                for ngram in choice["ngrams"]
            )
            choice_applicability_values.append(choice_applicability)
        estimated_answer_key = choice_applicability_values.index(
            max(choice_applicability_values)
        )
        return {"answer": estimated_answer_key, "score": choice_applicability_values}

    def fit(self):
        self._extract_ngrams(self.training_data)
        self._analyze_training_set_applicability()
        self._compute_productivities_statistics()
        return self

    def important_features(self):
        return {
            "correctness": self.most_productivity_ngrams(n=100),
            "incorrectness": self.widest_coverage_ngrams(n=100),
        }

    def evaluate(self, data):
        predictions = self.predict(data)[0]
        labels = [item["label"] for item in data]
        results = {
            "acc": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions),
            "f1": f1_score(labels, predictions),
        }
        return results

    def predict(self, data):
        predictions = []
        applicability_scores = []
        self._extract_ngrams(data)
        for item in data:
            for choice in item["choices"]:
                choice["ngram_list"] = self._get_ngrams(choice["tokens"], self.n)
            predicted_answer = self.estimate_correct_answer(item)
            predictions.append(predicted_answer["answer"])
            applicability_scores.append(predicted_answer["score"])
        return predictions, applicability_scores

    def load(self):
        pass

    def save(self):
        pass


@dataclass
class PPMINgramModelForMultipleChoice(BaseModel):
    training_data: list[dict[Any, Any]]
    id_field: str = "ind"
    label_field: str = "label"
    choices_field: str = "choices"
    context_field: str = "context"
    question_field: str | None = None
    n: int = 1
    smoothing: float = 3.0
    metric: str = "pmi"

    def __post_init__(self, *args, **kwargs):
        if self.metric != "pmi":
            raise ValueError("Invalid metric")
        self.ppmi = {}

    def _extract_ngrams(self, data):
        for item in tqdm(data, desc="Extracting ngrams"):
            for answer in item[self.choices_field]:
                answer["ngrams"] = self._get_ngrams(answer["tokens"], self.n)

    def _calculate_ngram_counts(self, data):
        self.ngram_coocurrence_counts = fast_count_ngram_cooccurrences(json.dumps(data))
        data = get_vocab_size_total_counts(self.ngram_coocurrence_counts)
        self.vocab_size = data["vocab_size"]
        self.total_size = data["total_size"]

    def _pmi(self, ngram, label, smoothing=0):
        if (ngram, label) in self.ppmi:
            return self.ppmi[(ngram, label)]

        pmi = compute_pmi(
            self.ngram_coocurrence_counts,
            ngram,
            label,
            self.vocab_size,
            self.total_size,
            smoothing,
        )

        return pmi

    def _calculate_training_pmi_values(self):
        print("Calculating PMI values for training data")
        return compute_pmi_for_all_keys(
            self.ngram_coocurrence_counts,
            self.vocab_size,
            self.total_size,
            self.smoothing,
        )

    def estimate_correct_answer(self, test_instance):
        choice_pmi_values = []

        for choice in test_instance["choices"]:
            choice_pmi_value = sum(
                self._pmi(ngram, CORRECT_TOKEN, smoothing=self.smoothing)
                for ngram in choice["ngrams"]
            ) - sum(
                self._pmi(ngram, INCORRECT_TOKEN, smoothing=self.smoothing)
                for ngram in choice["ngrams"]
            )
            choice_pmi_values.append(choice_pmi_value)
        estimated_answer_key = choice_pmi_values.index(max(choice_pmi_values))
        return {"answer": estimated_answer_key, "pmi": choice_pmi_values}

    def fit(self):
        self._extract_ngrams(self.training_data)
        self._calculate_ngram_counts(self.training_data)
        self.ppmi = self._calculate_training_pmi_values()

        return self

    def important_features(self):
        return sorted(self.ppmi, key=self.ppmi.get, reverse=True)[:100]

    def evaluate(self, data):
        predictions = self.predict(data)[0]
        labels = [item["label"] for item in data]
        return {
            "acc": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions),
            "f1": f1_score(labels, predictions),
        }

    def predict(self, data):
        self._extract_ngrams(data)
        predictions = []
        pmi_scores = []
        for item in tqdm(data, desc="Predicting"):
            predicted_answer = self.estimate_correct_answer(item)
            predictions.append(predicted_answer["answer"])
            pmi_scores.append(predicted_answer["pmi"])
        return predictions, pmi_scores

    def load(self):
        pass

    def save(self):
        pass
