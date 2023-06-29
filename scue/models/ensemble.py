from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .abc import BaseModel
from .lexical_overlap import LexicalOverlapModelForMultipleChoice
from .sequence_length import SequenceLengthModelForMultipleChoice
from .unbalanced_ngrams import UnbalancedNgramModelForMultipleChoice


@dataclass
class EnsembleModelForMultipleChoice(BaseModel):
    training_data: list[dict[Any, Any]]
    id_field: str = "id"
    label_field: str = "label"
    choices_field: str = "choices"
    context_field: str = "context"
    question_field: str | None = None
    metric: str = "applicability"
    n: int = 1
    smoothing: float = 3.0

    def __post_init__(self, *args, **kwargs):
        print("Preprocessing data for Sequence Length and Lexical Overlap...")

        self._len_model = SequenceLengthModelForMultipleChoice(
            copy.deepcopy(self.training_data),
            id_field=self.id_field,
            label_field=self.label_field,
            choices_field=self.choices_field,
            context_field=self.context_field,
            question_field=self.question_field,
            n=self.n,
        )

        self._overlap_model = LexicalOverlapModelForMultipleChoice(
            copy.deepcopy(self.training_data),
            id_field=self.id_field,
            label_field=self.label_field,
            choices_field=self.choices_field,
            context_field=self.context_field,
            question_field=self.question_field,
            n=self.n,
        )

        self._unbalanced_model = UnbalancedNgramModelForMultipleChoice(
            copy.deepcopy(self.training_data),
            id_field=self.id_field,
            label_field=self.label_field,
            choices_field=self.choices_field,
            context_field=self.context_field,
            question_field=self.question_field,
            n=self.n,
            metric=self.metric,
            smoothing=self.smoothing,
        )
        self._models = [self._len_model, self._overlap_model, self._unbalanced_model]

    def fit(self):
        [model.fit() for model in self._models]
        random_chance = 1 / len(self.training_data[0][self.choices_field])
        model_quality = [
            model.evaluate(self.training_data)["acc"] for model in self._models
        ]
        pruned_scores = []
        pruned_models = []
        for i, score in enumerate(model_quality):
            # keep at least one model
            if i == len(self._models) - 1 and len(pruned_scores) == 0:
                break

            if score > random_chance:
                pruned_scores.append(score)
                pruned_models.append(self._models[i])

        self._most_trusted_model_index = pruned_scores.index(max(pruned_scores))
        self._pruned_models = pruned_models

    def predict(self, data: list[dict[Any, Any]]) -> list[int]:
        predictions = []
        for row in data:
            preds = [model.predict([row]) for model in self._pruned_models]
            if len(preds) == 1:
                predictions.append(preds[0])
            else:
                votes = Counter(preds)
                choice, majority = votes.most_common(1)[0]
                if majority > 1:
                    print(f"Majority: {majority}, Choice: {choice}")
                    predictions.append(choice)
                else:
                    # if there is no majority, use the most trusted model
                    predictions.append(preds[self._most_trusted_model_index])
        return predictions

    def evaluate(self, data: list[dict[Any, Any]]) -> dict[str, float]:
        return self._models[self._most_trusted_model_index].evaluate(data)

    def save(self):
        pass

    def load(self):
        pass

    def important_features(self) -> list[str]:
        return self._pruned_models[self._most_trusted_model_index].important_features()
