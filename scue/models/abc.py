from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..token import Token
from ..scue import get_all_ngrams


@dataclass
class BaseModel(ABC):
    def _get_ngrams(self, tokens: list[Token], ngrams: int):
        return get_all_ngrams(tokens, ngrams)

    @property
    @abstractmethod
    def important_features(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass
