from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import spacy
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

from ..utils import CORRECT_TOKEN, INCORRECT_TOKEN


def create_token_list(
    doc, return_lemma, remove_stopwords=True, remove_punctuation=True
) -> list[str]:
    if return_lemma:
        return [
            token.lemma_
            for token in doc
            if not (remove_stopwords and token.is_stop)
            and not (remove_punctuation and token.is_punct)
        ]

    return [
        token.text
        for token in doc
        if not (remove_stopwords and token.is_stop)
        and not (remove_punctuation and token.is_punct)
    ]


@dataclass
class Prepocessor:
    id_field: str = "id"
    label_field: str = "label"
    choices_field: str = "choices"
    context_field: str = "context"
    question_field: str | None = None
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    lowercase: bool = True
    return_lemma: bool = False

    def __post_init__(self, *args, **kwargs):
        self.data = []
        self.label_counts = Counter()
        self.tokenizer = None
        try:
            self.tokenizer = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Downloading en_core_web_sm")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.tokenizer = spacy.load("en_core_web_sm")

    def _load_data(self, data_path: str):
        data = []
        with open(data_path, "r") as file:
            for line in file:
                item = json.loads(line)
                self.label_counts.update([item[self.label_field]])
                data.append(
                    {
                        "id": item[self.id_field],
                        "label": item[self.label_field],
                        "choices": [
                            {
                                "text": val,
                                "label": idx == item[self.label_field],
                                "correct": idx == item[self.label_field],
                                "label_str": CORRECT_TOKEN
                                if item[self.label_field]
                                else INCORRECT_TOKEN,
                            }
                            for idx, val in enumerate(item[self.choices_field])
                        ],
                        "context": {
                            "text": "\n".join(
                                item[field]
                                for field in [
                                    self.context_field,
                                    self.question_field,
                                ]
                                if field is not None
                            ),
                        },
                    }
                )
        return data

    def _preprocess(self, data):
        nlp = self.tokenizer

        if len(data) > 10000:
            batch_size = 1000
        elif len(data) > 1000:
            batch_size = 100
        elif len(data) > 100:
            batch_size = 32
        else:
            batch_size = 1

        text_tuples = []
        for item in data:
            if "context" in item:
                if self.lowercase:
                    text_tuples.append(
                        (item["context"]["text"].lower(), item["context"])
                    )
                else:
                    text_tuples.append((item["context"]["text"], item["context"]))

            for choice in item["choices"]:
                if self.lowercase:
                    text_tuples.append((choice["text"].lower(), choice))
                else:
                    text_tuples.append((choice["text"], choice))

        doc_tuples = nlp.pipe(
            text_tuples,
            disable=["ner", "parse"],
            batch_size=batch_size,
            as_tuples=True,
            n_process=1,
        )

        for doc, context in tqdm(doc_tuples, desc="Tokenizing", total=len(text_tuples)):
            context["tokens"] = create_token_list(
                doc,
                return_lemma=self.return_lemma,
                remove_stopwords=self.remove_stopwords,
                remove_punctuation=self.remove_punctuation,
            )

        return data

    def _extract_ngrams(self):
        pass

    def run(self, data_path: str) -> List[dict[Any, Any]]:
        data = self._load_data(data_path)
        data = self._preprocess(data)
        return data

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)
