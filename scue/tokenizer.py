from __future__ import annotations

import functools
import string
from dataclasses import dataclass

import nltk
import numba
import spacy
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from .token import Token

nlp = spacy.load(
    "en_core_web_sm",
    exclude=["ner", "tok2vec"],
)


@dataclass
class Tokenizer:
    lowercase: bool = True
    remove_stopwords: bool = True
    remove_punctuation: bool = True

    def __call__(self, text: str) -> list[Token]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> list[Token]:
        raise NotImplementedError


@dataclass
class NLTKTokenizer(Tokenizer):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    stop_words = set(nltk.corpus.stopwords.words("english"))
    punctuations = set(string.punctuation)

    _lemma_cache = {}
    _stem_cache = {}

    def __call__(self, text: str) -> list[Token]:
        return self.tokenize(text)

    def _pos(self, pos: str) -> str:
        if pos in {"NN", "NNS", "NNP", "NNPS"}:
            return "NOUN"
        elif pos in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
            return "VERB"
        elif pos in {"JJ", "JJR", "JJS"}:
            return "ADJ"
        elif pos in {"RB", "RBR", "RBS"}:
            return "ADV"
        elif pos in {"CD"}:
            return "NUM"

        else:
            return pos

    def _lemmatize(self, word):
        if word not in self._lemma_cache:
            self._lemma_cache[word] = self.lemmatizer.lemmatize(word)
        return self._lemma_cache[word]

    def _stem(self, word):
        if word not in self._stem_cache:
            self._stem_cache[word] = self.stemmer.stem(word)
        return self._stem_cache[word]

    def tokenize(self, text: str) -> list[Token]:
        if self.lowercase:
            text = text.lower()
        word_tokens = pos_tag(word_tokenize(text))

        tokens = []
        for text, pos in word_tokens:
            is_stop = text in self.stop_words
            is_punct = text in self.punctuations

            if ((not self.remove_stopwords) or (not is_stop)) and (
                (not self.remove_punctuation) or (not is_punct)
            ):
                lemma = self._lemmatize(text)
                tokens.append(
                    Token(
                        text=text,
                        lemma=lemma,
                        pos=self._pos(pos),
                        is_stop=is_stop,
                        stem=self._stem(text),
                    )
                )
        return tokens


@dataclass
class SpacyTokenizer(Tokenizer):
    def __call__(self, text: str) -> list[Token]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> list[Token]:
        if self.lowercase:
            text = text.lower()
        doc = nlp(text)
        tokens = [
            Token(
                text=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                is_stop=token.is_stop,
            )
            for token in doc
            if ((not self.remove_stopwords) or (not token.is_stop))
            and ((not self.remove_punctuation) or (not token.is_punct))
        ]
        return tokens
