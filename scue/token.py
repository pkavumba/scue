from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Token:
    text: str
    lemma: str
    pos: str
    is_stop: bool
    stem: str | None = None

    @property
    def is_noun(self) -> bool:
        return self.pos in {"NOUN", "PROPN"}

    @property
    def is_verb(self) -> bool:
        return self.pos == "VERB"

    @property
    def is_adjective(self) -> bool:
        return self.pos == "ADJ"

    @property
    def is_adverb(self) -> bool:
        return self.pos == "ADV"

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"<Token: {self.text}>"

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Token):
            return NotImplemented
        return self.lemma == __value.lemma

    def __hash__(self) -> int:
        return hash((self.lemma,))
