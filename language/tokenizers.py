import string
from typing import List

import nltk

from language.vocabulary import AbstractBaseTokenizer

_punctuation_translator = str.maketrans('', '', string.punctuation)


class BasicTokenizer(AbstractBaseTokenizer):
    def _tokenize(self, text):
        tokens = str(text).lower().translate(_punctuation_translator).strip().split()
        return tokens

    def _detokenize(self, tokens: List[str]) -> str:
        return ' '.join(tokens)


class NltkTokenizer(AbstractBaseTokenizer):
    def _tokenize(self, text: str) -> List[str]:
        return nltk.tokenize.word_tokenize(text.lower())

    def _detokenize(self, tokens: List[str]) -> str:
        return ' '.join(tokens)
