from typing import List

import abc

_UNK_TOKEN = '<unk>'
_BOS_TOKEN = '<str>'
_EOS_TOKEN = '<end>'
_PAD_TOKEN = '<pad>'
_DEFAULT_TOKEN2ID = {_PAD_TOKEN: 0, _UNK_TOKEN: 1, _BOS_TOKEN: 2, _EOS_TOKEN: 3}


class AbstractBaseVocabulary(abc.ABC):
    @abc.abstractmethod
    def add_text_to_vocab(self, text):
        raise NotImplementedError

    @abc.abstractmethod
    def convert_text_to_ids(self, text: str) -> List[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def convert_ids_to_text(self, ids: List[int]) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def threshold_rare_words(self, wordcount_threshold=5):
        raise NotImplementedError

    @staticmethod
    def pad_id():
        return _DEFAULT_TOKEN2ID[_PAD_TOKEN]

    @staticmethod
    def eos_id():
        return _DEFAULT_TOKEN2ID[_EOS_TOKEN]

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError
