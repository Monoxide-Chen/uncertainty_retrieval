from collections import defaultdict
from typing import List

import abc
from tqdm import tqdm

from language.abc import _UNK_TOKEN, _BOS_TOKEN, _EOS_TOKEN, _PAD_TOKEN, _DEFAULT_TOKEN2ID, AbstractBaseVocabulary


class AbstractBaseTokenizer(abc.ABC):
    def tokenize(self, text: str) -> List[str]:
        return [_BOS_TOKEN] + self._tokenize(text) + [_EOS_TOKEN]

    def detokenize(self, tokens: List[str]) -> str:
        start_idx = tokens.index(_BOS_TOKEN)
        end_idx = tokens.index(_EOS_TOKEN)
        tokens = tokens[start_idx + 1: end_idx]
        tokens = list(filter(_PAD_TOKEN.__ne__, tokens))
        return self._detokenize(tokens)

    @abc.abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def _detokenize(self, tokens: List[str]) -> str:
        raise NotImplementedError


# TODO: We can add read / write to persistent memory for optimizing this process
class SimpleVocabulary(AbstractBaseVocabulary):
    def __init__(self, tokenizer: AbstractBaseTokenizer):
        self.tokenizer = tokenizer
        self._token2id = _DEFAULT_TOKEN2ID
        self._id2token = {i: token for token, i in _DEFAULT_TOKEN2ID.items()}
        self._token_count = defaultdict(int)
        self._token_count[_UNK_TOKEN] = int(9e9)
        self._token_count[_PAD_TOKEN] = int(9e9)
        self._token_count[_BOS_TOKEN] = int(9e9)
        self._token_count[_EOS_TOKEN] = int(9e9)

    def add_text_to_vocab(self, text):
        tokens = self.tokenizer.tokenize(text)
        for token in tokens:
            if token not in self._token2id:
                idx = len(self._token2id)
                self._token2id[token] = idx
                self._id2token[idx] = token
            self._token_count[token] += 1

    def threshold_rare_words(self, wordcount_threshold=5):
        for w in self._token2id:
            if self._token_count[w] < wordcount_threshold:
                self._token2id[w] = _DEFAULT_TOKEN2ID[_UNK_TOKEN]

    def convert_text_to_ids(self, text):
        tokens = self.tokenizer.tokenize(text)
        encoded_text = [self._token2id.get(t, _DEFAULT_TOKEN2ID[_UNK_TOKEN]) for t in tokens]
        return encoded_text

    def convert_ids_to_text(self, ids):
        tokens = [self._id2token.get(token_id, _UNK_TOKEN) for token_id in ids]
        return self.tokenizer.detokenize(tokens)

    def __len__(self):
        return len(self._token2id)

    @staticmethod
    def create_and_store_vocabulary_from_txt_files(txt_file_paths, tokenizer, write_func, txt_reader_func):
        vocab = SimpleVocabulary(tokenizer)
        for txt_path in txt_file_paths:
            texts = txt_reader_func(txt_path)
            for t in tqdm(texts):
                vocab.add_text_to_vocab(t)
        write_func(vocab)
        return vocab

    @staticmethod
    def create_and_store_vocabulary_from_list(list_data, tokenizer, write_func):
        vocab = SimpleVocabulary(tokenizer)
        for l in tqdm(list_data):
            vocab.add_text_to_vocab(l)
        write_func(vocab)
        return vocab

    @staticmethod
    def create_and_store_vocabulary_from_datasets(datasets, tokenizer, write_func, caption_pos=(2, 1)):
        vocab = SimpleVocabulary(tokenizer)
        for pos, dataset in zip(caption_pos, datasets):
            for record in tqdm(dataset):
                vocab.add_text_to_vocab(record[pos])
        write_func(vocab)
        return vocab

    @staticmethod
    def create_vocabulary_from_storage(read_func):
        return read_func()
