from typing import List

import torch
from torchvision import transforms

from language import AbstractBaseVocabulary


class ToIds(object):
    def __init__(self, vocabulary: AbstractBaseVocabulary):
        self.vocabulary = vocabulary

    def __call__(self, text: str) -> List[int]:
        return self.vocabulary.convert_text_to_ids(text)


class ToLongTensor(object):
    def __call__(self, ids: List[int]) -> torch.LongTensor:
        return torch.LongTensor(ids)


def text_transform_factory(config: dict):
    vocabulary = config['vocabulary']

    return {
        'train': transforms.Compose([ToIds(vocabulary), ToLongTensor()]),
        'val': transforms.Compose([ToIds(vocabulary), ToLongTensor()])
    }
