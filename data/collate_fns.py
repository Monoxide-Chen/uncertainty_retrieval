from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

class PaddingCollateFunction(object):
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch: List[tuple]):
        reference_images, target_images, modifiers, lengths, ref_id, targ_id = zip(*batch)

        reference_images = torch.stack(reference_images, dim=0)
        target_images = torch.stack(target_images, dim=0)
        seq_lengths = torch.tensor(lengths).long()
        modifiers = pad_sequence(modifiers, padding_value=self.padding_idx, batch_first=True)
        # ref_id = torch.tensor(ref_id).long()
        # targ_id = torch.tensor(targ_id).long()
        return reference_images, target_images, modifiers, seq_lengths, None

class PaddingCollateFunctionTest(object):
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    @staticmethod
    def _collate_test_dataset(batch):
        reference_images, ids = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        return reference_images, ids

    def _collate_test_query_dataset(self, batch):
        reference_images, ref_attrs, modifiers, target_attrs, lengths = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        seq_lengths = torch.tensor(lengths).long()
        modifiers = pad_sequence(modifiers, padding_value=self.padding_idx, batch_first=True)
        return reference_images, ref_attrs, modifiers, target_attrs, seq_lengths, None

    def __call__(self, batch: List[tuple]):
        num_items = len(batch[0])
        if num_items > 2:
            return self._collate_test_query_dataset(batch)
        else:
            return self._collate_test_dataset(batch)

class BertPaddingCollateFunction(object):
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def __call__(self, batch: List[tuple]):
        reference_images, target_images, modifiers, lengths, ref_id, targ_id = zip(*batch)

        reference_images = torch.stack(reference_images, dim=0)
        target_images = torch.stack(target_images, dim=0)
        seq_lengths = torch.tensor(lengths).long()

        modifiers = list(modifiers)
        token = self.tokenizer.batch_encode_plus(modifiers, padding='longest', return_tensors='pt')

        attn_mask = token['attention_mask']
        modifiers = token['input_ids']

        # ref_id = torch.tensor(ref_id).long()
        # targ_id = torch.tensor(targ_id).long()
        return reference_images, target_images, modifiers, seq_lengths, attn_mask


class BertPaddingCollateFunctionTest(object):
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    @staticmethod
    def _collate_test_dataset(batch):
        reference_images, ids = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        return reference_images, ids

    def _collate_test_query_dataset(self, batch):
        reference_images, ref_attrs, modifiers, target_attrs, lengths = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        seq_lengths = torch.tensor(lengths).long()

        modifiers = list(modifiers)
        token = self.tokenizer.batch_encode_plus(modifiers, padding='longest', return_tensors='pt')

        attn_mask = token['attention_mask']
        modifiers = token['input_ids']

        return reference_images, ref_attrs, modifiers, target_attrs, seq_lengths, attn_mask

    def __call__(self, batch: List[tuple]):
        num_items = len(batch[0])
        if num_items > 2:
            return self._collate_test_query_dataset(batch)
        else:
            return self._collate_test_dataset(batch)
