import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data.fashionIQ import FashionIQDataset, FashionIQTestDataset, FashionIQTestQueryDataset
from data.fashion200k import Fashion200kDataset, Fashion200kTestDataset, Fashion200kTestQueryDataset
from data.shoes import ShoesDataset, ShoesTestDataset, ShoesTestQueryDataset
from data.collate_fns import PaddingCollateFunction, PaddingCollateFunctionTest, BertPaddingCollateFunction, BertPaddingCollateFunctionTest
from language import AbstractBaseVocabulary

DEFAULT_VOCAB_PATHS = {
    **dict.fromkeys(FashionIQDataset.all_codes(), FashionIQDataset.vocab_path()),
    **dict.fromkeys(ShoesDataset.all_codes(), ShoesDataset.vocab_path()),
    **dict.fromkeys(Fashion200kDataset.all_codes(), Fashion200kDataset.vocab_path())
}

def train_dataset_factory(transforms, config):
    image_transform = transforms['image_transform']
    text_transform = None if config['text_encoder'] == 'roberta' else transforms['text_transform']
    dataset_code = config['dataset']
    use_subset = config.get('use_subset', False)

    if FashionIQDataset.code() in dataset_code:
        # concat subsets of FashionIQ      
        fashionIQ_datasets  = [
            FashionIQDataset(split='train', clothing_type=clothing_type, img_transform=image_transform,
                                text_transform=text_transform)
            for clothing_type in FashionIQDataset.all_subset_codes()
            ]
        dataset = torch.utils.data.ConcatDataset(fashionIQ_datasets)
    elif ShoesDataset.code() in dataset_code:
        dataset = ShoesDataset(split='train', clothing_type=None, img_transform=image_transform,
                               text_transform=text_transform)
    elif Fashion200kDataset.code() in dataset_code:
        dataset = Fashion200kDataset(split='train', clothing_type=None, img_transform=image_transform,
                                     text_transform=text_transform)
    else:
        raise ValueError("There's no {} dataset".format(dataset_code))

    return dataset


def test_dataset_factory(transforms, config, split='val'):
    image_transform = transforms['image_transform']
    text_transform = None if config['text_encoder'] == 'roberta' else transforms['text_transform']
    dataset_code = config['dataset']
    test_datasets = {}

    if FashionIQDataset.code() in dataset_code:
        for clothing_type in FashionIQDataset.all_subset_codes():
            test_datasets['fashionIQ_' + clothing_type] = {
                "samples": FashionIQTestDataset(split=split, clothing_type=clothing_type,
                                                img_transform=image_transform, text_transform=text_transform),
                "query": FashionIQTestQueryDataset(split=split, clothing_type=clothing_type,
                                                img_transform=image_transform, text_transform=text_transform)
            }

    elif ShoesDataset.code() in dataset_code:
        test_datasets[ShoesDataset.code()] = {
            "samples": ShoesTestDataset(split=split, clothing_type=None,
                                                        img_transform=image_transform, text_transform=text_transform),
            "query": ShoesTestQueryDataset(split=split, clothing_type=None,
                                                        img_transform=image_transform, text_transform=text_transform)
        }

    elif Fashion200kDataset.code() in dataset_code:
        test_datasets[Fashion200kDataset.code()] = {
            "samples": Fashion200kTestDataset(split=split, clothing_type=None,
                                                        img_transform=image_transform, text_transform=text_transform),
            "query": Fashion200kTestQueryDataset(split=split, clothing_type=None,
                                                        img_transform=image_transform, text_transform=text_transform)
        }
    if len(test_datasets) == 0:
        raise ValueError("There's no {} dataset".format(dataset_code))

    return test_datasets


def train_dataloader_factory(dataset, config, collate_fn=None):
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 16)
    shuffle = config.get('shuffle', True)
    # TODO: remove this
    drop_last = batch_size == 32

    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_fn, drop_last=drop_last)


def test_dataloader_factory(datasets, config, collate_fn=None):
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 16)
    shuffle = False

    return {
        'query': DataLoader(datasets['query'], batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                            collate_fn=collate_fn),
        'samples': DataLoader(datasets['samples'], batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=True,
                              collate_fn=collate_fn)
    }

def create_dataloaders(image_transform, text_transform, configs):
    train_dataset = train_dataset_factory(
        transforms={'image_transform': image_transform['train'], 'text_transform': text_transform['train']},
        config=configs)
    test_datasets = test_dataset_factory(
        transforms={'image_transform': image_transform['val'], 'text_transform': text_transform['val']},
        config=configs)
    train_val_datasets = test_dataset_factory(
        transforms={'image_transform': image_transform['val'], 'text_transform': text_transform['val']},
        config=configs, split='train')
    
    if configs['text_encoder'] == 'roberta':
        padding_idx = 1
        collate_fn = BertPaddingCollateFunction(padding_idx=padding_idx)
        collate_fn_test = BertPaddingCollateFunctionTest(padding_idx=padding_idx)
    else:
        padding_idx = AbstractBaseVocabulary.pad_id()
        collate_fn = PaddingCollateFunction(padding_idx=padding_idx)
        collate_fn_test = PaddingCollateFunctionTest(padding_idx=padding_idx)
        
    train_dataloader = train_dataloader_factory(dataset=train_dataset, config=configs, collate_fn=collate_fn)
    test_dataloaders = {key: test_dataloader_factory(datasets=value, config=configs, collate_fn=collate_fn_test) for key, value in test_datasets.items()}
    train_val_dataloaders = {key: test_dataloader_factory(datasets=value, config=configs, collate_fn=collate_fn_test) for key, value in train_val_datasets.items()}
    return train_dataloader, test_dataloaders, train_val_dataloaders
