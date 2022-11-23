import json
import numpy as np
import os

from data.utils import _get_img_from_path
from data.abc import AbstractBaseDataset

_DEFAULT_FASHION_IQ_DATASET_ROOT = 'data/fashionIQ'
_DEFAULT_FASHION_IQ_VOCAB_PATH = 'data/fashionIQ/fashion_iq_vocab.pkl'


def _get_img_caption_json(dataset_root, clothing_type, split):
    with open(os.path.join(dataset_root, 'captions', 'cap.{}.{}.json'.format(clothing_type, split))) as json_file:
        img_caption_data = json.load(json_file)
    return img_caption_data

def _get_img_caption_txt(dataset_root, clothing_type, split):
    with open(os.path.join(dataset_root, 'captions_pairs', 'fashion_iq-{}-cap-{}.txt'.format(split, clothing_type))) as f:
        file_content = f.readlines()
    return file_content

def _get_img_split_json_as_list(dataset_root, clothing_type, split):
    with open(os.path.join(dataset_root, 'image_splits', 'split.{}.{}.json'.format(clothing_type, split))) as json_file:
        img_split_list = json.load(json_file)
    return img_split_list

def _create_img_path_from_id(root, id):
    return os.path.join(root, '{}.jpg'.format(id))

def _get_img_path_using_idx(img_caption_data, img_root, idx, is_ref=True):
    img_caption_pair = img_caption_data[idx]
    key = 'candidate' if is_ref else 'target'

    img = _create_img_path_from_id(img_root, img_caption_pair[key])
    id = img_caption_pair[key]
    return img, id

def _get_modifier(img_caption_data, idx, reverse=False):
    img_caption_pair = img_caption_data[idx]
    cap1, cap2 = img_caption_pair['captions']
    return _create_modifier_from_attributes(cap1, cap2) if not reverse else _create_modifier_from_attributes(cap2, cap1)

def _cat_captions(caps):
    I = []
    for i in range(len(caps)):
        if i % 2 == 0:
            I.append(_create_modifier_from_attributes(caps[i], caps[i+1]))
        else:
            I.append(_create_modifier_from_attributes(caps[i], caps[i-1]))
    return I

def _create_modifier_from_attributes(ref_attribute, targ_attribute):
    return ref_attribute + " and " + targ_attribute

def caption_post_process(s):
    return s.strip().replace('.', 'dotmark').replace('?', 'questionmark').replace('&', 'andmark').replace('*', 'starmark')

class AbstractBaseFashionIQDataset(AbstractBaseDataset):

    @classmethod
    def code(cls):
        return 'fashionIQ'

    @classmethod
    def all_codes(cls):
        return ['fashionIQ']
    
    @classmethod
    def all_subset_codes(cls):
        return ['dress', 'shirt', 'toptee']
    
    @classmethod
    def vocab_path(cls):
        return _DEFAULT_FASHION_IQ_VOCAB_PATH


class FashionIQDataset(AbstractBaseFashionIQDataset):
    """
    Fashion200K dataset.
    Image pairs in {root_path}/image_pairs/{split}_pairs.pkl

    """

    def __init__(self, root_path=_DEFAULT_FASHION_IQ_DATASET_ROOT, clothing_type='dress', split='train',
                 img_transform=None, text_transform=None, id_transform=None):
        super().__init__(root_path, split, img_transform, text_transform)
        self.root_path = root_path
        self.img_root_path = os.path.join(self.root_path, 'image_data')
        self.clothing_type = clothing_type
        self.split = split
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.id_transform = id_transform
        self.img_caption_data = _get_img_caption_txt(root_path, clothing_type, split)
        self.ref_img_path = np.array([ff.strip().split(';')[0] for ff in self.img_caption_data])
        self.targ_img_path = np.array([ff.strip().split(';')[1] for ff in self.img_caption_data])
        self.caps = [ff.strip('\n').split(';')[-1] for ff in self.img_caption_data]
        self.caps_cat = _cat_captions(self.caps)


    def __getitem__(self, idx):

        ref_img_path = os.path.join(self.img_root_path, self.ref_img_path[idx])
        targ_img_path = os.path.join(self.img_root_path, self.targ_img_path[idx])
        reference_img = _get_img_from_path(ref_img_path, self.img_transform)
        target_img = _get_img_from_path(targ_img_path, self.img_transform)

        modifier = self.caps_cat[idx]
        modifier = caption_post_process(modifier)
        modifier = self.text_transform(modifier) if self.text_transform else modifier

        ref_id = self.ref_img_path[idx].split('/')[-1].split('.')[0]
        targ_id = self.targ_img_path[idx].split('/')[-1].split('.')[0]

        if self.id_transform:
            ref_id = self.id_transform(ref_id)
            targ_id = self.id_transform(targ_id)

        return reference_img, target_img, modifier, len(modifier), ref_id, targ_id

    def __len__(self):
        return len(self.img_caption_data)# * 2


class FashionIQTestDataset(AbstractBaseFashionIQDataset):
    """
    FashionIQ Test (Samples) dataset.
    indexing returns target samples and their unique ID
    """

    def __init__(self, root_path=_DEFAULT_FASHION_IQ_DATASET_ROOT, clothing_type='dress', split='val',
                 img_transform=None, text_transform=None):
        super().__init__(root_path, split, img_transform, text_transform)
        self.root_path = root_path
        self.img_root_path = os.path.join(self.root_path, 'image_data')
        self.clothing_type = clothing_type
        self.img_transform = img_transform
        self.text_transform = text_transform

        #self.img_list = _get_img_split_json_as_list(root_path, clothing_type, split)

        ''' Uncomment below for VAL Evaluation method '''
        self.img_caption_data = _get_img_caption_txt(root_path, clothing_type, split)
        self.img_list = []
        for d in self.img_caption_data:
            ref = d.split(';')[0].split('/')[-1].split('.')[0] #dress/xxx.jpg --> xxx
            targ = d.split(';')[1].split('/')[-1].split('.')[0] #dress/xxx.jpg --> xxx
            self.img_list.append(ref)
            self.img_list.append(targ)
        self.img_list = list(set(self.img_list))

    def __getitem__(self, idx, use_transform=True):
        img_transform = self.img_transform if use_transform else None
        img_id = self.img_list[idx]
        img_path = _create_img_path_from_id(os.path.join(self.img_root_path,self.clothing_type), img_id)

        target_img = _get_img_from_path(img_path, img_transform)

        return target_img, img_id

    def __len__(self):
        return len(self.img_list)


class FashionIQTestQueryDataset(AbstractBaseFashionIQDataset):
    """
        FashionIQ Test (Query) dataset.
        indexing returns ref samples, modifier, target attribute (caption, text) and modifier length
        """

    def __init__(self, root_path=_DEFAULT_FASHION_IQ_DATASET_ROOT, clothing_type='dress', split='val',
                 img_transform=None, text_transform=None):
        super().__init__(root_path, split, img_transform, text_transform)
        self.root_path = root_path
        self.img_root_path = os.path.join(self.root_path, 'image_data')
        self.clothing_type = clothing_type
        self.img_transform = img_transform
        self.text_transform = text_transform

        self.img_caption_data = _get_img_caption_txt(root_path, clothing_type, split)
        self.ref_img_path = np.array([ff.strip().split(';')[0] for ff in self.img_caption_data])
        self.targ_img_path = np.array([ff.strip().split(';')[1] for ff in self.img_caption_data])
        self.caps = [ff.strip('\n').split(';')[-1] for ff in self.img_caption_data]
        self.caps_cat = _cat_captions(self.caps)

    def __getitem__(self, idx, use_transform=True):
        img_transform = self.img_transform if use_transform else None
        text_transform = self.text_transform if use_transform else None

        ref_img_path = os.path.join(self.img_root_path, self.ref_img_path[idx])
        ref_img = _get_img_from_path(ref_img_path, self.img_transform)
        ref_id = self.ref_img_path[idx].split('/')[-1].split('.')[0]
        targ_id = self.targ_img_path[idx].split('/')[-1].split('.')[0]

        modifier = self.caps_cat[idx]
        modifier = caption_post_process(modifier)
        modifier = self.text_transform(modifier) if self.text_transform else modifier

        return ref_img, ref_id, modifier, targ_id, len(modifier)

    def __len__(self):
        return len(self.img_caption_data)
