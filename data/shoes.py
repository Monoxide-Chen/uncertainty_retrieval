import json
import numpy as np
import os

from data.utils import _get_img_from_path
from data.abc import AbstractBaseDataset

_DEFAULT_FASHION_IQ_DATASET_ROOT = 'data/shoes'
_DEFAULT_FASHION_IQ_VOCAB_PATH = 'data/shoes/shoes_vocab.pkl'

def _get_img_caption_txt(dataset_root, split):
    split = 'test' if split == 'val' else 'train'
    with open(os.path.join(dataset_root, 'shoes-cap-{}.txt'.format(split))) as f:
        file_content = f.readlines()
    return file_content

def _create_img_path_from_id(root, id):
    return os.path.join(root, '{}.jpg'.format(id))

def _cat_captions(caps):
    I = []
    for i in range(len(caps)):
        I.append(caps[i])
    return I

def caption_post_process(s):
    return s.strip().replace('.', 'dotmark').replace('?', 'questionmark').replace('&', 'andmark').replace('*', 'starmark')

class AbstractBaseShoesDataset(AbstractBaseDataset):

    @classmethod
    def code(cls):
        return 'shoes'

    @classmethod
    def all_codes(cls):
        return ['shoes']

    @classmethod
    def vocab_path(cls):
        return _DEFAULT_FASHION_IQ_VOCAB_PATH


class ShoesDataset(AbstractBaseShoesDataset):
    """
    shoes dataset
    root_path = datasets/shoes/attributedata
    clothing=None (Not used in shoes dataset)
    """

    def __init__(self, root_path=_DEFAULT_FASHION_IQ_DATASET_ROOT, clothing_type=None, split='train',
                 img_transform=None, text_transform=None, id_transform=None):
        super().__init__(root_path, split, img_transform, text_transform)
        self.root_path = root_path
        self.img_root_path = os.path.join(self.root_path, 'attributedata')
        self.clothing_type = None
        self.split = split
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.id_transform = id_transform
        self.img_caption_data = _get_img_caption_txt(root_path, split)
        self.ref_img_path = np.array([ff.strip().split(';')[0] for ff in self.img_caption_data])
        self.targ_img_path = np.array([ff.strip().split(';')[1] for ff in self.img_caption_data])
        self.caps = [ff.strip().split(';')[-1] for ff in self.img_caption_data]
        self.caps_cat = _cat_captions(self.caps) #actually no catenate


    def __getitem__(self, idx):

        ref_img_path = self.ref_img_path[idx]
        targ_img_path = self.targ_img_path[idx]
        reference_img = _get_img_from_path(ref_img_path, self.img_transform)
        target_img = _get_img_from_path(targ_img_path, self.img_transform)

        modifier = self.caps_cat[idx]
        modifier = self.text_transform(modifier) if self.text_transform else modifier

        ref_id = self.ref_img_path[idx].split('/')[-1].split('.')[0]
        targ_id = self.targ_img_path[idx].split('/')[-1].split('.')[0]

        if self.id_transform:
            ref_id = self.id_transform(ref_id)
            targ_id = self.id_transform(targ_id)

        return reference_img, target_img, modifier, len(modifier), ref_id, targ_id

    def __len__(self):
        return len(self.img_caption_data)

class ShoesTestDataset(AbstractBaseShoesDataset):
    """
    Shoes Test (Samples) dataset.
    indexing returns target samples and their unique ID
    """

    def __init__(self, root_path=_DEFAULT_FASHION_IQ_DATASET_ROOT, clothing_type=None, split='val',
                 img_transform=None, text_transform=None):
        super().__init__(root_path, split, img_transform, text_transform)
        self.root_path = root_path
        self.img_root_path = os.path.join(self.root_path, 'image_data')
        self.clothing_type = None
        self.img_transform = img_transform
        self.text_transform = text_transform

        #self.img_list = _get_img_split_json_as_list(root_path, clothing_type, split)

        ''' Uncomment below for VAL Evaluation method '''
        self.img_caption_data = _get_img_caption_txt(root_path, split)
        self.img_list = []
        self.img_paths_list = []
        filename = 'data/shoes/shoes-test-all.txt'
        text_file = open(filename, 'r')
        self.img_list = text_file.readlines()
        self.img_paths_list = self.img_list[:] #new list of all paths
        self.img_paths_list = [imgpath.strip() for imgpath in self.img_paths_list]
        self.img_list = [imgname.strip().split('/')[-1].split('.')[0] for imgname in self.img_list]

        for d in self.img_caption_data:
            self.img_paths_list.extend(d.strip().split(';')[:2])
            ref = d.strip().split(';')[0].split('/')[-1].split('.')[0] #dress/xxx.jpg --> xxx
            targ = d.strip().split(';')[1].split('/')[-1].split('.')[0] #dress/xxx.jpg --> xxx
            self.img_list.append(ref)
            self.img_list.append(targ)
        img_index = self.img_list.index
        path_index = self.img_paths_list.index
        self.img_list = list(set(self.img_list))
        self.img_list.sort(key=img_index)
        self.img_paths_list = list(set(self.img_paths_list))
        self.img_paths_list.sort(key=path_index) # one to one correspond

    def __getitem__(self, idx, use_transform=True):
        img_transform = self.img_transform if use_transform else None
        img_id = self.img_list[idx]
        #img_path = _create_img_path_from_id(os.path.join(self.img_root_path,self.clothing_type), img_id)
        img_path = self.img_paths_list[idx]

        assert img_id in img_path

        target_img = _get_img_from_path(img_path, img_transform)

        return target_img, img_id

    def __len__(self):
        return len(self.img_list)


class ShoesTestQueryDataset(AbstractBaseShoesDataset):
    """
        Shoes Test (Query) dataset.
        indexing returns ref samples, modifier, target attribute (caption, text) and modifier length
        """

    def __init__(self, root_path=_DEFAULT_FASHION_IQ_DATASET_ROOT, clothing_type='dress', split='val',
                 img_transform=None, text_transform=None):
        super().__init__(root_path, split, img_transform, text_transform)
        self.root_path = root_path
        self.img_root_path = os.path.join(self.root_path, 'image_data')
        self.clothing_type = None
        self.img_transform = img_transform
        self.text_transform = text_transform

        self.img_caption_data = _get_img_caption_txt(root_path, split)
        self.ref_img_path = np.array([ff.strip().split(';')[0] for ff in self.img_caption_data])
        self.targ_img_path = np.array([ff.strip().split(';')[1] for ff in self.img_caption_data])
        self.caps = [ff.strip('\n').split(';')[-1] for ff in self.img_caption_data]
        self.caps_cat = _cat_captions(self.caps)

    def __getitem__(self, idx, use_transform=True):

        img_transform = self.img_transform if use_transform else None
        text_transform = self.text_transform if use_transform else None

        ref_img_path = self.ref_img_path[idx]
        ref_img = _get_img_from_path(ref_img_path, self.img_transform)
        ref_id = self.ref_img_path[idx].split('/')[-1].split('.')[0]
        targ_id = self.targ_img_path[idx].split('/')[-1].split('.')[0]

        modifier = self.caps_cat[idx]

        modifier = self.text_transform(modifier) if self.text_transform else modifier

        return ref_img, ref_id, modifier, targ_id, len(modifier)

    def __len__(self):
        return len(self.img_caption_data)
