import os
import glob
import random
import json
import numpy as np
from data.utils import _get_img_from_path
from data.abc import AbstractBaseDataset

_DEFAULT_FASHION_IQ_DATASET_ROOT = 'data/fashion200k'
_DEFAULT_FASHION_IQ_VOCAB_PATH = 'data/fashion200k/fashion200k_vocab.pkl'

def get_different_word(source_caption, target_caption):
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = 'replace ' + source_word + ' with ' + target_word
        return source_word, target_word, mod_str

def caption_post_process(s):
    return s.strip().replace('.', 'dotmark').replace('?', 'questionmark').replace('&', 'andmark').replace('*', 'starmark')

class AbstractBaseFashion200kDataset(AbstractBaseDataset):

    @classmethod
    def code(cls):
        return 'fashion200k'

    @classmethod
    def all_codes(cls):
        return ['fashion200k']

    @classmethod
    def vocab_path(cls):
        return _DEFAULT_FASHION_IQ_VOCAB_PATH

class Fashion200kDataset(AbstractBaseFashion200kDataset):

    def __init__(self, root_path=_DEFAULT_FASHION_IQ_DATASET_ROOT, clothing_type=None,  split='train',
                 img_transform=None, text_transform=None, id_transform=None):
        super().__init__(root_path, split, img_transform, text_transform)
        self.root_path = root_path
        self.clothing_type = clothing_type
        self.split = split
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.id_transform = id_transform

        self.imgs = []
        self.filenames = []
        self.texts = []

        label_path = os.path.join(self.root_path, 'labels')
        print("Processing {} set".format(split))
        label_files = glob.glob(os.path.join(label_path, "*_" + split + "_*.txt"))
        label_files.sort()

        self.readfiles(label_files)

        self.caption_index_init_()
        self.generate_random_train_queries_(n_modifications_per_image=5)

    def __getitem__(self, idx, use_transform=True):
        img_transform = self.img_transform if use_transform else None

        ref_img_path = self.source_files[idx]
        targ_img_path = self.target_files[idx]
        reference_img = _get_img_from_path(ref_img_path, self.img_transform)
        target_img = _get_img_from_path(targ_img_path, self.img_transform)

        modifier = self.modify_texts[idx]
        modifier = self.text_transform(modifier) if self.text_transform else modifier

        ref_id = self.source_texts[idx]
        targ_id = self.target_texts[idx]

        if self.id_transform:
            ref_id = self.id_transform(ref_id)
            targ_id = self.id_transform(targ_id)

        return reference_img, target_img, modifier, len(modifier), ref_id, targ_id

    def __len__(self):
        return len(self.modify_texts)

    def readfiles(self, label_files):
        for label_file in label_files:
            print('read ' + label_file)
            with open(label_file, "r", encoding="utf8") as fd:
                for line in fd.readlines():
                    line = line.split("\t")
                    img = {
                            'file_path': line[0],
                            'captions': [caption_post_process(line[2])],
                            'modifiable': False
                    }
                    self.filenames += [os.path.join(self.root_path, img['file_path'])]
                    self.texts += img['captions']
                    self.imgs += [img]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly"""
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.imgs):
            for c in img['captions']:
                if not c in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        print('unique cations = %d' % len(caption2imgids))

        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('    ', ' ').strip()
                if not p in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        for img in self.imgs:
            img['modifiable'] = False
            img['parent_captions'] = []
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:
                for c in parent2children_captions[p]:
                    for imgid in caption2imgids[c]:
                        self.imgs[imgid]['modifiable'] = True
                        self.imgs[imgid]['parent_captions'] += [p]

        num_modifiable_imgs = 0
        for img in self.imgs:
            if img['modifiable']:
                num_modifiable_imgs += 1
        self.num_modifiable_imgs = num_modifiable_imgs
        print('Modifiable images = %d' % num_modifiable_imgs)


    def caption_index_sample_(self, idx):
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['captions'][0]
        source_word, target_word, mod_str = get_different_word(
                source_caption, target_caption)
        return idx, target_idx, source_caption, target_caption, mod_str

    def generate_random_train_queries_(self, n_modifications_per_image=3):
        self.source_files =[]
        self.target_files = []
        self.modify_texts = []
        self.source_texts = []
        self.target_texts = []
        already_visited = set()

        for i, img in enumerate(self.imgs):
            if img['modifiable']:
                for j in range(n_modifications_per_image):
                    idx, target_idx, source_caption, target_caption, mod_str = self.caption_index_sample_(i)
                    # ensure the choosen pairs does not share the same words even the ordering is different
                    set1 = set(self.imgs[idx]['captions'][0].split(' '))
                    set2 = set(self.imgs[target_idx]['captions'][0].split(' '))
                    if set1 != set2:
                        key = "{}-{}".format(target_idx, idx)
                        inv_key = "{}-{}".format(idx, target_idx)
                        if not (key in already_visited or inv_key in already_visited):
                            self.source_files += [os.path.join(self.root_path, self.imgs[idx]['file_path'])]
                            self.target_files += [os.path.join(self.root_path, self.imgs[target_idx]['file_path'])]
                            self.modify_texts += [mod_str]
                            self.source_texts += self.imgs[idx]['captions']
                            self.target_texts += self.imgs[target_idx]['captions']
                            already_visited.add(key)

        # randomly shuffle the epoch wise sampled pairs
        shuffle_idx = list(range(len(self.source_files)))
        random.shuffle(shuffle_idx)
        self.source_files = [self.source_files[i] for i in shuffle_idx]
        self.target_files = [self.target_files[i] for i in shuffle_idx]
        self.modify_texts = [self.modify_texts[i] for i in shuffle_idx]
        self.source_texts = [self.source_texts[i] for i in shuffle_idx]
        self.target_texts = [self.target_texts[i] for i in shuffle_idx]
        print('shuffling the random source-target pairs. It gives %d pairs.' % len(self.source_files))

class Fashion200kTestDataset(AbstractBaseFashion200kDataset):

    def __init__(self, root_path=_DEFAULT_FASHION_IQ_DATASET_ROOT, clothing_type=None,  split='test',
                 img_transform=None, text_transform=None):
        super().__init__(root_path, split, img_transform, text_transform)
        if split == 'val':
            split = 'test'
        self.root_path = root_path
        self.clothing_type = clothing_type
        self.split = split
        self.img_transform = img_transform
        self.text_transform = text_transform

        self.imgs = []
        self.filenames = []
        self.texts = []

        label_path = os.path.join(self.root_path, 'labels')
        print("Processing {} set".format(split))
        label_files = glob.glob(os.path.join(label_path, "*_" + split + "_*.txt"))
        label_files.sort()

        self.readfiles(label_files)
        '''little gallery'''
        #self.little_gallery = self.generate_gallery()

    def generate_gallery(self):
        file2imgid = {}
        gallery = []
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
        with open(os.path.join(self.root_path, 'test_queries.txt')) as f:
            lines = f.readlines()
        for line in lines:
            ref_path = line.split()[0]
            ref_idx = file2imgid[ref_path]
            ref_cap = self.imgs[ref_idx]['captions'][0]
            targ_path = line.split()[1]
            targ_idx = file2imgid[targ_path]
            targ_cap = self.imgs[targ_idx]['captions'][0]
            gallery.append(ref_path+';'+ref_cap)
            gallery.append(targ_path+';'+targ_cap)
        return list(set(gallery))


    def __getitem__(self, idx, use_transform=True):
        img_transform = self.img_transform if use_transform else None

        img = self.imgs[idx]
        target_img = _get_img_from_path(os.path.join(self.root_path, img['file_path']), self.img_transform)
        img_id = caption_post_process(img['captions'][0])
        #img = self.little_gallery[idx]
        #target_img = _get_img_from_path(os.path.join(self.root_path, img.split(';')[0]), self.img_transform)
        #img_id = caption_post_process(img.split(';')[1])

        return target_img, img_id


    def __len__(self):
        return len(self.imgs)
        #return len(self.little_gallery)

    def readfiles(self, label_files):
        for label_file in label_files:
            print('read ' + label_file)
            with open(label_file, "r", encoding="utf8") as fd:
                for line in fd.readlines():
                    line = line.split("\t")
                    img = {
                            'file_path': line[0],
                            'captions': [caption_post_process(line[2])],
                            'modifiable': False
                    }
                    self.filenames += [os.path.join(self.root_path, img['file_path'])]
                    self.texts += img['captions']
                    self.imgs += [img]

class Fashion200kTestQueryDataset(AbstractBaseFashion200kDataset):

    def __init__(self, root_path=_DEFAULT_FASHION_IQ_DATASET_ROOT, clothing_type=None,  split='test',
                 img_transform=None, text_transform=None):
        if split == 'val':
            split = 'test'
        super().__init__(root_path, split, img_transform, text_transform)
        self.root_path = root_path
        self.clothing_type = clothing_type
        self.split = split
        self.img_transform = img_transform
        self.text_transform = text_transform

        self.imgs = []
        self.filenames = []
        self.texts = []

        label_path = os.path.join(self.root_path, 'labels')
        print("Processing {} set".format(split))
        label_files = glob.glob(os.path.join(label_path, "*_" + split + "_*.txt"))
        label_files.sort()

        self.readfiles(label_files)
        if self.split == 'test':
            self.generate_test_queries()

    def __getitem__(self, idx, use_transform=True):
        img_transform = self.img_transform if use_transform else None
        text_transform = self.text_transform if use_transform else None

        ref_img_path = self.source_files[idx]
        ref_img = _get_img_from_path(ref_img_path, self.img_transform)
        ref_id = caption_post_process(self.ref_ids[idx])
        targ_id = caption_post_process(self.targ_ids[idx])
        modifier = self.modify_texts[idx]

        modifier = self.text_transform(modifier) if self.text_transform else modifier
        return ref_img, ref_id, modifier, targ_id, len(modifier)

    def __len__(self):
        return len(self.modify_texts)

    def readfiles(self, label_files):
        for label_file in label_files:
            print('test query read ' + label_file)
            with open(label_file, "r", encoding="utf8") as fd:
                for line in fd.readlines():
                    line = line.split("\t")
                    img = {
                            'file_path': line[0],
                            'captions': [caption_post_process(line[2])],
                            'modifiable': False
                    }
                    self.filenames += [os.path.join(self.root_path, img['file_path'])]
                    self.texts += img['captions']
                    self.imgs += [img]

    def generate_test_queries(self):
        file2imgid = {}
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
        with open(os.path.join(self.root_path, 'test_queries.txt')) as f:
            lines = f.readlines()
        
        self.test_queries = []
        self.modify_texts = []
        self.source_files = []
        self.ref_ids = []
        self.targ_ids = []
        for line in lines:
            source_file, target_file = line.split()
            idx = file2imgid[source_file]
            target_idx = file2imgid[target_file]
            source_caption = self.imgs[idx]['captions'][0]
            target_caption = self.imgs[target_idx]['captions'][0]
            source_word, target_word, mod_str = get_different_word(
                    source_caption, target_caption)
            self.test_queries += [{
                    'source_img_id': idx,
                    'source_caption': source_caption,
                    'target_caption': target_caption,
                    'mod': {
                            'str': mod_str
                    }
            }]
            self.source_files += [os.path.join(self.root_path, source_file)]
            self.ref_ids += [source_caption]
            self.targ_ids += [target_caption]
            self.modify_texts += [mod_str]
