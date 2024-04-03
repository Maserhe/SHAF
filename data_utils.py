import json
from pathlib import Path
from typing import List
from os import listdir
from os.path import isfile
from os.path import join
import random
import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np


base_path = Path(__file__).absolute().parents[1].absolute()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class ShoesDataset(Dataset):
    """
    Shoes dataset class which manage Shoes data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.split = split
        self.preprocess = preprocess

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")

        self.image_path_list: list = []
        self.lines_train: list = []  # 用于保存每一行字符串的数组
        self.lines_test: list = []  # 用于保存每一行字符串的数组

        with open(base_path / 'data' / 'shoes_dataset' / f"shoes-cap-test.txt") as f:
            for line in f:
                temp_list = line.strip().split(';')
                self.lines_test.append(temp_list)  # strip()用于移除行末尾的换行符和空白字符
                self.image_path_list.append(temp_list[0])
                self.image_path_list.append(temp_list[1])

        with open(base_path / 'data' / 'shoes_dataset' / f"shoes-cap-train.txt") as f:
            for line in f:
                temp_list = line.strip().split(';')
                self.lines_train.append(temp_list)  # strip()用于移除行末尾的换行符和空白字符
                # self.image_path_list.append(temp_list[0])
                # self.image_path_list.append(temp_list[1])

        self.image_path_list = list(set(self.image_path_list))

        print(f"Shoes {split} dataset in {mode} mode initialized")

        # print(len(self.image_path_list), len(self.lines_train), len(self.lines_test))
        # print(base_path)

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                if self.split == 'train':

                    reference_name = self.lines_train[index][0]
                    target_name = self.lines_train[index][1]
                    caption = self.lines_train[index][2]

                    reference_image_path = base_path / reference_name
                    reference_image = self.preprocess(images=PIL.Image.open(reference_image_path), return_tensors='pt', padding=True)
                    reference_image['pixel_values'] = reference_image['pixel_values'].squeeze(0)

                    target_image_path = base_path / target_name
                    target_image = self.preprocess(images=PIL.Image.open(target_image_path), return_tensors='pt', padding=True)
                    target_image['pixel_values'] = target_image['pixel_values'].squeeze(0)
                    return reference_image, target_image, caption

                elif self.split == 'val':

                    reference_name = self.lines_test[index][0]
                    target_name = self.lines_test[index][1]
                    caption = self.lines_test[index][2]

                    return reference_name, target_name, caption

            elif self.mode == 'classic':

                image_name = self.image_path_list[index]
                image_path = base_path / image_name

                image = self.preprocess(images=PIL.Image.open(image_path), return_tensors='pt', padding=True)
                image['pixel_values'] = image['pixel_values'].squeeze(0)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            if self.split == 'train':
                return len(self.lines_train)
            return len(self.lines_test)
        elif self.mode == 'classic':
            return len(self.image_path_list)

        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(
                    base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split == 'train':
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.jpg"
                    # reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    reference_image = self.preprocess(images=PIL.Image.open(reference_image_path), return_tensors='pt',
                                                      padding=True)
                    reference_image['pixel_values'] = reference_image['pixel_values'].squeeze(0)

                    target_name = self.triplets[index]['target']
                    target_image_path = base_path  / 'fashionIQ_dataset' / 'images' / f"{target_name}.jpg"
                    # target_image = self.preprocess(PIL.Image.open(target_image_path))

                    target_image = self.preprocess(images=PIL.Image.open(target_image_path), return_tensors='pt',
                                                   padding=True)
                    target_image['pixel_values'] = target_image['pixel_values'].squeeze(0)
                    return reference_image, target_image, image_captions

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions

                elif self.split == 'test':
                    reference_image_path = base_path  / 'fashionIQ_dataset' / 'images' / f"{reference_name}.jpg"
                    # reference_image = self.preprocess(PIL.Image.open(reference_image_path))

                    reference_image = self.preprocess(images=PIL.Image.open(reference_image_path), return_tensors='pt',
                                                      padding=True)
                    reference_image['pixel_values'] = reference_image['pixel_values'].squeeze(0)
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.jpg"
                # image = self.preprocess(PIL.Image.open(image_path))

                image = self.preprocess(images=PIL.Image.open(image_path), return_tensors='pt', padding=True)
                image['pixel_values'] = image['pixel_values'].squeeze(0)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class Fashion200kDataset(Dataset):
    """Fashion200k dataset."""

    def __init__(self, split: str, mode: str, preprocess: callable):
        super(Fashion200kDataset, self).__init__()

        self.split = split

        if split == "val":
            split = "test"
            self.split = "test"


        self.preprocess = preprocess

        self.img_path = base_path / 'data' / 'fashion200k_dataset'
        self.mode = mode

        # get label files for the split
        label_path = base_path / 'data' / 'fashion200k_dataset' / 'labels'

        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]

        # print(label_files)
        label_files = [f for f in label_files if split in f]

        # print(label_files)

        # read image info from label files
        self.imgs = []

        def caption_post_process(s):
            return s.strip().replace('.',
                                     'dotmark').replace('?', 'questionmark').replace(
                '&', 'andmark').replace('*', 'starmark')

        for filename in label_files:
            print('read ' + filename)
            with open(label_path / f"{filename}", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('	')
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    'captions': [caption_post_process(line[2])],
                    'split': split,
                    'modifiable': False
                }
                self.imgs += [img]
        print('Fashion200k:', len(self.imgs), 'images')

        self.test_set_images = []

        # generate query for training or testing
        if split == 'train':
            self.caption_index_init_()
        else:
            self.generate_test_queries_()
            # for classic
            q1 = [i['source_file'] for i in self.test_queries]
            q2 = [i['target_file'] for i in self.test_queries]
            self.test_set_images = list(set(q1 + q2))

    def get_different_word(self, source_caption, target_caption):
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

    def generate_test_queries_(self):
        file2imgid = {}
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
        with open(self.img_path / 'test_queries.txt', encoding="utf-8") as f:
            lines = f.readlines()
        self.test_queries = []
        for line in lines:
            source_file, target_file = line.split()
            idx = file2imgid[source_file]
            target_idx = file2imgid[target_file]
            source_caption = self.imgs[idx]['captions'][0]
            target_caption = self.imgs[target_idx]['captions'][0]
            source_word, target_word, mod_str = self.get_different_word(
                source_caption, target_caption)

            self.test_queries += [{
                'source_file': source_file,
                'source_caption': source_caption,
                'target_file': target_file,
                'target_caption': target_caption,
                'mod': mod_str,
            }]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly later"""

        # index caption 2 caption_id and caption 2 image_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.imgs):
            for c in img['captions']:
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        print(len(caption2imgids), 'unique cations')

        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        # identify parent captions for each image
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
        print('Modifiable images', num_modifiable_imgs)

    def caption_index_sample_(self, idx):
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))

        # find random target image (same parent)
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['captions'][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        return idx, target_idx, source_word, target_word, mod_str

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img['captions']:
                texts.append(c)
        return texts

    def __len__(self):

        if self.mode == 'classic':
            return len(self.test_set_images)

        if self.split == 'train':
            return len(self.imgs)
        else:
            return len(self.test_queries)

    def __getitem__(self, idx):

        if self.mode == "relative":

            if self.split == 'train':
                idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(idx)
                reference_image = self.get_img(idx)
                target_image = self.get_img(target_idx)

                return reference_image, target_image, mod_str
            else:
                reference_path = self.test_queries[idx]['source_file']
                target_path = self.test_queries[idx]['target_file']
                mod_str = self.test_queries[idx]['mod']

                return reference_path, target_path, mod_str

        elif self.mode == 'classic':

            img_path = self.test_set_images[idx]
            # img = self.preprocess(PIL.Image.open(self.img_path / f"{self.test_set_images[idx]}"))

            image = self.preprocess(images=PIL.Image.open(self.img_path / f"{self.test_set_images[idx]}"), return_tensors='pt',
                                    padding=True)
            image['pixel_values'] = image['pixel_values'].squeeze(0)

            return img_path, image

        else:
            raise ValueError("mode should be in ['relative', 'classic']")

    def get_img(self, idx, raw_img=False):
        img_path = self.img_path / f"{self.imgs[idx]['file_path']}"
        # img = self.preprocess(PIL.Image.open(img_path))

        image = self.preprocess(images=PIL.Image.open(img_path), return_tensors='pt',
                                          padding=True)
        image['pixel_values'] = image['pixel_values'].squeeze(0)

        return image