import json
import torch
from torchvision import transforms as T
import numpy as np
import random
import os
import re
from PIL import Image
from models.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageFolder


class OST(Dataset):
    def __init__(self, data_file, img_file, pseudo_cls=None, pseudo_length=None, split='train'):
        with open(data_file, 'r') as f:
            data = json.load(f)

        self.meta = data[split]

        if split == 'train':
            self.split = 'train2014'
            self.transform = T.Compose([
                            T.Resize(256, interpolation=Image.BICUBIC),
                            T.RandomCrop(224),
                            lambda image: image.convert("RGB"),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                        ])
        else:
            self.split = 'val2014'
            self.transform = T.Compose([
                            T.Resize(224, interpolation=Image.BICUBIC),
                            T.CenterCrop(224),
                            lambda image: image.convert("RGB"),
                            T.ToTensor(),
                            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                        ])

        self.img_file = img_file
        categories = [(j['id'], i) for i, j in enumerate(data['categories'])]
        categories = dict(categories)
        self.target_transform = lambda x: categories[x]
        self.categories_name = [j['name'] for i, j in enumerate(data['categories'])]
        #print(self.categories_name)
        
        self._tokenizer = _Tokenizer()
        self.contex_legth = 77
        
        self.pseudo_cls = pseudo_cls
        self.pseudo_length = pseudo_length

    def __len__(self):
        return len(self.meta)

    def get_ClsName(self):
        class_num = len(self.categories_name)
        results = torch.zeros(class_num, self.contex_legth, dtype=torch.long)
        
        for i in range(len(self.categories_name)):
            text_label = self.categories_name[i]
            caption = "this is a photo of " + text_label + " in the scene ."
            sot_token = [self._tokenizer.encoder["<|startoftext|>"]]
            eot_token = [self._tokenizer.encoder["<|endoftext|>"]]
            all_tokens = sot_token + self._tokenizer.encode(caption) + eot_token
            results[i, :len(all_tokens)] = torch.tensor(all_tokens)
        return results, self.categories_name

    def get_PseudoCls(self):
        class_num = self.pseudo_cls
        results = torch.randint(high=self._tokenizer.encoder["<|startoftext|>"]-1, size=(class_num, self.pseudo_length))
        results[:, 0] = self._tokenizer.encoder["<|startoftext|>"]
        results[:, -1] = self._tokenizer.encoder["<|endoftext|>"]

        random_mem = torch.zeros((class_num, self.contex_legth), dtype=torch.long)
        random_mem[:, :self.pseudo_length] = results
        
        return random_mem

    def __getitem__(self, index):
        
        image_path = self.img_file + '%s/COCO_%s_%012d.jpg' %(self.split, self.split, self.meta[index]['image_id'])
        
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        bbox = self.meta[index]['bbox']
        bbox = [bbox[0], bbox[1], (bbox[0]+bbox[2]), (bbox[1]+bbox[3])]
        img_crop = img.crop(bbox)
        img_crop = self.transform(img_crop)
        
        category_id = self.meta[index]['category_id']
        target = self.target_transform(category_id)
        target_name = self.categories_name[target]
        caption = "this is a photo of " + target_name + " in the scene ."
        target_tokenized = self._tokenizer.encode(target_name)
        
        sot_token = [self._tokenizer.encoder["<|startoftext|>"]]
        eot_token = [self._tokenizer.encoder["<|endoftext|>"]]
        
        all_tokens = sot_token + self._tokenizer.encode(caption) + eot_token
        result = torch.zeros(self.contex_legth, dtype=torch.long)
        result[:len(all_tokens)] = torch.tensor(all_tokens)
        
        sample = {'image':img_crop, 'target':target, 'caption':result, 'raw':caption}
        
        return sample

class MST(Dataset):
    def __init__(self, data_file, img_file, phase, pseudo_cls=None, pseudo_length=None):
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        phase = str(phase)
        self.meta = data['phase'][phase]
        self.split = 'train2014'
        self.img_file = img_file

        categories = [(j['id'], i) for i, j in enumerate(data['categories'])]
        categories = dict(categories)
        self.target_transform = lambda x: categories[x]
        self.categories_name = [j['name'] for i, j in enumerate(data['categories'])]
        #print(self.categories_name)
        
        self.transform = T.Compose([
                        T.Resize(256, interpolation=Image.BICUBIC),
                        T.RandomCrop(224),
                        lambda image: image.convert("RGB"),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
        self._tokenizer = _Tokenizer()
        self.contex_legth = 77

        self.pseudo_cls = pseudo_cls
        self.pseudo_length = pseudo_length
        
    def __len__(self):
        return len(self.meta)

    def get_ClsName(self):
        class_num = len(self.categories_name)
        results = torch.zeros(class_num, self.contex_legth, dtype=torch.long)
        
        for i in range(len(self.categories_name)):
            text_label = self.categories_name[i]
            caption = "this is a photo of " + text_label + " in the scene ."

            sot_token = [self._tokenizer.encoder["<|startoftext|>"]]
            eot_token = [self._tokenizer.encoder["<|endoftext|>"]]
            all_tokens = sot_token + self._tokenizer.encode(caption) + eot_token
            results[i, :len(all_tokens)] = torch.tensor(all_tokens)
        return results, self.categories_name

    def get_PseudoCls(self):
        class_num = self.pseudo_cls
        results = torch.randint(high=self._tokenizer.encoder["<|startoftext|>"]-1, size=(class_num, self.pseudo_length))
        results[:, 0] = self._tokenizer.encoder["<|startoftext|>"]
        results[:, -1] = self._tokenizer.encoder["<|endoftext|>"]

        random_mem = torch.zeros((class_num, self.contex_legth), dtype=torch.long)
        random_mem[:, :self.pseudo_length] = results
        
        return random_mem
    
    def __getitem__(self, index):

        image_path = self.img_file + '%s/COCO_%s_%012d.jpg' %(self.split, self.split, self.meta[index]['image_id'])
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        bbox = self.meta[index]['bbox']
        bbox = [bbox[0], bbox[1], (bbox[0]+bbox[2]), (bbox[1]+bbox[3])]
        img_crop = img.crop(bbox)
        img_crop = self.transform(img_crop)

        category_id = self.meta[index]['category_id']
        target = self.target_transform(category_id)
        target_name = self.categories_name[target]
        caption = "this is a photo of " + target_name + " in the scene ."
        target_tokenized = self._tokenizer.encode(target_name)
        
        sot_token = [self._tokenizer.encoder["<|startoftext|>"]]
        eot_token = [self._tokenizer.encoder["<|endoftext|>"]]
        
        all_tokens = sot_token + self._tokenizer.encode(caption) + eot_token
        result = torch.zeros(self.contex_legth, dtype=torch.long)
        result[:len(all_tokens)] = torch.tensor(all_tokens)
        
        sample = {'image':img_crop, 'target':target, 'caption':result, 'raw':caption}
        return sample

class RetriTask(Dataset):
    def __init__(self, data_file, img_file):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        
        self.img_file = img_file
        self.transform = T.Compose([
                        T.Resize(224, interpolation=Image.BICUBIC),
                        T.CenterCrop(224),
                        lambda image: image.convert("RGB"),
                        T.ToTensor(),
                        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
        self._tokenizer = _Tokenizer()
        self.contex_legth = 77    
    
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
    
        image_path = os.path.join(self.img_file + self.meta[index]['image'])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        
        text_label = self.meta[index]['sent'].lower()
        caption = text_label
        
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [sot_token] + self._tokenizer.encode(caption)[:75] + [eot_token]
        result = torch.zeros(self.contex_legth, dtype=torch.long)
        result[:len(all_tokens)] = torch.tensor(all_tokens)
        
        sample = {'image':img, 'caption':result}
        return sample

class ZSTask(Dataset):
    def __init__(self, data_file, img_file):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        
        self.img_file = img_file
        self.transform = T.Compose([
                        T.Resize(224, interpolation=Image.BICUBIC),
                        T.CenterCrop(224),
                        lambda image: image.convert("RGB"),
                        T.ToTensor(),
                        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
        
        self._tokenizer = _Tokenizer()
        self.contex_legth = 77
        
    def __len__(self):
        return len(self.meta['image_names'])
        
    def get_ClsName(self):
        class_num = len(self.meta['label_names'])
        result = torch.zeros(class_num, self.contex_legth, dtype=torch.long)
        
        for i in range(len(self.meta['label_names'])):
            text_label = self.meta['label_names'][i].lower()
            caption = "this is a photo of "+text_label+" ."
            #caption = text_label
            
            sot_token = self._tokenizer.encoder["<|startoftext|>"]
            eot_token = self._tokenizer.encoder["<|endoftext|>"]
            all_tokens = [sot_token] + self._tokenizer.encode(caption) + [eot_token]
            result[i, :len(all_tokens)] = torch.tensor(all_tokens)
        return result, self.meta['label_names']
    
    def __getitem__(self, index):

        image_path = os.path.join(self.img_file + self.meta['image_names'][index])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        
        target = self.meta['image_labels'][index]
        text_label = self.meta['label_names'][target].lower()
        caption = "this is a photo of "+text_label+" ."
        #caption = text_label
        
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [sot_token] + self._tokenizer.encode(caption) + [eot_token]
        result = torch.zeros(self.contex_legth, dtype=torch.long)
        result[:len(all_tokens)] = torch.tensor(all_tokens)

        sample = {'image':img, 'target':target, 'caption':result}
        return sample

class Replayed(Dataset):
    def __init__(self, data_file):
        
        meta = []
        for (root, dirs, files) in os.walk(data_file, topdown=True):
            for name in files:
                if os.path.splitext(name)[1] == ".json":
                    with open(os.path.join(root, name), "r") as f:
                        inst = json.load(f)
                        meta.append({"image_id": inst["key"]+".jpg", "caption": inst["caption"]})        

        self.meta = meta
        self.split = 'train2014'
        self.img_file = data_file
        
        self.transform = T.Compose([
                        T.Resize(256, interpolation=Image.BICUBIC),
                        T.RandomCrop(224),
                        lambda image: image.convert("RGB"),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
        self._tokenizer = _Tokenizer()
        self.contex_legth = 77
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        
        image_path = self.img_file + self.meta[index]['image_id']
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        img_crop = self.transform(img)
        
        caption = self.meta[index]['caption']
        target = -1
        
        sot_token = [self._tokenizer.encoder["<|startoftext|>"]]
        eot_token = [self._tokenizer.encoder["<|endoftext|>"]]
        
        all_tokens = sot_token + self._tokenizer.encode(caption)[:75] + eot_token
        result = torch.zeros(self.contex_legth, dtype=torch.long)
        result[:len(all_tokens)] = torch.tensor(all_tokens)
        
        sample = {'image':img_crop, 'target':target, 'caption':result, 'raw':caption}
        return sample

class Replayed_MST(Dataset):
    def __init__(self, data_file, img_file, phase, pseudo_cls=None, pseudo_length=None):
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        phase = str(phase)
        self.meta = random.sample(data['phase'][phase], 250)
        self.split = 'train2014'
        self.img_file = img_file

        categories = [(j['id'], i) for i, j in enumerate(data['categories'])]
        categories = dict(categories)
        self.target_transform = lambda x: categories[x]
        self.categories_name = [j['name'] for i, j in enumerate(data['categories'])]
        #print(self.categories_name)
        
        self.transform = T.Compose([
                        T.Resize(256, interpolation=Image.BICUBIC),
                        T.RandomCrop(224),
                        lambda image: image.convert("RGB"),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
        self._tokenizer = _Tokenizer()
        self.contex_legth = 77

        self.pseudo_cls = pseudo_cls
        self.pseudo_length = pseudo_length
        
    def __len__(self):
        return len(self.meta)

    def get_ClsName(self):
        class_num = len(self.categories_name)
        results = torch.zeros(class_num, self.contex_legth, dtype=torch.long)
        
        for i in range(len(self.categories_name)):
            text_label = self.categories_name[i]
            caption = "this is a photo of " + text_label + " in the scene ."

            sot_token = [self._tokenizer.encoder["<|startoftext|>"]]
            eot_token = [self._tokenizer.encoder["<|endoftext|>"]]
            all_tokens = sot_token + self._tokenizer.encode(caption) + eot_token
            results[i, :len(all_tokens)] = torch.tensor(all_tokens)
        return results, self.categories_name

    def get_PseudoCls(self):
        class_num = self.pseudo_cls
        results = torch.randint(high=self._tokenizer.encoder["<|startoftext|>"]-1, size=(class_num, self.pseudo_length))
        results[:, 0] = self._tokenizer.encoder["<|startoftext|>"]
        results[:, -1] = self._tokenizer.encoder["<|endoftext|>"]

        random_mem = torch.zeros((class_num, self.contex_legth), dtype=torch.long)
        random_mem[:, :self.pseudo_length] = results
        
        return random_mem
    
    def __getitem__(self, index):

        image_path = self.img_file + '%s/COCO_%s_%012d.jpg' %(self.split, self.split, self.meta[index]['image_id'])
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        bbox = self.meta[index]['bbox']
        bbox = [bbox[0], bbox[1], (bbox[0]+bbox[2]), (bbox[1]+bbox[3])]
        img_crop = img.crop(bbox)
        img_crop = self.transform(img_crop)

        category_id = self.meta[index]['category_id']
        target = self.target_transform(category_id)
        target_name = self.categories_name[target]
        caption = "this is a photo of " + target_name + " in the scene ."
        target_tokenized = self._tokenizer.encode(target_name)
        
        sot_token = [self._tokenizer.encoder["<|startoftext|>"]]
        eot_token = [self._tokenizer.encoder["<|endoftext|>"]]
        
        all_tokens = sot_token + self._tokenizer.encode(caption) + eot_token
        result = torch.zeros(self.contex_legth, dtype=torch.long)
        result[:len(all_tokens)] = torch.tensor(all_tokens)
        
        sample = {'image':img_crop, 'target':target, 'caption':result, 'raw':caption}
        return sample

class EuroSAT(ImageFolder):
    def __init__(self, root, is_test=False, is_replay=False, phase=None, transform=None, target_transform=None, pseudo_cls=None, pseudo_length=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.transform = T.Compose([
                        T.Resize(256, interpolation=Image.BICUBIC),
                        T.RandomCrop(224),
                        lambda image: image.convert("RGB"),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
        self._tokenizer = _Tokenizer()
        self.contex_legth = 77
        self.pseudo_cls = pseudo_cls
        self.pseudo_length = pseudo_length
        self.idx_to_class = {self.class_to_idx[k]: " ".join(re.findall('[A-Z][^A-Z]*', k)).lower() for k in self.class_to_idx.keys()}

        samples_split = {_:[] for _ in list(self.idx_to_class.keys())}
        for (path, target) in self.imgs:
            samples_split[target].append((path, target))

        if is_test:
            self.samples = []
            for _ in samples_split:
                self.samples.extend(samples_split[_][-500:])
        else:
            self.samples = []
            if phase is None:
                for _ in samples_split:
                    self.samples.extend(samples_split[_][:-500])
            else:
                for _ in samples_split:
                    if _//2 == phase:
                        if is_replay:
                            self.samples.extend(random.sample(samples_split[_][:-500], 250))
                        else:
                            self.samples.extend(samples_split[_][:-500])
                    else:
                        continue

    def get_ClsName(self):
        class_num = len(self.class_to_idx)
        results = torch.zeros(class_num, self.contex_legth, dtype=torch.long)
        
        for i in range(len(self.idx_to_class)):
            text_label = self.idx_to_class[i]
            caption = "this is a satellite photo of " + text_label + " ."

            sot_token = [self._tokenizer.encoder["<|startoftext|>"]]
            eot_token = [self._tokenizer.encoder["<|endoftext|>"]]
            all_tokens = sot_token + self._tokenizer.encode(caption) + eot_token
            results[i, :len(all_tokens)] = torch.tensor(all_tokens)
        return results, self.idx_to_class

    def get_PseudoCls(self):
        class_num = self.pseudo_cls
        results = torch.randint(high=self._tokenizer.encoder["<|startoftext|>"]-1, size=(class_num, self.pseudo_length))
        results[:, 0] = self._tokenizer.encoder["<|startoftext|>"]
        results[:, -1] = self._tokenizer.encoder["<|endoftext|>"]

        random_mem = torch.zeros((class_num, self.contex_legth), dtype=torch.long)
        random_mem[:, :self.pseudo_length] = results
        
        return random_mem

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        target_name = self.idx_to_class[target]
        caption = "this is a satellite photo of " + target_name + " ."
        target_tokenized = self._tokenizer.encode(target_name)
        
        sot_token = [self._tokenizer.encoder["<|startoftext|>"]]
        eot_token = [self._tokenizer.encoder["<|endoftext|>"]]
        
        all_tokens = sot_token + self._tokenizer.encode(caption) + eot_token
        result = torch.zeros(self.contex_legth, dtype=torch.long)
        result[:len(all_tokens)] = torch.tensor(all_tokens)

        sample = {'image':sample, 'target':target, 'caption':result, 'raw':caption}
        return sample

if __name__ == "__main__":
    pass