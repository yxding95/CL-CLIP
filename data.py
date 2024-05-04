import json
import torch
from torchvision import transforms as T
import numpy as np
import random
import os
from PIL import Image
from models.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.utils.data import Dataset

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
        
        image_path = self.img_file + 'COCO_%s_%012d/%d.jpg' %(self.split, self.meta[index]['image_id'], self.meta[index]['obj_id'])
        
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        img_crop = self.transform(img)
        
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
        
        image_path = self.img_file + 'COCO_%s_%012d/%d.jpg' %(self.split, self.meta[index]['image_id'], self.meta[index]['obj_id'])
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        img_crop = self.transform(img)
        
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