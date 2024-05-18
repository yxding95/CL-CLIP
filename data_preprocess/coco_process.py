# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:05:07 2019

@author: yuxuan
"""


import json

with open('instances_train2014.json') as f:
    trainset = json.load(f)

with open('instances_val2014.json') as f:
    valset = json.load(f)

with open('dataset_coco.json') as f:
    GT = json.load(f)
    GT = GT['images']

test5k = []
val5k = []
for j,i in enumerate(GT):
    if i['split'] == 'test':
        test5k.append(i['filename'])
    elif i['split'] == 'val':
        val5k.append(i['filename'])

crop = {}
crop['categories'] = trainset['categories']

crop['train'] = []
for i in trainset['annotations']:
    bbox = i['bbox']
    if not(bbox[2]<=4. or bbox[3]<=4.):
        crop['train'].append(i)

crop['val'] = []
for i in valset['annotations']:
    bbox = i['bbox']
    img_id = i['image_id']
    image_path = 'COCO_val2014_%012d.jpg' %img_id
    if not(bbox[2]<=4. or bbox[3]<=4.) and (image_path in val5k):
        crop['val'].append(i)

crop['test'] = []
for i in valset['annotations']:
    bbox = i['bbox']
    img_id = i['image_id']
    image_path = 'COCO_val2014_%012d.jpg' %img_id
    if not(bbox[2]<=4. or bbox[3]<=4.) and (image_path in test5k):
        crop['test'].append(i)

with open('cl_clip_instances_train2014.json', 'w') as f:
    json.dump(crop, f)

phase = {}
phase['categories'] = trainset['categories']
phase['val'] = crop['val']
phase['test'] = crop['test']
phase['phase'] = {}

with open('cl_clip_instances_train2014.json') as f:
    data = json.load(f)

categories = [j['id'] for i, j in enumerate(data['categories'])]

for anno in data['train']:
    cid = anno['category_id']
    cid_phase = categories.index(cid)
    cid_phase = cid_phase // 10
    phase['phase'].setdefault(str(cid_phase), [])
    phase['phase'][str(cid_phase)].append(anno)

with open('cl_clip_phase_instances_train2014.json', 'w') as f:
    json.dump(phase, f)