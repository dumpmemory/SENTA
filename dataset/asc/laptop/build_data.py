#! -*- coding:utf-8 -*-
import json
import numpy as np

train_data = []
with open('test_enriched.json') as f:
    content = json.load(f)
    length_ava = 0
    length_max = 0
    length = []
    num = 384
    count = 0
    results = {}
    print(len(content.keys()))
    for idx in content.keys():
        id = content[idx]['id']
        # if idx.split('_')[-1] == 'adv1' or idx.split('_')[-1] == 'adv2' or idx.split('_')[-1] == 'adv3':
        if idx.split('_')[-1] == 'adv2':
            count = count + 1
            results[idx] = content[idx]

        term = content[idx]['term']
        polarity = content[idx]['polarity']
        sentence = content[idx]['sentence']
        length_ava = len(sentence) + length_ava
        length_max = max(len(sentence), length_max)
        length.append(len(sentence))
        if num >= len(sentence):
            # count = count + 1
            a = 1

tmp = json.dumps(results)
with open('test_arts_adv2.json', 'w') as f2:
    f2.write(tmp)
    print(count)