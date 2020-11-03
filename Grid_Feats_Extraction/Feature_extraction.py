'''
    使用 grid-feats-vqa 来抽取特征
    Extract Grid Features Of Your Customized Datasets Using https://github.com/facebookresearch/grid-feats-vqa
'''
import json
from PIL import Image
import os
from Grid_Feats_Extraction.COCO_format import coco_example
############
# First Step
############
'''
    You need to create json file of your dataset and this json file should be in COCO.json format.
    This json file should have four keys: ['info', 'images', 'licenses', 'categories']
    you can get the coco.json format example from this file './COCO_format.py'
'''
example = coco_example()

def show_example(example):
    coco_example = example.get_coco_example()
    print(coco_example)

# show_example(example)
