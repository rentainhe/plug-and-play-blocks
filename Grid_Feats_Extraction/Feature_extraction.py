'''
    使用 grid-feats-vqa 来抽取特征
    Extract Grid Features Of Your Customized Datasets Using https://github.com/facebookresearch/grid-feats-vqa
'''
import json
from PIL import Image
import os
import tqdm
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

#############
# Second Step
#############
'''
    These APIs will help you to create your own coco.json file
'''
def get_image_name_list(path):
    '''
    获得存储图片文件夹下所有图片文件的名称，带.jpg后缀
    :param path:
    :return:
    '''
    image_name_list = os.listdir(path)
    return image_name_list

def create_image_info(image_path):
    '''
    :param image_path: where you store the original image
    :param image_name_list: 'path/image_name' to reach the original picture
    :return:
    '''
    image_name_list = get_image_name_list(image_path)
    images = []
    for i in tqdm(range(len(image_name_list))):
        try:
            temp = {}
            temp['license'] = 2
            temp['file_name'] = image_name_list[i]
            temp['coco_url'] = '' # It's OK to be nothing here
            temp_image = Image.open(image_path + image_name_list[i])
            temp['height'] = temp_image.height
            temp['width'] = temp_image.width
            temp['data_captured'] = '2020-11-3'
            temp['id'] = str(os.path.splitext(image_name_list[i])[0].split('_')[-1])
            images.append(temp)
        except:
            continue
    return images


def create_coco_file(example, image_path,save_path):
    '''

    :param example: 刚刚提供的范例数据，里面的info licences categories可以直接使用
    :return:
    '''
    coco_file = {}
    coco_file['info'] = example['info']
    coco_file['images'] = create_image_info(image_path)
    coco_file['licenses'] = example['licenses']
    coco_file['categories'] = example['categories']
    # sava as json file
    json_file = json.dumps(coco_file)
    with open(save_path,'w') as json_ :
        json_.write(json_file)

# example:
# create_coco_file(example,image_path="/home/luoyp/rth/VG/VG_100K_2/", save_path="/home/luoyp/rth/VG/VG_100K_images.json")

############
# Third Step
############
'''
    需要将数据集注册到 detectron2.data.datasets.builtin.py 文件下
    格式为
    "GQA": ("/home/luoyp/rth/GQA/GQA", "coco/annotations/GQA_images.json")
    数据集名称: (数据集图片位置，json文件位置）
'''