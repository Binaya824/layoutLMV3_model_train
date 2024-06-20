# utils.py

from paddleocr import PaddleOCR
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from typing import List


def read_json(json_path: str) -> dict:
    with open(json_path, 'r') as fp:
        data = json.loads(fp.read())
    return data


def train_data_format(json_to_dict: list) -> list:
    final_list = []
    count = 0
    for item in json_to_dict:
        count += 1
        test_dict = {"id": int, "tokens": [], "bboxes": [], "ner_tag": []}
        test_dict["id"] = count
        test_dict["img_path"] = item['file_name']
        for cont in item['annotations']:
            test_dict['tokens'].append(cont['text'])
            test_dict['bboxes'].append(cont['box'])
            test_dict['ner_tag'].append(cont['label'])
        final_list.append(test_dict)
    return final_list


ocr = PaddleOCR(use_angle_cls=False, lang='en', rec=False)


def scale_bounding_box(box: List[int], width: float, height: float) -> List[int]:
    return [
        100 * box[0] / width,
        100 * box[1] / height,
        (100 * box[0] / width) + box[2],
        (100 * box[1] / height) + box[3]
    ]


def process_bbox(box: list) -> list:
    return [box[0][0], box[1][1], box[2][0] - box[0][0], box[2][1] - box[1][1]]


def dataSetFormat(img_file):
    width, height = img_file.size
    ress = ocr.ocr(np.asarray(img_file))
    test_dict = {'tokens': [], "bboxes": []}
    test_dict['img_path'] = img_file
    for item in ress[0]:
        test_dict['tokens'].append(item[1][0])
        test_dict['bboxes'].append(scale_bounding_box(process_bbox(item[0]), width, height))
    return test_dict, width, height

def normalize_bboxes(bboxes, width, height):
    def normalize(value, scale):
        return min(int(1000 * (value / scale)), 1000)

    normalized_bboxes = []
    for bbox in bboxes:
        normalized_bboxes.append([
            normalize(bbox[0], width),
            normalize(bbox[1], height),
            normalize(bbox[2], width),
            normalize(bbox[3], height)
        ])
    return normalized_bboxes


def plot_img(im, bbox_list, label_list, prob_list, width, height):
    plt.imshow(im)
    ax = plt.gca()
    for i, (item) in enumerate(zip(bbox_list)):
        item = item[0]
        rect = Rectangle((item[0] * width / 100, item[1] * height / 100), item[2] - item[0], item[3] - item[1],
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(
            item[0] * width / 100,
            item[1] * height / 100,
            f"{label_list[i]}",
            bbox={'facecolor': [1, 1, 1], 'alpha': 0.5},
            clip_box=ax.clipbox,
            clip_on=True
        )
    plt.show()
    plt.savefig("test_image.jpg")
    plt.clf()
