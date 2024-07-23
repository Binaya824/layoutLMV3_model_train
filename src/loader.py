# Loader

from utils import *
import torch
from tqdm import tqdm
from PIL import Image


class dataSet:
    def __init__(self,json_path,processor=None) -> None:
        self.json_data = train_data_format(read_json(json_path))

        self.processor = processor


    def __len__(self)->int:
        # print(self.json_data)
        return len(self.json_data)

    def __getitem__(self, index) -> dict:
        data = self.json_data[index]

        # Open the image and prepare the tokens, labels, and bounding boxes
        image = Image.open(data['img_path']).convert('RGB')
        words = data['tokens']
        label = data['ner_tag']
        bboxes = data['bboxes']

        # Ensure words, labels, and bboxes are lists
        if not isinstance(words, list):
            words = [words]
        if not isinstance(label, list):
            label = [label]
        if not isinstance(bboxes, list):
            bboxes = [bboxes]

        # Process the input data using the processor
        encoding = self.processor(
            images=image,
            text=words,
            boxes=bboxes,
            word_labels=label,
            max_length=512,
            padding="max_length",
            truncation=True,  # Use truncation=True to handle sequences longer than max_length
            return_tensors='pt'
        )

        # Flatten the tensors where appropriate and ensure correct dtype
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox": encoding["bbox"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["label"].squeeze(0)
        }

