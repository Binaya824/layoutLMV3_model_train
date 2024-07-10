import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import numpy as np
from engine import *
from trainer import *
from loader import *
from utils import *
import torch.nn.functional as nnf

# Initialize the processor
model_name_or_path = "hmart824/layoutlmv3-finetuned"  # Update with your model path
processor = LayoutLMv3Processor.from_pretrained(model_name_or_path, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name_or_path)

# Load the image
image = Image.open("./image/Training_Images/page_1.png")
image.show()
test_dict, width_scale, height_scale = dataSetFormat(image)

# Normalize bbox coordinates
image_width, image_height = image.size
test_dict['bboxes'] = normalize_bboxes(test_dict['bboxes'], image_width, image_height)

print("test_dict['bboxes']:", test_dict['bboxes'])

# Process the inputs
encoding = processor(
    text=test_dict['tokens'],
    images=image.convert('RGB'),
    boxes=test_dict['bboxes'],
    max_length=256,
    padding="max_length",
    truncation=True,
    return_tensors='pt',
    return_offsets_mapping=True
)

print("encoding['bbox']:", encoding['bbox'])

# Load model state
model.load_state_dict(torch.load("./model_20.bin"))

input_ids = encoding['input_ids'].squeeze()
attention_mask = encoding['attention_mask'].squeeze()
bbox = encoding['bbox'].squeeze()
pixel_values = encoding['pixel_values'].squeeze()

print("bbox:", bbox)

with torch.no_grad():
    outputs = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        bbox=bbox.unsqueeze(0),
        pixel_values=pixel_values.unsqueeze(0)
    )
    predictions = outputs.logits.argmax(-1).squeeze().tolist()

    prob = nnf.softmax(outputs.logits, dim=2)
    txt = prob.squeeze().numpy() / np.sum(prob.squeeze().numpy(), axis=1).reshape(-1, 1)
    output_prob = np.max(txt, axis=1)

pred = torch.tensor(predictions)
offset_mapping = encoding['offset_mapping']
is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
true_predictions = torch.tensor([pred[idx].item() for idx in range(len(pred)) if not is_subword[idx]])

true_prob = torch.tensor([output_prob[idx].item() for idx in range(len(output_prob)) if not is_subword[idx]])

true_boxes = torch.tensor([bbox[idx].tolist() for idx in range(len(bbox)) if not is_subword[idx]])

concat_torch = torch.column_stack((true_boxes, true_predictions, true_prob))

# Extract results for all 8 labels
labels = list(range(1, 9))  # Assuming labels are 1 through 8
class_results = []

for label in labels:
    class_result = concat_torch[torch.where(
        (concat_torch[:, 4] == label) &
        (concat_torch[:, 3] == 0) &
        (concat_torch[:, 2] == 0)
    )]
    class_results.append(class_result)

finl = torch.row_stack(class_results)
unique_ = torch.unique(finl, dim=0)

plot_img(test_dict['img_path'], unique_[:, :4], unique_[:, 4].tolist(), unique_[:, 5].tolist(), width_scale, height_scale)

print(unique_)
