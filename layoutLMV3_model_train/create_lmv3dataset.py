import json
from transformers import LayoutLMv3TokenizerFast

# Load JSON file
json_file = './project-2-at-2024-07-18-05-03-e52ddc23.json'
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize LayoutLMv3 tokenizer
tokenizer = LayoutLMv3TokenizerFast.from_pretrained('microsoft/layoutlmv3-base')

# Function to prepare dataset
def prepare_dataset(data):
    dataset = []
    for item in data:
        ocr_text = item['transcription']
        token_boxes = []

        # Prepare token text and corresponding bounding boxes
        for token_text in ocr_text:
            found_box = False
            for bbox in item['bbox']:
                if token_text in bbox.get('transcription', ''):
                    token_boxes.append([bbox['x'], bbox['y'], bbox['width'], bbox['height']])
                    found_box = True
                    break
            if not found_box:
                # If no corresponding bbox found, handle appropriately
                token_boxes.append([0, 0, 0, 0])  # Placeholder for missing bounding boxes

        # Tokenize OCR text with associated bounding boxes
        inputs = tokenizer(ocr_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512,
                           token_boxes=token_boxes)

        # Add labels (ignore_content labels can be used to mask during training)
        labels = [1 if 'ignore_content' in bbox.get('labels', []) else 0 for bbox in item.get('label', [])]

        # Prepare dataset entry
        entry = {
            'input_ids': inputs['input_ids'].squeeze(0).tolist(),
            'attention_mask': inputs['attention_mask'].squeeze(0).tolist(),
            'boxes': token_boxes,
            'labels': labels
        }
        dataset.append(entry)

    return dataset

# Prepare the dataset
dataset = prepare_dataset(data)

# Save dataset to a JSON file
output_json_file = 'layoutlmv3_dataset.json'
with open(output_json_file, 'w', encoding='utf-8') as json_output_file:
    json.dump(dataset, json_output_file, ensure_ascii=False, indent=4)

print(f"Dataset saved to {output_json_file}")
