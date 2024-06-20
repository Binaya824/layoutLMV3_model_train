import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# Load the trained model
model = LayoutLMv3ForTokenClassification.from_pretrained('../inputs/layoutlmv3Microsoft', local_files_only=True)
checkpoint = torch.load('./model_20.bin')

# Create a new state dict that maps keys from the loaded checkpoint to the model
new_state_dict = {}
for key, value in checkpoint.items():
    new_key = key.replace('model.', '')  # Adjust prefix if needed
    new_state_dict[new_key] = value

# Load the new state dict into the model
model.load_state_dict(new_state_dict)

# Load the processor
processor = LayoutLMv3Processor.from_pretrained('../inputs/layoutlmv3Microsoft', local_files_only=True)

# Convert PDF to images
pdf_path = './T.S-GWR.pdf'
images = convert_from_path(pdf_path)

# Function to preprocess image and get OCR data
def preprocess_image(image):
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = ocr_data['text']
    boxes = []
    for i in range(len(words)):
        (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
        boxes.append([x, y, x + w, y + h])
    return words, boxes

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to process a single page
def process_page(image):
    words, boxes = preprocess_image(image)
    encoded_inputs = processor(images=image, words=words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")

    # Move tensors to the same device as the model
    for k, v in encoded_inputs.items():
        encoded_inputs[k] = v.to(device)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        logits = outputs.logits

    # Process logits to get predictions
    predictions = torch.argmax(logits, dim=2)
    predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]

    return words, predicted_labels

# Function to extract key-value pairs from predictions
def extract_key_value_pairs(words, labels):
    key_value_pairs = {}
    current_key = None
    current_value = []
    for word, label in zip(words, labels):
        if label.startswith("B-"):  # Beginning of a key
            if current_key:
                key_value_pairs[current_key] = " ".join(current_value)
            current_key = label[2:]
            current_value = [word]
        elif label.startswith("I-") and current_key:  # Continuation of a key
            current_value.append(word)
        elif label == "O" and current_key:  # Outside any key, end current key-value pair
            key_value_pairs[current_key] = " ".join(current_value)
            current_key = None
            current_value = []
    if current_key:
        key_value_pairs[current_key] = " ".join(current_value)  # Add the last key-value pair
    return key_value_pairs

# Process each page and collect key-value pairs
all_key_value_pairs = []
for image in images:
    words, predicted_labels = process_page(image)
    key_value_pairs = extract_key_value_pairs(words, predicted_labels)
    all_key_value_pairs.append(key_value_pairs)

# Print or save the extracted key-value pairs
for i, key_value_pairs in enumerate(all_key_value_pairs):
    print(f"Page {i+1}:")
    for key, value in key_value_pairs.items():
        print(f"{key}: {value}")
