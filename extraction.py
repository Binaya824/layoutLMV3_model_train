import fitz  # PyMuPDF
from PIL import Image
import io
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import json

# Load the trained LayoutLMv3 model and processor
model_name_or_path = "hmart824/layoutlmv3-finetuned_1.2"  # Update with your model path

try:
    processor = LayoutLMv3Processor.from_pretrained(model_name_or_path, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_name_or_path)
except Exception as e:
    print(f"Error loading model '{model_name_or_path}': {str(e)}")
    processor, model = None, None  # Set processor and model to None if loading fails

def extract_text_from_page(page):
    sentences = []
    lines = page.get_text("blocks")
    
    for line in lines:
        if line[4]:  # Ensure there is text in the block
            sentences.append((line[4], fitz.Rect(line[:4])))

    return sentences

def page_to_image(page):
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("ppm")
    img = Image.open(io.BytesIO(img_bytes))
    return img

def convert_bbox_to_int(bbox):
    return [int(coord) for coord in bbox]

def generate_word_bboxes(sentence, bbox):
    words = sentence.split()
    word_bboxes = []
    x1, y1, x2, y2 = bbox
    word_width = (x2 - x1) / len(words)
    for i, word in enumerate(words):
        word_bbox = [
            x1 + i * word_width,
            y1,
            x1 + (i + 1) * word_width,
            y2
        ]
        word_bboxes.append(convert_bbox_to_int(word_bbox))
    return words, word_bboxes

def process_pdf(file_path):
    if processor is None or model is None:
        print("Model and processor are not loaded. Exiting...")
        return

    # Open the PDF file
    doc = fitz.open(file_path)

    json_output = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        sentences = extract_text_from_page(page)
        width, height = page.rect.width, page.rect.height

        # Convert the page to an image
        image = page_to_image(page)

        page_output = []

        for sentence, bbox in sentences:
            words, word_bboxes = generate_word_bboxes(sentence, convert_bbox_to_int(bbox))
            encoding = processor(
                text=[words],  # Wrap the list of words in another list
                images=image,
                boxes=[word_bboxes],  # Use the generated word bounding boxes
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )

            with torch.no_grad():
                outputs = model(**encoding)

            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            tokens = processor.tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze().tolist())
            token_boxes = encoding.bbox.squeeze().tolist()

            sentence_output = []

            for token, box, prediction in zip(tokens, token_boxes, predictions):
                if token in processor.tokenizer.all_special_tokens:
                    continue
                label = model.config.id2label[prediction]
                sentence_output.append({
                    "text": token,
                    "bbox": box,
                    "label": label
                })

            page_output.append({
                "sentence": sentence,
                "bbox": convert_bbox_to_int(bbox),
                "content": sentence_output
            })

        json_output.append({
            "page": page_num + 1,
            "content": page_output
        })

    return json_output

def main():
    pdf_path = "./document with list style dot.pdf"  # Update with your PDF file path
    result = process_pdf(pdf_path)

    if result:
        # Save to JSON file
        with open("output.json", "w") as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
