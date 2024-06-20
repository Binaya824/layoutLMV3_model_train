import fitz  # PyMuPDF
from PIL import Image
import io
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import json

# Load the trained LayoutLMv3 model and processor
model_name_or_path = "hmart824/layoutlmv3"  # Update with your model path

try:
    processor = LayoutLMv3Processor.from_pretrained(model_name_or_path , apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_name_or_path)
except Exception as e:
    print(f"Error loading model '{model_name_or_path}': {str(e)}")
    processor, model = None, None  # Set processor and model to None if loading fails

def extract_text_from_page(page):
    words = []
    boxes = []

    for word in page.get_text("words"):
        words.append(word[4])
        boxes.append(fitz.Rect(word[:4]))

    return words, boxes

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * bbox.x0 / width),
        int(1000 * bbox.y0 / height),
        int(1000 * bbox.x1 / width),
        int(1000 * bbox.y1 / height),
    ]

def page_to_image(page):
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("ppm")
    img = Image.open(io.BytesIO(img_bytes))
    return img

def process_pdf(file_path):
    if processor is None or model is None:
        print("Model and processor are not loaded. Exiting...")
        return

    # Open the PDF file
    doc = fitz.open(file_path)

    json_output = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        words, boxes = extract_text_from_page(page)
        width, height = page.rect.width, page.rect.height

        # Convert the page to an image
        image = page_to_image(page)

        encoding = processor(
            text=words,
            images=image,
            boxes=[normalize_bbox(box, width, height) for box in boxes],
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        with torch.no_grad():
            outputs = model(**encoding)

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        tokens = processor.tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze().tolist())
        token_boxes = encoding.bbox.squeeze().tolist()

        page_output = []

        for token, box, prediction in zip(tokens, token_boxes, predictions):
            if token in processor.tokenizer.all_special_tokens:
                continue
            label = model.config.id2label[prediction]
            page_output.append({
                "text": token,
                "bbox": box,
                "label": label
            })

        json_output.append({
            "page": page_num + 1,
            "content": page_output
        })

    return json_output

def main():
    pdf_path = "./T.S-GWR.pdf"  # Update with your PDF file path
    result = process_pdf(pdf_path)

    if result:
        # Save to JSON file
        with open("output.json", "w") as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
