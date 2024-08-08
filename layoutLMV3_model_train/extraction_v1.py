import fitz  # PyMuPDF
from PIL import Image
import io
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import json

# Load the trained LayoutLMv3 model and processor
model_name_or_path = "hmart824/layoutlmv3-finetuned_0.1"  # Update with your model path

try:
    processor = LayoutLMv3Processor.from_pretrained(model_name_or_path, apply_ocr=False)
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
        current_line_text = []
        current_line_boxes = []
        current_label = None

        for token, box, prediction in zip(tokens, token_boxes, predictions):
            if token in processor.tokenizer.all_special_tokens:
                continue
            label = model.config.id2label[prediction]

            if current_label is None:
                current_label = label

            if label != current_label:
                if current_line_text:
                    page_output.append({
                        "text": " ".join(current_line_text),
                        "bbox": [
                            min(b[0] for b in current_line_boxes),
                            min(b[1] for b in current_line_boxes),
                            max(b[2] for b in current_line_boxes),
                            max(b[3] for b in current_line_boxes)
                        ],
                        "label": current_label
                    })
                    current_line_text = []
                    current_line_boxes = []
                current_label = label

            current_line_text.append(token.replace("Ġ", ""))
            current_line_boxes.append(box)

        if current_line_text:
            page_output.append({
                "text": " ".join(current_line_text),
                "bbox": [
                    min(b[0] for b in current_line_boxes),
                    min(b[1] for b in current_line_boxes),
                    max(b[2] for b in current_line_boxes),
                    max(b[3] for b in current_line_boxes)
                ],
                "label": current_label
            })

        # Assemble the JSON output in the expected format
        json_output.append({
            "file_name": file_path,
            "page": page_num + 1,
            "height": height,
            "width": width,
            "annotations": [
                {
                    "box": item["bbox"],
                    "text": item["text"],
                    "label": item["label"]
                } for item in page_output
            ]
        })

    return json_output

def main():
    pdf_path = "./T.S-GWR-pages_F_7.pdf"  # Update with your PDF file path
    result = process_pdf(pdf_path)

    if result:
        # Save to JSON file
        with open("output_v1.json", "w") as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
