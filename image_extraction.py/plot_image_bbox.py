import fitz  # PyMuPDF
import re
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.patches import Rectangle

pattern = r'^(?:\d+(\.\d+)*\.)$'

def extract_content_with_positions(pdf_path):
    pdf_document = fitz.open(pdf_path)
    content_info = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 1:
                rect = fitz.Rect(block["bbox"])
                image_position = {
                    "type": "image",
                    "page_num": page_num + 1,
                    "bbox": [rect.x0, rect.y0, rect.x1, rect.y1]
                }
                content_info.append(image_position)
            elif block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        rect = fitz.Rect(span["bbox"])
                        text_position = {
                            "type": "text",
                            "page_num": page_num + 1,
                            "text": span["text"],
                            "bbox": [rect.x0, rect.y0, rect.x1, rect.y1]
                        }
                        content_info.append(text_position)

    return content_info

def associate_images_with_points(content_positions):
    associations = []
    current_point = None

    for content in content_positions:
        if content["type"] == "text":
            if re.match(pattern, content["text"].strip()):
                current_point = content
        elif content["type"] == "image":
            if current_point and content["page_num"] == current_point["page_num"]:
                if content["bbox"][1] > current_point["bbox"][3] + 10 and content["bbox"][3] < current_point["bbox"][3] + 200:
                    associations.append({
                        "point_text": current_point["text"],
                        "point_bbox": current_point["bbox"],
                        "image_bbox": content["bbox"]
                    })
                    current_point = None

    return associations

def plot_pdf_with_bounding_boxes(pdf_path, associations, output_path="output_image.png"):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(0)

    # Convert PDF page to an image
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = np.array(img)

    fig, ax = plt.subplots()
    ax.imshow(img)

    # Adjust coordinates for the image scale
    for assoc in associations:
        # Draw point bbox
        point_bbox = assoc['point_bbox']
        rect = Rectangle((point_bbox[0], img.shape[0] - point_bbox[3]), 
                         point_bbox[2] - point_bbox[0], 
                         point_bbox[3] - point_bbox[1], 
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Draw image bbox
        image_bbox = assoc['image_bbox']
        rect = Rectangle((image_bbox[0], img.shape[0] - image_bbox[3]), 
                         image_bbox[2] - image_bbox[0], 
                         image_bbox[3] - image_bbox[1], 
                         linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

pdf_path = "./pdf with image.pdf"
content_positions = extract_content_with_positions(pdf_path)
associations = associate_images_with_points(content_positions)
plot_pdf_with_bounding_boxes(pdf_path, associations, "output_image.png")

for association in associations:
    print(f"Point: {association['point_text']}, Point BBox: {association['point_bbox']}, Image BBox: {association['image_bbox']}")
