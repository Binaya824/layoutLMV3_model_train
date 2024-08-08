import fitz  # PyMuPDF
import re

pattern = r'^(?:\d+(\.\d+)*\.)$'

def extract_content_with_positions(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    content_info = []

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        
        # Get the text and image blocks on the page
        blocks = page.get_text("dict")["blocks"]

        # Iterate through each block
        for block in blocks:
            if block["type"] == 1:  # Type 1 indicates an image
                rect = fitz.Rect(block["bbox"])
                image_position = {
                    "type": "image",
                    "page_num": page_num + 1,
                    "bbox": [rect.x0, rect.y0, rect.x1, rect.y1]
                }
                content_info.append(image_position)
                print(f"Found image on page {page_num + 1}: {image_position['bbox']}")
            elif block["type"] == 0:  # Type 0 indicates text
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
                        print(f"Found text on page {page_num + 1}: '{text_position['text']}' {text_position['bbox']}")

    return content_info

def associate_images_with_points(content_positions):
    associations = []
    current_point = None
    print("content position :", content_positions)

    for content in content_positions:
        if content["type"] == "text":
            if re.match(pattern, content["text"].strip()):
                current_point = content
                print("----------------- content", content)
                current_point = content
                print(f"Found numbered point: '{current_point['text']}' on page {current_point['page_num']}")
        elif content["type"] == "image":
            if current_point:
                # Check if the image's top edge is below the current point's bottom edge
                if content["page_num"] == current_point["page_num"]:
                    print(f"Checking image at {content['bbox']} with point '{current_point['text']}' at {current_point['bbox']}")
                    # Check if the image is below the text with a buffer of 10 units
                    if content["bbox"][1] > current_point["bbox"][3] + 10:
                        # Optionally check if the image's bottom edge is not too far below the text
                        if content["bbox"][3] < current_point["bbox"][3] + 200:  # Adjust buffer as needed
                            associations.append({
                                "point_text": current_point["text"],
                                "point_bbox": current_point["bbox"],
                                "image_bbox": content["bbox"]
                            })
                            print(f"Associating image on page {content['page_num']} with point '{current_point['text']}'")
                            current_point = None  # Reset after associating the current image with the point

    return associations

pdf_path = "./pdf with image.pdf"
content_positions = extract_content_with_positions(pdf_path)
associations = associate_images_with_points(content_positions)

for association in associations:
    print(f"Point: {association['point_text']}, Point BBox: {association['point_bbox']}, Image BBox: {association['image_bbox']}")
