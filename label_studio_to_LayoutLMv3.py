import json

def convert_bounding_box(x, y, width, height):
    """Converts the given bounding box coordinates to the YOLO format.

    Args:
    x: The x-coordinate of the top-left corner of the bounding box.
    y: The y-coordinate of the top-left corner of the bounding box.
    width: The width of the bounding box.
    height: The height of the bounding box.

    Returns:
    A list of four coordinates [x1, y1, x2, y2] in the YOLO format.
    """
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height

    return [x1, y1, x2, y2]

# Load JSON data
with open("./Training_json.json") as f:
    data = json.load(f)

output = []

for item in data:  # Assuming data is a list of annotated images
    data_dict = {}
    ann_list = []

    # Extract OCR file name
    ocr_url = item["data"]["ocr"].split('8080/')[-1]
    data_dict["file_name"] = f"./images/Training_Images/{ocr_url}"

    # Initialize image dimensions
    data_dict["height"] = None
    data_dict["width"] = None

    for prediction in item["predictions"]:
        for result in prediction["result"]:
            if result["from_name"] == "bbox":
                bbox = result["value"]
                x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                data_dict["height"] = bbox["height"]  # Assuming a single height value
                data_dict["width"] = bbox["width"]    # Assuming a single width value

                # Convert to YOLO format
                ann_dict = {
                    "box": convert_bounding_box(x, y, width, height),
                    "text": "",
                    "label": ""  # Assuming label can be added here or extracted similarly
                }
                ann_list.append(ann_dict)

            elif result["from_name"] == "transcription":
                text = result["value"]["text"][0]  # Assuming the text is in the first element
                x, y, width, height = result["value"]["x"], result["value"]["y"], result["value"]["width"], result["value"]["height"]
                
                # Update the last annotation with text
                if ann_list:
                    ann_list[-1]["text"] = text

    data_dict["annotations"] = ann_list
    output.append(data_dict)

# Verify the output data
print(json.dumps(output, indent=4))

# Save the output to a new JSON file
with open("Training_layoutLMV3.json", "w") as f:
    json.dump(output, f, indent=4)
