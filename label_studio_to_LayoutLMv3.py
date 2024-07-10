import json

def convert_bounding_box(x, y, width, height):
    """Converts the given bounding box coordinates to the YOLO format.

    Args:
        x: The x-coordinate of the top-left corner of the bounding box.
        y: The y-coordinate of the top-left corner of the bounding box.
        width: The width of the bounding box.
        height: The height of the bounding box.

    Returns:
        A tuple of four coordinates (x1, y1, x2, y2) in the YOLO format.
    """
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height

    return [x1, y1, x2, y2]


####################################### Loading json data ###################################
with open("./Training.json") as f:
    data = json.load(f)

output = []

for annotated_image in data:
    data_entry = {}
    annotation_list = []

    if len(annotated_image) < 8:
        continue

    for k, v in annotated_image.items():
        if k == 'ocr':
            v = v.split('8080/')[-1]
            print(f'filename: {v}')

            data_entry["file_name"] = f"./image/Training_Images/{v}"
            output.append(data_entry)

        if k == 'bbox':
            width = v[0]['original_width']
            height = v[0]['original_height']

            data_entry["height"] = height
            data_entry["width"] = width

    for bb, text, label in zip(annotated_image['bbox'], annotated_image['transcription'], annotated_image['label']):
        annotation_dict = {}

        print('text :', text)

        annotation_dict["box"] = convert_bounding_box(bb['x'], bb['y'], bb['width'], bb['height'])
        annotation_dict["text"] = text

        if 'labels' in label and label['labels']:
            annotation_dict["label"] = label['labels'][-1]
        else:
            annotation_dict["label"] = 'unknown'  # or any default label you prefer

        annotation_list.append(annotation_dict)

    data_entry["annotations"] = annotation_list

print(output)
with open("Training_layoutLMV3.json", "w") as f:
    json.dump(output, f, indent=4)
