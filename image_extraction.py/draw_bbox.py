from PIL import Image, ImageDraw

def draw_bounding_box(image_path, bbox, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = bbox

    # Ensure coordinates are valid
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    # Draw the bounding box
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    image.save(output_path)

# Example usage after saving the image
draw_bounding_box("page_1_img_1.png", (0, 2048, 1152, 2048), "boxed_image.png")
