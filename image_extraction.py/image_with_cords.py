import fitz  # PyMuPDF

def extract_images_and_positions(pdf_path):
    pdf_document = fitz.open(pdf_path)
    
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        image_list = page.get_images(full=True)
        
        print(f"Page {page_number + 1}:")
        if not image_list:
            print("  No images found on this page.")
        else:
            print(f"  Found {len(image_list)} images.")

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Extract bounding box coordinates directly from img tuple
            x0, y0, x1, y1 = img[1:5]
            
            print(f"  Image {img_index + 1}:")
            print(f"    Position: Top-left corner: ({x0}, {y0})")
            print(f"    Dimensions: Width: {x1 - x0}, Height: {y1 - y0}")

            # Optionally, save the image
            image_filename = f"page_{page_number + 1}_img_{img_index + 1}.png"
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)
            print(f"    Saved as: {image_filename}")

    pdf_document.close()

# Usage
extract_images_and_positions("./pdf with image.pdf")
