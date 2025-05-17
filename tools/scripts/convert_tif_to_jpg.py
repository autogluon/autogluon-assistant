from PIL import Image
import os

def convert_tiff_to_jpg(root_dir):
    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if file is a TIFF
            if filename.lower().endswith(('.tiff', '.tif')):
                tiff_path = os.path.join(dirpath, filename)
                # Create JPG filename by replacing extension
                jpg_path = os.path.splitext(tiff_path)[0] + '.jpg'
                
                try:
                    # Open and convert the image
                    with Image.open(tiff_path) as img:
                        # Convert to RGB if necessary (in case of RGBA TIFF)
                        if img.mode in ('RGBA', 'LA'):
                            img = img.convert('RGB')
                        # Save as JPG
                        img.save(jpg_path, 'JPEG', quality=95)
                    print(f"Converted: {tiff_path} -> {jpg_path}")
                except Exception as e:
                    print(f"Error converting {tiff_path}: {str(e)}")

# Usage example
if __name__ == "__main__":
    directory = "/media/deephome/maab/datasets/rvl_cdip"  # Replace with your directory path
    convert_tiff_to_jpg(directory)
