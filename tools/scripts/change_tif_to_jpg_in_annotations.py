def process_file(input_file):
    # Read all lines from the file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Process each line
    modified_lines = []
    for line in lines:
        # Replace .tif with .tiff
        if line.strip():  # Skip empty lines
            modified_line = line.replace('.jpgf', '.jpg')
            modified_lines.append(modified_line)
    
    # Write back to the file
    with open(input_file, 'w') as f:
        f.writelines(modified_lines)

def main(files):
    for file in files:
        try:
            process_file(file)
            print(f"Successfully processed {file}")
        except FileNotFoundError:
            print(f"Error: {file} not found")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    main(["/media/deephome/maab/datasets/rvl_cdip/training/train.txt",
    "/media/deephome/maab/datasets/rvl_cdip/training/val.txt",
    "/media/deephome/maab/datasets/rvl_cdip/training/inference.txt"])
