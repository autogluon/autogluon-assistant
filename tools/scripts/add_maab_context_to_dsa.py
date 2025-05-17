import os

# List of dataset names
dataset_names = [
    "abalone", "airbnb_melbourne", "airlines", "bioresponse", "camo_sem_seg",
    "cd18", "covertype", "electricity_hourly", "europeanflooddepth", "fiqabeir",
    "gnad10", "ham10000", "hateful_meme", "isic2017", "kick_starter_funding",
    "memotion", "mldoc", "nn5_daily_without_missing", "petfinder", "road_segmentation",
    "rvl_cdip", "solar_10_minutes", "women_clothing_review", "yolanda"
]

# Text to append
#text_to_append = """
#ONLY save files to the working directory: "./".
#Make predictions on the test data
#Save the predicted results to "./", result file name should be "results", the format and extension should be same as the test data file
#Output column names must exactly match those in the training or sample submission files without adding "predicted_" prefixes or creating any new columns.
#"""
text_to_append = """
Tensorflow is not installed. But you can use pytorch when needed.
"""

# Iterate through each dataset
for dataset in dataset_names:
    # Construct the file path
    file_path = f"/media/agent/DS-Agent/deployment/benchmarks/{dataset}/scripts/research_problem.txt"
    
    try:
        # Check if file exists
        if os.path.exists(file_path):
            # Read the original content
            with open(file_path, 'r') as file:
                original_content = file.read()
                
            # Append the new text to the original content
            with open(file_path, 'w') as file:
                file.write(original_content + text_to_append)
            
            print(f"Successfully updated {file_path}")
        else:
            print(f"File does not exist: {file_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

print("Script execution completed.")
