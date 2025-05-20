#!/bin/bash

# Execute the Python script
echo "Running the Airbnb Melbourne Price Category Prediction script..."
python /opt/dlami/nvme/autogluon-assistant/output/iteration_0/generated_code.py

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Script executed successfully."
else
    echo "Error: Script execution failed."
    exit 1
fi