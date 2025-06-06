import pandas as pd
import numpy as np

try:
    # Try reading as tabular data
    df = pd.read_csv("/home/ubuntu/.autogluon_assistant/7f8251488f534328bea000a859406dbd/upload_a49aeeb0/train.csv")
    
    # Display column names
    num_cols = len(df.columns)
    if num_cols > 20:
        print("Column names (first 10):")
        print(df.columns[:10].tolist())
        print("...")
        print("Column names (last 10):")
        print(df.columns[-10:].tolist())
    else:
        print("Column names:")
        print(df.columns.tolist())
    
    # Display first few rows with truncated content
    pd.set_option('display.max_colwidth', 50)
    print("\nSample data (first 3 rows):")
    print(df.head(3))
    
    # Basic statistics
    print("\nDataset shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values:")
        print(missing[missing > 0])

except Exception as e:
    # If failed, try reading as text
    try:
        with open("/home/ubuntu/.autogluon_assistant/7f8251488f534328bea000a859406dbd/upload_a49aeeb0/train.csv", 'r') as f:
            content = f.read(1024)
            print("File appears to be text. First 1024 characters:")
            print(content)
    except Exception as e:
        print(f"Could not read file: {e}")