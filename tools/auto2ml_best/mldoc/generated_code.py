"""
Multilingual Document Classification using AutoGluon MultiModal

This script trains a multilingual document classifier using AutoGluon MultiModal
and makes predictions on the inference dataset. The model is trained on documents
in multiple languages (English, Spanish, Italian, French, German) and classifies
them into categories: CCAT, ECAT, GCAT, MCAT.

Requirements:
    - pip install autogluon.multimodal
    - pip install pandas
    - pip install numpy
"""

import os
import warnings
from datetime import datetime

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # Define paths
    base_path = "/media/agent/maab/datasets/mldoc/training"
    output_path = "./"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate a timestamp for the model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(output_path, f"mldoc_model_{timestamp}")
    
    print("Loading training data from multiple languages...")
    
    # List of languages to process
    languages = ['en', 'es', 'it', 'fr', 'de']
    
    # Initialize empty DataFrames for training and validation
    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()
    
    # Load and combine data from all languages
    for lang in languages:
        try:
            # Load training data for this language
            lang_train_path = os.path.join(base_path, lang, "train.csv")
            if os.path.exists(lang_train_path):
                lang_train = pd.read_csv(lang_train_path)
                # Add language column to identify the source language
                lang_train['language'] = lang
                train_data = pd.concat([train_data, lang_train], ignore_index=True)
                print(f"Loaded {len(lang_train)} training samples from {lang}")
            
            # Load validation data for this language
            lang_valid_path = os.path.join(base_path, lang, "validation.csv")
            if os.path.exists(lang_valid_path):
                lang_valid = pd.read_csv(lang_valid_path)
                # Add language column to identify the source language
                lang_valid['language'] = lang
                valid_data = pd.concat([valid_data, lang_valid], ignore_index=True)
                print(f"Loaded {len(lang_valid)} validation samples from {lang}")
        except Exception as e:
            print(f"Error loading data for language {lang}: {e}")
    
    # Check if we have any training data
    if len(train_data) == 0:
        raise ValueError("No training data was loaded. Please check the data paths.")
    
    # Clean the data
    # Remove samples without valid labels (if any)
    train_data = train_data.dropna(subset=['label'])
    if len(valid_data) > 0:
        valid_data = valid_data.dropna(subset=['label'])
    
    # Process the text data - convert byte strings to regular strings if needed
    def process_text(text):
        if isinstance(text, str) and text.startswith("b'") and text.endswith("'"):
            # Remove the b' prefix and ' suffix and handle escape characters
            return text[2:-1].encode('latin1').decode('unicode_escape')
        return text
    
    train_data['text'] = train_data['text'].apply(process_text)
    if len(valid_data) > 0:
        valid_data['text'] = valid_data['text'].apply(process_text)
    
    print(f"Combined training data: {len(train_data)} samples")
    print(f"Combined validation data: {len(valid_data)} samples")
    
    # Load inference data
    inference_path = os.path.join(base_path, "inference.csv")
    inference_data = pd.read_csv(inference_path)
    
    # Process inference text data
    inference_data['text'] = inference_data['text'].apply(process_text)
    
    print(f"Inference data: {len(inference_data)} samples")
    
    # Initialize and train the model
    print("Training the multilingual document classifier...")
    
    # Define hyperparameters
    hyperparameters = {
        "optimization.max_epochs": 10,  # Adjust based on time constraints
        "model.hf_text.checkpoint_name": "microsoft/mdeberta-v3-base",  # Multilingual model
        "env.num_gpus": 1,  # Use 1 GPU
    }
    
    # Initialize the predictor
    predictor = MultiModalPredictor(
        label='label',
        path=model_save_path
    )
    
    # Train the model
    if len(valid_data) > 0:
        # Use validation data if available
        predictor.fit(time_limit=24*3600,
            train_data=train_data,
            tuning_data=valid_data,
            presets="best_quality",
            hyperparameters=hyperparameters
        )
    else:
        # Train without validation data
        predictor.fit(time_limit=24*3600,
            train_data=train_data,
            presets="best_quality",
            hyperparameters=hyperparameters
        )
    
    print(f"Model training completed. Model saved to {model_save_path}")
    
    # Make predictions on inference data
    print("Making predictions on inference data...")
    
    # Since the model was trained with a 'language' column, we need to add it to inference data
    # We'll use a default value since we can't easily detect the language
    # In a real-world scenario, you might want to use a language detection library
    inference_data['language'] = 'en'  # Default to English or use a language detector
    
    # Make predictions
    predictions = predictor.predict(inference_data)
    
    # Create results DataFrame
    results = pd.DataFrame({'text': inference_data['text'], 'label': predictions})
    
    # Save results to the output directory
    results_path = os.path.join(output_path, "results.csv")
    results.to_csv(results_path, index=False)
    
    print(f"Predictions completed. Results saved to {results_path}")
    print("Distribution of predicted labels:")
    print(results['label'].value_counts())