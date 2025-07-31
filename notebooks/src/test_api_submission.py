"""
Script to test the local Kaggle API submission mechanism.
This verifies that we can correctly initialize the API, iterate through sequences,
and submit predictions (even dummy ones) without errors.
This is a crucial step before integrating our trained model and feature pipeline.
"""

import pandas as pd
import numpy as np
import data.kaggle_evaluation as kaggle_evaluation
import os

# --- Configuration ---
# Adjust these paths to match your local data directory structure
PROJECT_PATH = '~/code/kaggle/kaggle-cmi-detect-behavior/'
DATA_PATH = os.path.join(PROJECT_PATH, 'notebooks/data')
TEST_DATA_FILE = 'test.csv'
TEST_DEMOGRAPHICS_FILE = 'test_demographics.csv'

# --- Proactive State Management: Verify Files Exist ---
required_files = [TEST_DATA_FILE, TEST_DEMOGRAPHICS_FILE]
for file in required_files:
    if not os.path.exists(os.path.join(DATA_PATH, file)):
        raise FileNotFoundError(f"Required test file not found: {os.path.join(DATA_PATH, file)}")
    

print("=== Starting Local API Submission Test ===")


try:
    # --- 1. SETUP ---
    print("1. Initializing KaggleEvaluationAPI...")
    # Initialize the competition's API environment
    # This will load test.csv and test_demographics.csv and merge them
    env = kaggle_evaluation.api.KaggleEvaluationAPI(
        data_path=DATA_PATH,
        test_data_file=TEST_DATA_FILE,
        test_demographics_file=TEST_DEMOGRAPHICS_FILE
    )
    print("   API initialized successfully.")

    
    # Create the iterator. This is the object we will loop over.
    api_iterator = iter(env)
    print("   API iterator created.") 
    
    # --- Get a sample sequence ID for placeholder prediction ---
    # We need a valid gesture name. Let's assume one exists in the training set.
    # In a real scenario, you'd load your inv_gesture_map.
    # For testing, we'll use a placeholder. Ensure this matches one of the 18 classes.
    # Based on prior context, let's assume 'Cheek - pinch skin' is a valid target gesture.
    PLACEHOLDER_PREDICTION = 'Cheek - pinch skin' # <--- ADJUST IF NEEDED BASED ON YOUR CLASSES
    print(f"   Using placeholder prediction: '{PLACEHOLDER_PREDICTION}'")
    
    
    sequence_counter = 0
    MAX_SEQUENCES_TO_TEST = 5 # Limit for testing to avoid long runtimes
    
    # --- 2. THE MAIN SUBMISSION LOOP (Testing) ---
    print(f"2. Entering API loop (testing first {MAX_SEQUENCES_TO_TEST} sequences)...")
    for (sequence_id, test_sequence_df) in api_iterator:
        sequence_counter += 1
        print(f"   --- Processing Sequence {sequence_counter}: {sequence_id} ---")

        # --- Proactive Debugging: Inspect the data served ---
        print(f"       Data shape: {test_sequence_df.shape}")
        print(f"       Columns present: {list(test_sequence_df.columns)}")
        # Check if demographic columns are merged (e.g., 'age_group')
        demo_cols = ['age_group', 'gender', 'handedness', 'orientation']
        present_demo_cols = [col for col in demo_cols if col in test_sequence_df.columns]
        if present_demo_cols:
            print(f"       Merged demographic columns found: {present_demo_cols}")
        else:
            print("       WARNING: No expected demographic columns found in served data.")

        # --- Proactive Debugging: Check for 'phase' column (should NOT be present) ---
        if 'phase' in test_sequence_df.columns:
             print(f"       WARNING: 'phase' column found in test data. This is a train-only column and must NOT be used!")

        # --- 3. YOUR PIPELINE LOGIC (Placeholder/Dummy) ---
        # In a real test, this is where you'd call your feature engineering function
        # and your model's predict method.
        # For this test, we skip complex logic and use a fixed placeholder prediction.
        print(f"       Generating dummy prediction: '{PLACEHOLDER_PREDICTION}'")

        # --- 4. SUBMIT PREDICTION ---
        # Pass the prediction back to the API. The loop will not advance otherwise.
        print(f"       Submitting prediction for {sequence_id}...")
        env.predict(sequence_id, PLACEHOLDER_PREDICTION)
        print(f"       Prediction submitted for {sequence_id}.")

        # Limit the number of sequences for quick testing
        if sequence_counter >= MAX_SEQUENCES_TO_TEST:
            print(f"   Reached test limit of {MAX_SEQUENCES_TO_TEST} sequences. Stopping loop.")
            break

    print(f"3. API loop completed successfully for {sequence_counter} sequences.")

    # If the loop completes without error, the basic API mechanism works.
    print("\n=== Local API Submission Test PASSED ===")
    print("The API initialization, iteration, and prediction submission mechanisms are working correctly.")
    print("You can now proceed to integrate your trained model and feature engineering pipeline.")

except FileNotFoundError as e:
    print(f"\n!!! API Test FAILED: File Error !!!")
    print(f"Error: {e}")
    print("Please check your DATA_PATH and file names.")
except kaggle_evaluation.api.KaggleEvaluationAPIError as e: # Assuming a specific error class exists
    print(f"\n!!! API Test FAILED: API Error !!!")
    print(f"Error from KaggleEvaluationAPI: {e}")
except Exception as e:
    print(f"\n!!! API Test FAILED: Unexpected Error !!!")
    print(f"An unexpected error occurred: {e}")
    print("Traceback details would be helpful for debugging.")

