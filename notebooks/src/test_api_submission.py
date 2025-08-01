# --- 19-test-api-submission.py ---
"""
Script to test the local Kaggle API submission mechanism.
This verifies that we can correctly initialize the API, iterate through sequences,
and submit predictions using our Wave 1 model and feature engineering pipeline.
This is a crucial step before the final submission.
"""

import pandas as pd
import numpy as np
import kaggle_evaluation
import os
import sys
# --- Proactive State Management: Add project root to path for module imports ---
# Assuming this script is run from the project root directory.
# Adjust the path if the script is located elsewhere.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__))) # Gets the directory of this script
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Import our specific feature engineering function ---
try:
    from src.feature_enginnering import create_wave1_features # Note the typo in the filename
    print("Successfully imported create_wave1_features.")
except ImportError as e:
    print(f"Error importing feature engineering function: {e}")
    print("Please check the path and filename in 'notebooks/src/'.")
    raise

# --- Configuration ---
# Adjust these paths to match your local data directory structure
DATA_PATH = './data/'  # Path to the directory containing test files
TEST_DATA_FILE = 'test.csv'
TEST_DEMOGRAPHICS_FILE = 'test_demographics.csv'

# --- Model Path ---
MODEL_PATH = 'notebooks/models_rev/wave1-catboost-best-cbm'

# --- Proactive State Management: Verify Files Exist ---
required_files = [TEST_DATA_FILE, TEST_DEMOGRAPHICS_FILE]
for file in required_files:
    if not os.path.exists(os.path.join(DATA_PATH, file)):
        raise FileNotFoundError(f"Required test file not found: {os.path.join(DATA_PATH, file)}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("=== Starting Local API Submission Test (Wave 1 Model) ===")

try:
    # --- 1. SETUP ---
    print("1. Initializing KaggleEvaluationAPI...")
    # Initialize the competition's API environment
    env = kaggle_evaluation.api.KaggleEvaluationAPI(
        data_path=DATA_PATH,
        test_data_file=TEST_DATA_FILE,
        test_demographics_file=TEST_DEMOGRAPHICS_FILE
    )
    print("   API initialized successfully.")

    # Create the iterator.
    api_iterator = iter(env)
    print("   API iterator created.")

    # --- 2. LOAD MODEL ---
    print("2. Loading trained CatBoost model...")
    import catboost as cat
    # Load the CatBoost model
    model = cat.CatBoostClassifier() # Create an instance
    model.load_model(MODEL_PATH)   # Load the trained weights
    print(f"   Model loaded successfully from {MODEL_PATH}")

    # --- 3. LOAD GLOBAL MAPPINGS ---
    # --- Proactive State Management: Ensure maps are defined ---
    # These are assumed to be available globally, matching the training setup.
    # Example structure (you need to ensure these are correctly defined in your environment):
    # inv_gesture_map = {0: 'Gesture A', 1: 'Gesture B', ...}
    # gesture_map = {'Gesture A': 0, 'Gesture B': 1, ...}
    # gesture_to_seq_type_map = {'Gesture A': 'Target', 'Gesture B': 'Non-Target', ...}

    required_maps = ['inv_gesture_map', 'gesture_map'] # gesture_to_seq_type_map might not be needed for prediction
    for map_name in required_maps:
        if map_name not in globals():
            raise NameError(f"Global variable '{map_name}' is required but not found.")

    print("   Global maps found.")

    # --- Helper Function for Prediction ---
    def predict_single_sequence(model, single_sequence_df: pd.DataFrame, feature_func, inv_map: dict) -> str:
        """
        Performs feature engineering and prediction for a single sequence DataFrame.
        """
        try:
            # --- Feature Engineering ---
            # The feature function should process the single sequence DataFrame
            # It needs to return a DataFrame suitable for the model (single row of features)
            features_df = feature_func(single_sequence_df)

            # --- Prepare Features for Model ---
            # Identify feature columns (exclude metadata like 'subject', 'gesture', 'gesture_encoded')
            # Crucially, exclude 'sequence_id' and 'phase' if they exist in the features_df
            # Wave 1 features should not inherently contain 'sequence_id' after aggregation/unstacking
            # But 'phase' might be present in column names (e.g., 'acc_x_mean_Transition')
            # The model was trained on features without 'sequence_id', 'subject', 'gesture', 'gesture_encoded'
            # We need to match the training feature set.
            # Assuming the output of create_wave1_features is ready, but let's be explicit.
            # Let's assume the training features were derived from features_df_wave1.drop(columns=['subject', 'gesture', 'gesture_encoded'])
            # The output of create_wave1_features should be similar. Let's drop non-feature columns.
            # We need to know the exact columns the model expects. Let's get them from the model.
            expected_feature_names = model.feature_names_ # CatBoost stores feature names used during training
            print(f"     Model expects {len(expected_feature_names)} features.")

            # Filter the generated features to match what the model expects
            # Handle potential mismatch if feature engineering produces extra/different columns
            available_features = set(features_df.columns)
            missing_features = set(expected_feature_names) - available_features
            extra_features = available_features - set(expected_feature_names)

            if missing_features:
                print(f"     WARNING: Missing features for model: {missing_features}")
                # This might lead to an error in predict_proba. Handle appropriately.
                # For now, let it raise an error to catch the mismatch.
                raise ValueError(f"Feature mismatch: Missing features {missing_features}")

            # Select only the features the model expects
            model_input_features = features_df[expected_feature_names]
            print(f"     Features selected for model input. Shape: {model_input_features.shape}")

            # --- Model Prediction ---
            y_pred_proba = model.predict_proba(model_input_features)
            predicted_class_index = np.argmax(y_pred_proba, axis=1)[0] # Get index for first (only) row
            predicted_gesture_string = inv_map[predicted_class_index]
            print(f"     Prediction made: '{predicted_gesture_string}'")
            return predicted_gesture_string

        except Exception as e:
            print(f"     ERROR during feature engineering or prediction: {e}")
            # --- Proactive Debugging: Fallback or Raise ---
            # Raising the error is good for debugging in this test script.
            # In a final submission, a fallback might be safer.
            raise e
            # Example fallback (uncomment if preferred for robustness testing):
            # default_gesture = list(inv_map.values())[0] # Pick first available gesture
            # print(f"     Fallback prediction: '{default_gesture}'")
            # return default_gesture


    # --- 4. THE MAIN SUBMISSION LOOP (Testing) ---
    print("3. Entering API loop (testing first few sequences)...")
    sequence_counter = 0
    MAX_SEQUENCES_TO_TEST = 3 # Limit for quick testing locally

    for (sequence_id, test_sequence_df) in api_iterator:
        sequence_counter += 1
        print(f"\n   --- Processing Sequence {sequence_counter}: {sequence_id} ---")

        # --- Proactive Debugging: Inspect the data served ---
        print(f"       Data shape served by API: {test_sequence_df.shape}")
        # print(f"       Columns: {list(test_sequence_df.columns)}") # Uncomment for detailed inspection

        # --- Proactive Debugging: Check for 'phase' column (should NOT be used for features in submission) ---
        if 'phase' in test_sequence_df.columns:
             print(f"       WARNING: 'phase' column found in test data served by API. Ensure feature engineering does not use it.")

        # --- 5. PIPELINE LOGIC: Feature Engineering + Model Prediction ---
        try:
            predicted_gesture_string = predict_single_sequence(
                model=model,
                single_sequence_df=test_sequence_df,
                feature_func=create_wave1_features, # Use the imported function
                inv_map=inv_gesture_map # Pass the global map
            )
        except Exception as e:
            print(f"   !!! CRITICAL ERROR for {sequence_id}: {e}")
            # Decide whether to stop or continue testing other sequences
            # For debugging, stopping might be better to see the first error clearly.
            raise e # Stop on first error
            # To test robustness, you might continue:
            # continue

        # --- 6. SUBMIT PREDICTION ---
        try:
            print(f"   4. Submitting prediction '{predicted_gesture_string}' for {sequence_id}...")
            env.predict(sequence_id, predicted_gesture_string)
            print(f"       Prediction submitted successfully for {sequence_id}.")
        except Exception as e:
            print(f"   !!! ERROR submitting prediction for {sequence_id}: {e}")
            raise e # Submission errors are critical

        # Limit the number of sequences for quick testing
        if sequence_counter >= MAX_SEQUENCES_TO_TEST:
            print(f"\n   Reached test limit of {MAX_SEQUENCES_TO_TEST} sequences. Stopping loop.")
            break

    print(f"\n5. API loop completed successfully for {sequence_counter} sequences.")

    print("\n=== Local API Submission Test PASSED ===")
    print("The API mechanism, model loading, feature engineering, and prediction pipeline are working correctly for the Wave 1 setup.")
    print("You are ready to proceed with the final submission notebook.")

except FileNotFoundError as e:
    print(f"\n!!! API Test FAILED: File Error !!!")
    print(f"Error: {e}")
    print("Please check your DATA_PATH, MODEL_PATH, and file names.")
except ImportError as e:
    print(f"\n!!! API Test FAILED: Import Error !!!")
    print(f"Error: {e}")
    print("Please check the feature engineering module path and filename.")
except NameError as e:
    print(f"\n!!! API Test FAILED: Variable Error !!!")
    print(f"Error: {e}")
    print("Please ensure global maps (inv_gesture_map, gesture_map) are defined.")
except kaggle_evaluation.api.KaggleEvaluationAPIError as e:
    print(f"\n!!! API Test FAILED: API Error !!!")
    print(f"Error from KaggleEvaluationAPI: {e}")
except Exception as e:
    print(f"\n!!! API Test FAILED: Unexpected Error !!!")
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc() # Print full traceback for debugging
