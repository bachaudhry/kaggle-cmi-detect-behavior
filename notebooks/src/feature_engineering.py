import pandas as pd
import numpy as np
from sklearn.decomposition import PCA



def create_wave2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Wave 2 features: Adds advanced IMU (magnitude, jerk) and
    Thermopile (gradients) features before performing phase-specific aggregation
    (Wave 1 style).

    Ensures direct continuity with the provided Wave 1 feature engineering logic.
    Correctly uses 'sequence_counter' as an ordering index for calculations like jerk.
    Handles ToF -1.0 values by converting to NaN before aggregation.
    Explicitly manages grouping columns to prevent aggregation errors.
    Aligns with Wave 1's exclusion of thm_0 and tof_0.

    Args:
        df (pd.DataFrame): The raw training dataframe. Must be the SAME as used for Wave 1.

    Returns:
        pd.DataFrame: A dataframe with engineered Wave 2 features, aggregated by phase.
                      Structured identically to Wave 1 output for seamless continuation.
    """
    print("Starting Wave 2 Feature Engineering (Aligned with Wave 1)...")

    # --- State Management: Validate Inputs (Mirroring Wave 1) ---
    # Note: thm_0 and tof_0 are intentionally excluded, matching Wave 1 logic.
    required_cols = ['sequence_id', 'phase', 'subject', 'gesture', 'acc_x', 'acc_y', 'acc_z',
                     'rot_w', 'rot_x', 'rot_y', 'rot_z', 'thm_1', 'thm_2',
                     'thm_3', 'thm_4', 'thm_5', 'sequence_counter']

    # Check for presence of some ToF columns (excluding tof_0)
    tof_cols_exist = any(col.startswith('tof_') and col != 'tof_0' for col in df.columns)
    if not tof_cols_exist:
        print(" Warning: No ToF columns (other than potentially missing 'tof_0') found in DataFrame.")

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for Wave 2 (based on Wave 1 structure): {missing_cols}")

    df_feat = df.copy()
    print(f" Input DataFrame shape: {df_feat.shape}")

    # --- 1. Add Advanced IMU Features (Aligned with Wave 1 base structure) ---
    print(" Calculating advanced IMU features...")
    df_feat['acc_mag'] = np.sqrt(df_feat['acc_x']**2 + df_feat['acc_y']**2 + df_feat['acc_z']**2)
    df_feat['rot_mag'] = np.sqrt(df_feat['rot_w']**2 + df_feat['rot_x']**2 + df_feat['rot_y']**2 + df_feat['rot_z']**2)
    print("  - Calculated acc_mag and rot_mag.")

    # Jerk (Derivative of Acceleration Magnitude)
    print(" Calculating acc_mag_jerk using sequence_counter for ordering...")
    df_feat.sort_values(by=['sequence_id', 'sequence_counter'], inplace=True)
    df_feat['acc_mag_jerk'] = df_feat.groupby('sequence_id')['acc_mag'].diff()
    # Note: fillna(0) was used in the original Gemini chat for jerk. We can keep or remove based on strategy.
    # df_feat['acc_mag_jerk'].fillna(0, inplace=True) # Optional, matching original chat
    print("  - Calculated acc_mag_jerk.")

    # --- 2. Add Thermopile Gradient Features (Excluding thm_0, matching Wave 1) ---
    thm_cols_for_grad = ['thm_1', 'thm_2', 'thm_3', 'thm_4'] # Excluding thm_0, thm_5
    print(f" Calculating Thermopile gradients for columns: {thm_cols_for_grad}...")
    for i in range(len(thm_cols_for_grad) - 1):
        col1 = thm_cols_for_grad[i]
        col2 = thm_cols_for_grad[i+1]
        grad_col_name = f"thm_grad_{col1.split('_')[1]}_{col2.split('_')[1]}"
        df_feat[grad_col_name] = df_feat[col2] - df_feat[col1]
    print("  - Calculated Thermopile gradients.")

    # --- 3. Handle -1.0 in ToF columns (if they exist, excluding tof_0) ---
    # Find ToF columns, excluding 'tof_0' as it doesn't exist / is excluded
    tof_columns = [col for col in df_feat.columns if col.startswith('tof_') and col != 'tof_0']
    if tof_columns:
        print(f" Found {len(tof_columns)} ToF columns (excluding 'tof_0'). Handling -1.0 values...")
        df_feat[tof_columns] = df_feat[tof_columns].replace(-1.0, np.nan)
        print(f"  - Replaced -1.0 with NaN in ToF columns.")
    else:
        print(" No ToF columns found to process (excluding 'tof_0').")

    # --- 4. Perform Phase-Specific Aggregation (Directly Mirroring Wave 1) ---
    print(" Performing phase-specific aggregation (Wave 1 style)...")

    # --- Define Aggregations (Including NEW Wave 2 features) ---
    # Base sensor columns (matching Wave 1, excluding thm_0, tof_0)
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4'] # Excluding thm_0, thm_5 (matching Wave 1 logic for thm_5)
    # Note: tof_columns list is already defined above, excluding tof_0

    # NEW Wave 2 derived columns to aggregate
    derived_wave2_cols = ['acc_mag', 'rot_mag', 'acc_mag_jerk'] + \
                         [f"thm_grad_{col1.split('_')[1]}_{col2.split('_')[1]}"
                          for i, col1 in enumerate(thm_cols_for_grad[:-1])
                          for col2 in thm_cols_for_grad[i+1:i+2]]

    # Combine all columns for aggregation
    cols_to_aggregate = imu_cols + thm_cols + tof_columns + derived_wave2_cols
    print(f"  Columns identified for aggregation: {len(cols_to_aggregate)}")

    # --- Proactive Debugging: Check for non-numeric columns before aggregation ---
    non_numeric_cols = df_feat[cols_to_aggregate].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise TypeError(f"Non-numeric columns found for aggregation: {non_numeric_cols}. "
                        f"Check data types or exclude these columns. ")

    # Define aggregation functions (matching Wave 1)
    aggregations = ['mean', 'std', 'min', 'max', 'median', 'skew']

    # Create aggregation dictionary
    aggs = {}
    for col in cols_to_aggregate:
        aggs[col] = aggregations

    # Perform the groupby and aggregation (matching Wave 1 structure)
    try:
        print("  - Grouping by ['sequence_id', 'phase'] and aggregating...")
        phase_agg_df = df_feat.groupby(['sequence_id', 'phase']).agg(aggs)
        print(f"  - Aggregation completed. Shape before flattening: {phase_agg_df.shape}")

        # Flatten column names (matching Wave 1)
        print("  - Flattening MultiIndex columns...")
        phase_agg_df.columns = ['_'.join(col).strip() for col in phase_agg_df.columns]
        phase_agg_df.reset_index(inplace=True)
        print(f"  - Columns flattened.")

        # Unstack phase dimension (matching Wave 1)
        print("  - Unstacking 'phase' level...")
        phase_agg_df.set_index(['sequence_id', 'phase'], inplace=True)
        phase_agg_df_unstacked = phase_agg_df.unstack(level='phase')
        print(f"  - Unstacking completed. Shape: {phase_agg_df_unstacked.shape}")

        # Flatten multi-index columns and create corresponding column names (matching Wave 1)
        print("  - Flattening final MultiIndex columns...")
        # Use the exact format from Wave 1: f"{sensor_stat}_{phase}"
        new_cols = [f"{sensor_stat}_{phase}" for sensor_stat, phase in phase_agg_df_unstacked.columns]
        phase_agg_df_unstacked.columns = new_cols
        phase_agg_df_unstacked.reset_index(inplace=True) # sequence_id becomes a column
        print(f"  - Final columns flattened. New column count: {len(phase_agg_df_unstacked.columns)}")

    except KeyError as e:
        raise KeyError(f"Aggregation failed due to missing key/column: {e}. "
                       f"Check if 'phase' or feature columns exist correctly.") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during aggregation: {e}") from e

    # --- 5. Merge with Metadata (Directly Mirroring Wave 1) ---
    print(" Merging with metadata (subject, gesture)...")
    meta_df = df_feat.groupby('sequence_id')[['subject', 'gesture']].first() 
    final_df = pd.merge(phase_agg_df_unstacked, meta_df, on='sequence_id', how='left')
    print(f"  - Merge completed. Shape: {final_df.shape}")

    # --- Encode Target (Matching Wave 1) ---
    print(" Encoding target variable...")
    final_df['gesture_encoded'] = final_df['gesture'].astype('category').cat.codes
    # If a global gesture_map is preferred: final_df['gesture_encoded'] = final_df['gesture'].map(gesture_map)
    print(f"  - Target encoded.")

    print(f"Wave 2 Feature Engineering completed. Shape of features: {final_df.shape}")
    # Ensure column order consistency if needed (e.g., metadata last)
    # meta_cols = ['subject', 'gesture', 'gesture_encoded']
    # other_cols = [c for c in final_df.columns if c not in meta_cols]
    # final_df = final_df[other_cols + meta_cols]

    return final_df


def create_wave3_features(df: pd.DataFrame, n_components: int = 50, random_state: int = 42) -> pd.DataFrame:
    """
    Creates Wave 3 features: Applies PCA directly to the raw ToF sensor data,
    generates PCA components for each raw reading, then incorporates these
    components into the standard phase-specific aggregation framework (Wave 1 style).

    This function correctly handles the full pipeline:
    1.  Applies PCA to the raw ToF data (all readings, all sequences, all phases).
    2.  Adds the resulting PCA components back to the raw dataframe.
    3.  Performs standard phase-specific aggregation (Wave 1 style) on the augmented data.
    4.  Unstacks phase and merges metadata.

    Correctly uses 'sequence_counter' implicitly via the aggregation logic.
    Handles ToF -1.0 values by converting to NaN before PCA.
    Ensures continuity with Waves 1 and 2 by re-using their core aggregation logic.
    Notes that tof_0 does not exist in the training dataset.

    Args:
        df (pd.DataFrame): The raw training dataframe containing sensor data,
                           sequence_id, phase, subject, gesture, and sequence_counter.
                           This MUST be the same raw df used for Waves 1 & 2.
        n_components (int, optional): Number of PCA components to generate.
                                      Defaults to 50.
        random_state (int, optional): Random state for PCA. Defaults to 42.

    Returns:
        pd.DataFrame: A dataframe with features aggregated by phase and unstacked,
                      ready for model training. Includes ToF PCA components.
    """
    print("Starting Wave 3 Feature Engineering (ToF PCA - Raw Data Approach)...")

    # --- Proactive State Management: Validate Input ---
    required_base_cols = [
        'sequence_id', 'phase', 'subject', 'gesture', 'sequence_counter'
    ]
    # Check for presence of other ToF columns (assuming they start with 'tof_' and skip tof_0)
    tof_cols_exist = any(col.startswith('tof_') and col != 'tof_0' for col in df.columns)

    if not tof_cols_exist:
         raise ValueError("No ToF columns (other than potentially missing 'tof_0') found in input DataFrame. "
                          "PCA on ToF cannot proceed.")

    missing_cols = [col for col in required_base_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required core columns for Wave 3: {missing_cols}")

    df_raw = df.copy()
    print(f" Raw input DataFrame shape: {df_raw.shape}")

    # --- 1. Prepare Raw ToF Data for PCA ---
    print(" Preparing raw ToF data for PCA...")
    # Find ToF columns in the raw data, excluding 'tof_0'
    tof_columns = [col for col in df_raw.columns if col.startswith('tof_') and col != 'tof_0']
    if not tof_columns:
        raise ValueError("No ToF columns found for PCA after excluding 'tof_0'.")

    print(f"  - Found {len(tof_columns)} ToF columns for PCA.")

    # Handle -1.0 in ToF columns (in the raw data for PCA)
    print("  - Handling -1.0 values in ToF data...")
    df_tof_raw_for_pca = df_raw[tof_columns].copy()
    initial_nans_before = df_tof_raw_for_pca.isna().sum().sum()
    df_tof_raw_for_pca = df_tof_raw_for_pca.replace(-1.0, np.nan)
    nans_introduced = df_tof_raw_for_pca.isna().sum().sum() - initial_nans_before
    print(f"  - Replaced {nans_introduced} instances of -1.0 with NaN in ToF data for PCA.")

    # --- 2. Apply PCA to Raw ToF Data ---
    print(f" Applying PCA to raw ToF data (n_components={n_components})...")
    
    # --- Proactive Debugging: Check for non-numeric data ---
    non_numeric_tof_cols = df_tof_raw_for_pca.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_tof_cols:
         raise TypeError(f"Non-numeric ToF columns found for PCA: {non_numeric_tof_cols}. ")

    # --- CRITICAL FIX: Impute NaN values before PCA ---
    nan_count_before_impute = df_tof_raw_for_pca.isna().sum().sum()
    if nan_count_before_impute > 0:
        print(f"  - Found {nan_count_before_impute} NaN values in raw ToF data. Imputing with 0...")
        df_tof_raw_imputed = df_tof_raw_for_pca.fillna(0)
    else:
        print(f"  - No NaN values found in raw ToF data.")
        df_tof_raw_imputed = df_tof_raw_for_pca

    # Fit PCA on the imputed raw data
    pca = PCA(n_components=n_components, random_state=random_state)
    try:
        X_pca = pca.fit_transform(df_tof_raw_imputed)
        print(f"  - PCA completed. Transformed data shape: {X_pca.shape}")
    except ValueError as e:
        raise RuntimeError(f"PCA failed unexpectedly on raw ToF data after imputation: {e}") from e

    # --- 3. Add PCA Components Back to Raw Data ---
    print(" Adding PCA components to raw dataframe...")
    # Create column names for PCA features
    pca_columns = [f"tof_pca_{i+1}" for i in range(n_components)]
    
    # Create a DataFrame for PCA results with the same index as the raw data
    df_pca_components = pd.DataFrame(X_pca, columns=pca_columns, index=df_raw.index)
    
    # Concatenate the PCA components with the original raw dataframe
    df_with_pca = pd.concat([df_raw, df_pca_components], axis=1)
    print(f"  - Raw dataframe with PCA components shape: {df_with_pca.shape}")

    # --- 4. Perform Standard Phase-Specific Aggregation (Wave 1/2 Style) ---
    # Wave 2 will aggregate the PCA components along with all other features.
    print(" Performing standard phase-specific aggregation using Wave 2 logic (includes PCA features)...")
    try:
        # This re-uses the exact logic of create_wave2_features on the augmented data.
        # It will aggregate the new 'tof_pca_*' columns.
        final_features_df = create_wave2_features(df_with_pca)
        print(f"  - Aggregation completed using Wave 2 logic. Final shape: {final_features_df.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to perform aggregation on data with PCA features: {e}") from e

    print("Wave 3 Feature Engineering (ToF PCA - Raw Data Approach) completed successfully.")
    return final_features_df



def create_wave4_features(df: pd.DataFrame, df_demographics: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Wave 4 features: Generates interaction and normalized features
    based on the aggregated sensor statistics from Wave 3 and demographic data.

    Ensures continuity with Waves 1, 2, and 3.
    Directly builds upon the output of create_wave3_features.
    Correctly merges demographics on 'subject'.
    Adds interaction features (product, division) between key *aggregated* sensor stats and demographics.
    Adds normalized features (e.g., sensor stat / height).
    Avoids redundant OHE/processing for binary demographics.
    Explicitly manages columns to prevent errors.

    Args:
        df (pd.DataFrame): The raw training dataframe containing sensor data,
                           sequence_id, phase, subject, gesture, sequence_counter.
                           This MUST be the same raw df used for Waves 1, 2 & 3.
        df_demographics (pd.DataFrame): The demographics dataframe with columns:
                           subject(subject_id), adult_child, age, sex, handedness,
                           height_cm, shoulder_to_wrist_cm, elbow_to_wrist_cm.

    Returns:
        pd.DataFrame: A dataframe with Wave 3 features plus new interaction/normalized
                      features. Indexed by 'sequence_id'. Ready for model training.
    """
    print("Starting Wave 4 Feature Engineering (Interaction & Demographics on Aggregated Features)...")

    # --- Proactive State Management: Validate Inputs ---
    required_sensor_cols = [
        'sequence_id', 'phase', 'subject', 'gesture', 'sequence_counter'
    ]
    required_demo_cols = [
        'subject', 'adult_child', 'age', 'sex', 'handedness',
        'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'
    ]

    missing_sensor_cols = [col for col in required_sensor_cols if col not in df.columns]
    if missing_sensor_cols:
        raise ValueError(f"Missing required sensor columns for Wave 4: {missing_sensor_cols}")

    missing_demo_cols = [col for col in required_demo_cols if col not in df_demographics.columns]
    if missing_demo_cols:
        raise ValueError(f"Missing required demographic columns for Wave 4: {missing_demo_cols}")

    df_raw = df.copy()
    df_demo = df_demographics.copy()
    print(f" Raw sensor DataFrame shape: {df_raw.shape}")
    print(f" Demographics DataFrame shape: {df_demo.shape}")
    print(f" Unique subjects in sensor data: {df_raw['subject'].nunique()}")
    print(f" Unique subjects in demographics data: {df_demo['subject'].nunique()}")

    # --- 1. Generate Wave 3 Features (Baseline with ToF PCA) ---
    print(" Generating Wave 3 features (includes Waves 1, 2)...")
    wave3_features_df = create_wave3_features(df_raw, n_components=50)
    print(f" Wave 3 features generated. Shape: {wave3_features_df.shape}")

    # --- 2. Prepare Demographic Data ---
    print(" Preparing demographic data...")
    df_demo_selected = df_demo[required_demo_cols].copy()
    print(f"  - Selected demographic columns: {list(df_demo_selected.columns)}")

    # --- 3. Merge Wave 3 Features with Demographics ---
    print(" Merging Wave 3 features with demographics...")
    # wave3_features_df index is 'sequence_id'
    # df_demo_selected has 'subject' column. We need to merge based on the 'subject' associated with each 'sequence_id'.
    
    # --- Proactive Debugging: Check if 'subject' is in wave3_features_df ---
    if 'subject' not in wave3_features_df.columns:
         raise KeyError("Column 'subject' not found in wave3_features_df. "
                        "Ensure create_wave3_features correctly includes metadata.")

    # Perform the merge on the 'subject' column
    df_wave3_with_demo = pd.merge(wave3_features_df, df_demo_selected, on='subject', how='left')
    
    # --- Proactive Debugging: Check for merge issues ---
    if df_wave3_with_demo.shape[0] != wave3_features_df.shape[0]:
        print(f" Warning: Merge resulted in {df_wave3_with_demo.shape[0]} rows, expected {wave3_features_df.shape[0]}. "
              f"Check for duplicate 'subject' entries in df_demographics or missing subjects.")
    if df_wave3_with_demo[required_demo_cols[1:]].isnull().any().any(): # Check demo cols (excluding 'subject')
         print(" Warning: NaN values found in merged demographic data. Check for subjects in sensor data "
               "not present in demographics.")

    print(f"  - Merged DataFrame shape: {df_wave3_with_demo.shape}")

    # --- 4. Add Interaction and Normalized Features ---
    print(" Adding interaction and normalized features...")
    
    # Define demographic features to interact with
    demo_features_continuous = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    demo_features_binary = ['adult_child', 'sex', 'handedness'] 
    
    # Focus on mean and std for now
    key_sensor_features = wave3_features_df.columns[(wave3_features_df.columns.str.contains("std") | (wave3_features_df.columns.str.contains("mean"))) & 
                                                (wave3_features_df.columns.str.contains("acc_mag") | wave3_features_df.columns.str.contains("rot_mag") |
                                                wave3_features_df.columns.str.contains("thm_grad_") | wave3_features_df.columns.str.contains("tof_pca"))].tolist()
                    
    # Limit to a reasonable number to start, or use all found if small
    key_sensor_features = list(set(key_sensor_features))
    print(f"  - Identified {len(key_sensor_features)} key aggregated sensor features for interaction.")
    # print(f"    Example features: {key_agg_sensor_features[:5]}") # Uncomment for debugging

    new_features_created = 0
    features_to_add = {} # Dictionary to hold new feature data before adding to DataFrame

    # --- Interaction by Division (Normalization) ---
    print("  - Creating interaction features by division (normalization)...")
    for sensor_feat in key_sensor_features:
        for demo_feat in demo_features_continuous:
            if demo_feat in df_wave3_with_demo.columns: # Check if demographic feature exists
                new_col_name_div = f"{sensor_feat}_div_{demo_feat}"
                # Use .loc to avoid potential SettingWithCopyWarning and for clarity
                features_to_add[new_col_name_div] = (
                    df_wave3_with_demo[sensor_feat] / (df_wave3_with_demo[demo_feat] + 1e-6)
                )
                new_features_created += 1

    # --- Interaction by Multiplication ---
    print("  - Creating interaction features by multiplication...")
    for sensor_feat in key_sensor_features:
        for demo_feat in demo_features_binary:
            if demo_feat in df_wave3_with_demo.columns: # Check if demographic feature exists
                new_col_name_mul = f"{sensor_feat}_mul_{demo_feat}"
                features_to_add[new_col_name_mul] = (
                    df_wave3_with_demo[sensor_feat] * df_wave3_with_demo[demo_feat]
                )
                new_features_created += 1
                
    print(f"  - Created {new_features_created} aggregate-level interaction/normalized features.")

    # --- 5. Add New Features to DataFrame Efficiently ---
    print("  - Adding new features to the DataFrame...")
    if features_to_add:
        new_features_df = pd.DataFrame(features_to_add, index=df_wave3_with_demo.index)
        final_features_df = pd.concat([df_wave3_with_demo, new_features_df], axis=1)
    else:
        print("  - No new features were created. Returning Wave 3 features.")
        final_features_df = df_wave3_with_demo

    # Drop the raw demographic columns if they are not desired as direct features
    final_features_df.drop(columns=required_demo_cols[1:], inplace=True, errors='ignore') # Exclude 'subject'
    
    if 'sequence_id' in final_features_df.columns:
        final_features_df.set_index('sequence_id', inplace=True)

    print(f" Final Wave 4 DataFrame shape: {final_features_df.shape}")
    print("Wave 4 Feature Engineering (Interaction & Demographics on Aggregated Features) completed successfully.")
    return final_features_df