import pandas as pd
import numpy as np


def create_wave0_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Wave 0 features: simple sequence-level aggregates.
    Handles ToF -1.0 values by converting to NaN before aggregation.
    """
    print("Starting Wave 0 Feature Engineering...")
    
    # --- Proactive State Management: Validate Input ---
    required_cols = ['sequence_id', 'subject', 'gesture', 'acc_x', 'acc_y', 'acc_z', 
                     'rot_w', 'rot_x', 'rot_y', 'rot_z', 'thm_1', 'thm_2', 
                     'thm_3', 'thm_4', 'thm_5']
    # Check for presence of some ToF columns (assuming they start with 'tof_')
    tof_cols_exist = any(col.startswith('tof_') for col in df.columns)
    if not tof_cols_exist:
        raise ValueError("No ToF columns (starting with 'tof_') found in DataFrame.")
        
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for Wave 0: {missing_cols}")
    
    df_feat = df.copy()
    
    # --- Handle -1.0 in ToF columns ---
    # Identify ToF columns
    tof_columns = [col for col in df_feat.columns if col.startswith('tof_')]
    print(f"  Found {len(tof_columns)} ToF columns. Handling -1.0 values...")
    # Replace -1.0 with NaN in ToF columns before aggregation
    df_feat[tof_columns] = df_feat[tof_columns].replace(-1.0, np.nan)
    
    # --- Define Aggregations ---
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4'] # Excluding thm_5 due to high nulls
    # Note: tof_columns list is already defined above
    
    aggs = {}
    for col in imu_cols + thm_cols + tof_columns:
        aggs[col] = ['mean', 'std', 'min', 'max', 'median', 'skew']
        
    # --- Perform Aggregation ---
    print("  Performing sequence-level aggregation...")
    agg_df = df_feat.groupby('sequence_id').agg(aggs)
    
    # --- Flatten Column Names ---
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns]
    agg_df.reset_index(inplace=True)
    
    # --- Merge with Metadata ---
    print("  Merging with metadata...")
    meta_df = df_feat.groupby('sequence_id')[['subject', 'gesture']].first().reset_index(drop=True)
    # Ensure sequence_id alignment
    final_df = pd.concat([agg_df, meta_df[['subject', 'gesture']]], axis=1)
    # Defragmenting
    #final_df = pre_final_df.copy()
    #del(pre_final_df)
    
    # --- Encode Target ---
    print("  Encoding target variable...")
    # Using .astype('category').cat.codes for robustness if gesture_map not available
    # Or use gesture_map if preferred and available
    final_df['gesture_encoded'] = final_df['gesture'].astype('category').cat.codes 
    # If using gesture_map: final_df['gesture_encoded'] = final_df['gesture'].map(gesture_map)
    
    print(f"Feature engineering complete. Shape of features: {final_df.shape}")
    return final_df