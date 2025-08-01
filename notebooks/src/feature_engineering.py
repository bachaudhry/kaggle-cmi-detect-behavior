import pandas as pd
import numpy as np


def create_wave1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Wave 1 features: phase-specific statistical aggregates.
    Correctly uses 'sequence_counter' as an ordering index for calculations if needed.
    Handles ToF -1.0 values by converting to NaN before aggregation.
    """
    print("Starting Wave 1 Feature Engineering...")
    # State Management: Validate Inputs
    required_cols = ['sequence_id', 'phase', 'subject', 'gesture', 'acc_x', 'acc_y', 'acc_z', 
                     'rot_w', 'rot_x', 'rot_y', 'rot_z', 'thm_1', 'thm_2', 
                     'thm_3', 'thm_4', 'thm_5', 'sequence_counter'] # sequence_counter is key for Wave 1
    tof_cols_exist = any(col.startswith('tof_') for col in df.columns)
    if not tof_cols_exist:
        raise ValueError("No ToF columns (starting with 'tof_') found in DataFrame.")
        
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for Wave 1: {missing_cols}")
    
    df_feat = df.copy()
    
    # Handle -1.0 in ToF columns
    tof_columns = [col for col in df_feat.columns if col.startswith('tof_')]
    print(f"  Found {len(tof_columns)} ToF columns. Handling -1.0 values...")
    df_feat[tof_columns] = df_feat[tof_columns].replace(-1.0, np.nan)
    
    # --- Define Aggregations ---
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4'] # Excluding thm_5
    # Note: tof_columns list is already defined above
    
    aggs = {}
    for col in imu_cols + thm_cols + tof_columns:
        aggs[col] = ['mean', 'std', 'min', 'max', 'median', 'skew']
        
    # ----- Addition to Wave 0 ------ Perform phase specifc aggregation
    phase_agg_df = df_feat.groupby(['sequence_id', 'phase']).agg(aggs)
    
    # Flatten column names
    phase_agg_df.columns = ['_'.join(col).strip() for col in phase_agg_df.columns]
    phase_agg_df.reset_index(inplace=True)
    
    # Unstack phase dimension
    phase_agg_df.set_index(['sequence_id', 'phase'], inplace=True)
    phase_agg_df_unstacked = phase_agg_df.unstack(level='phase')
    
    # Flatten multi-index columns and create corresponding column names
    new_cols = [f"{sensor_stat}_{phase}" for sensor_stat, phase in phase_agg_df_unstacked.columns]
    phase_agg_df_unstacked.columns = new_cols
    phase_agg_df_unstacked.reset_index(inplace=True)
    
    # Merge with Metadata
    print("  Merging with metadata...")
    meta_df = df_feat.groupby('sequence_id')[['subject', 'gesture']].first().reset_index(drop=True)
    # Ensure sequence_id alignment (phase_agg_df_unstacked should have sequence_id as a column now)
    final_df = pd.concat([phase_agg_df_unstacked, meta_df[['subject', 'gesture']]], axis=1)
    
    # --- Encode Target ---
    print("  Encoding target variable...")
    final_df['gesture_encoded'] = final_df['gesture'].astype('category').cat.codes
    # If using gesture_map: final_df['gesture_encoded'] = final_df['gesture'].map(gesture_map)
    
    print(f"Feature engineering complete. Shape of features: {final_df.shape}")
    return final_df