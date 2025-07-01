import pandas as pd

def create_wave0_features(df):
    """
    Creates baseline features for Wave 0
    - Simple aggregations on core IMU signals over the entire sequence
    """
    
    # Define sensor columns and aggregations
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    aggregations = ['mean', 'std', 'min', 'max', 'median', 'skew']
    
    # Groupby sequence and flatten multi-level column names
    agg_df = df.groupby('sequence_id')[imu_cols].agg(aggregations)
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    
    # Get static (non-temporal) features for each sequence
    static_cols = ['subject', 'sequence_type', 'gesture', 'orientation']
    static_df = df.groupby('sequence_id')[static_cols].first()
    
    # Merge aggregates with static features
    features_df = pd.merge(static_df, agg_df, on='sequence_id', how='left').reset_index()
    
    return features_df