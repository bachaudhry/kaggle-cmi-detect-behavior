import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SEED = 42

def create_wave4_features_train(df):
    """
    Creates Wave-4 features: Adds demographic features to Wave 3a-PCA
    feature set.
    """
    df_feat = df.copy()
    
    # Generate Wave3a-PCA feature set
    df_feat['acc_mag'] = np.sqrt(df_feat['acc_x']**2 + df_feat['acc_y']**2 + df_feat['acc_z']**2)
    df_feat['rot_mag'] = np.sqrt(df_feat['rot_w']**2 + df_feat['rot_x']**2 + df_feat['rot_y']**2 + df_feat['rot_z']**2)
    df_feat['jerk'] = df_feat.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    for i in range(1, 5): 
        df_feat[f'thm_grad_{i}_{i+1}'] = df_feat[f'thm_{i}'] - df_feat[f'thm_{i+1}']
    
    # PCA on ToF
    tof_cols = [f'tof_{s}_v{p}' for s in range(1, 6) for p in range(64)]
    tof_data = df_feat[tof_cols].replace(-1, np.nan)
    df_feat['tof_invalid_pct'] = tof_data.isna().mean(axis=1)
    pca = PCA(n_components=10, random_state=SEED)
    tof_pca_features = pca.fit_transform(tof_data.fillna(0))
    for i in range(10): 
        df_feat[f'tof_pca_{i}'] = tof_pca_features[:, i]
    
    # Aggregations
    base_cols_to_agg = [col for col in df.columns if 'acc_' in col or 'rot_' in col or 'thm_' in col]
    derived_cols_to_agg = ['acc_mag', 'rot_mag', 'jerk'] + [f'thm_grad_{i}_{i+1}' for i in range(1, 5)]
    tof_derived_cols_to_agg = ['tof_invalid_pct'] + [f'tof_pca_{i}' for i in range(10)]
    aggs = {}
    for col in base_cols_to_agg + derived_cols_to_agg + tof_derived_cols_to_agg:
        aggs[col] = ['mean', 'std', 'min', 'max', 'skew']

    phase_agg_df = df_feat.groupby(['sequence_id', 'phase']).agg(aggs)
    phase_agg_df.columns = ['_'.join(col).strip() for col in phase_agg_df.columns.values]
    
    phase_agg_df_unstacked = phase_agg_df.unstack(level='phase') # Allow Catboost to handle NaNs as possible phase signals
    phase_agg_df_unstacked.columns = ['_'.join(col).strip() for col in phase_agg_df_unstacked.columns.values]
    
    meta_df = df.groupby('sequence_id').first()
    final_df = pd.concat([meta_df[['subject', 'gesture'] + list(train_demos.columns[1:])], phase_agg_df_unstacked], axis=1).reset_index()
    
    # Create interaction features
    key_sensor_features = [
        'acc_mag_mean_Gesture', 'acc_mag_std_Gesture', 'jerk_mean_Gesture',
        'jerk_std_Gesture', 'tof_pca_0_mean_Gesture', 'tof_invalid_pct_mean_Gesture' # fix for tofl
    ]
    demographic_features = ['age', 'height_cm', 'shoulder_to_wrist_cm'] # For interaction by division and multiplication
    
    for sensor_feat in key_sensor_features:
        for demo_feat in demographic_features:
            if sensor_feat in final_df.columns and demo_feat in final_df.columns:
                # Interaction by division (normalizing sensor reading by demographic)
                final_df[f'{sensor_feat}_div_{demo_feat}'] = final_df[sensor_feat] / (final_df[demo_feat] + 1e-6)
                # Ineraction by multiplication
                final_df[f'{sensor_feat}_mul_{demo_feat}'] = final_df[sensor_feat] * final_df[demo_feat]
    
    #final_df.drop(columns='subject', inplace=True)
    final_df['gesture_encoded'] = final_df['gesture'].map(gesture_map)
    
    print(f"Feature engineering complete. Shape of features: {final_df.shape}")
    return final_df



