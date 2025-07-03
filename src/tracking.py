import pandas as pd
import os
from datetime import datetime
import wandb
import json

class ExperimentTracker:
    def __init__(self, project_path, use_wandb=False, wandb_project_name=None, wandb_entity=None):
        """
        Initialize the tracker
        
        Args:
            project_path (str): The root path of the project.
            use_wandb (bool): If True, logs to Weights & Biases.
            wandb_project_name (str): The W&B project name.
            wandb_entity (str): Your W&B username or entity.
        """
        self.log_file_path = os.path.join(project_path, 'experiment_log.csv')
        self.use_wandb = use_wandb
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity
        
        # Initialize the log if it doesn't exist
        if not os.path.exists(self.log_file_path):
            df = pd.DataFrame(columns=[
                'timestamp', 'experiment_name', 'model_name',
                'feature_wave', 'cv_score', 'params', 'notes'
            ])
            df.to_csv(self.log_file_path, index=False)
            
    def log_experiment(self, experiment_name, model_name, feature_wave, cv_score, params, notes=""):
        """
        Logs details of a single experiment run to a local CSV and/or W&B
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_data = {
            'timestamp': timestamp,
            'experiment_name': experiment_name,
            'model_name': model_name,
            'feature_wave': feature_wave,
            'cv_score': cv_score,
            'params': str(params),
            'notes': notes
        }
        # Log to local CSV file
        log_df = pd.DataFrame([log_data])
        log_df.to_csv(self.log_file_path, mode='a', header=False, index=False)
        print(f"Experiment '{experiment_name}' logged to {self.log_file_path}")
        
        # Log to Weights and Biases
        if self.use_wandb:
            if not self.wandb_project_name or not self.wandb_entity:
                raise ValueError("W&B project name and entity must be provided if use_wandb is True.")
            
            try:
                wandb.init(
                    project=self.wandb_project_name,
                    entity=self.wandb_entity,
                    name=experiment_name,
                    config=params,
                    reinit=True
                )
                wandb.log({
                    'feature_wave': feature_wave,
                    'cv_score': cv_score,
                    'model_name': model_name,
                })
                wandb.finish()
                print(f"Experiment '{experiment_name}' logged to W&B")
            except Exception as e:
                print(f"Failed to log to W&B: {e}")