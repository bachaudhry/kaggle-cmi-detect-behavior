import os

# Define the root directory of the project
# This assumes the script is run from within the project structure
PROJECT_PATH = '~/code/kaggle/kaggle-cmi-detect-behavior/'
DATA_PATH = os.path.join(PROJECT_PATH, 'data')

# W&B configuration
USE_WANDB = True
WANDB_PROJECT = 'kaggle-cmi-detect-behavior'
WANDB_ENTITY = 'b-a-chaudhry-'
