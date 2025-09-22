import torch
import numpy as np
import yaml
import os
from typing import Dict, Any
os.environ['LANGUAGE'] = 'en_US.UTF-8' 
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from model.tcn_autoencoder import TCNAutoEncoder
from training.trainer import Trainer
from data.dataGenerator import generate_multi_data 
from utils.r_utils import init_r_packages

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str = './configs/default_config.yaml') -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    print("Initializing R environment...")
    init_r_packages()
    print("R environment initialized successfully.")
    
    # 1. load config
    print("Loading configuration...")
    config = load_config()
    print("Configuration loaded successfully:")
    print(yaml.dump(config, indent=2))

    # 2. seed setting
    print("Setting random seed for reproducibility...")
    set_seed()

    # 3. get data
    print("Generating training data...")
    train_config = config['data']['train_data_generation']
    all_data, _ = generate_multi_data(
        n=train_config['n_samples'],
        d=train_config['n_dims'],
        scenario=train_config['scenario']
    )
    print(f"Training data generated with shape: {all_data.shape}")

    # 4. Initialized TCN-AutoEncoder model
    print("Initializing the TCN-AutoEncoder model...")
    model_config = config['model']
    model_config['in_channels'] = all_data.shape[1] 
    
    model = TCNAutoEncoder(
        in_channels=model_config['in_channels'],
        embedding_dim=model_config['embedding_dim'],
        hidden_channels=model_config['hidden_channels'],
        depth=model_config['depth'],
        kernel_size=model_config['kernel_size']
    )
    print("Model initialized successfully.")

    # 5. Initialize Trainer
    print("Initializing the Trainer...")
    trainer = Trainer(model=model, all_data=all_data, config=config)
    print("Trainer initialized successfully.")

    # 6. Run adaptive online learning process
    trainer.run()

    # 7. Save the trained final encoder model
    print("Training complete. Saving the final encoder model...")
    save_path = config['data']['model_save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(trainer.model.encoder.state_dict(), save_path)
    print(f"Encoder model saved to: {save_path}")

if __name__ == '__main__':
    main()