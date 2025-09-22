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
from model.g_bottomup import gBottomup
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


    data_config = config['data']['train_data_generation']
    all_data, true_cps = generate_multi_data(
        n=data_config['n_samples'],
        d=data_config['n_dims'],
        scenario=data_config['scenario']
    )
    print(f"Data generated. Shape: {all_data.shape}, True CPs: {true_cps}")

    # ===================================================================
    # PHASE 1: 적응형 학습으로 인코더 훈련
    # ===================================================================
    print("\n\n--- PHASE 1: Adaptive Encoder Training ---")
    
    model_config = config['model']
    model_config['in_channels'] = all_data.shape[1]
    autoencoder = TCNAutoEncoder(**model_config)
    trainer = Trainer(model=autoencoder, all_data=all_data, config=config)

    trained_encoder = trainer.run_adaptive_learning()
    
    # 학습된 인코더 저장
    save_path = config['data']['model_save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(trained_encoder.state_dict(), save_path)
    print(f"Final trained encoder saved to: {save_path}")

    # ===================================================================
    # PHASE 2: 학습된 인코더로 최종 변화점 탐지
    # ===================================================================
    print("\n\n--- PHASE 2: Final Change Point Detection ---")

    # 2.1. 학습된 인코더로 새로운 gBottomup 탐지기 인스턴스 생성
    detector = gBottomup(encoder=trained_encoder, config=config['gbottomup'])
    
    # 2.2. detect 메소드를 호출하여 최종 변화점 확정
    estimated_cps, G_values, cp_candidates = detector.detect(all_data, selection_method='topk')

    # 2.3. 결과 출력
    print("\n\n--- FINAL RESULTS ---")
    print(f"True Change Points:      {true_cps}")
    print(f"Estimated Change Points: {estimated_cps}")
    print(f"cp_candidates: {cp_candidates}")
    print("---------------------")


if __name__ == '__main__':
    main()