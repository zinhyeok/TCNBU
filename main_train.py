import torch
import numpy as np
import yaml
import os
from typing import Dict, Any

# 프로젝트의 다른 모듈들을 임포트합니다.
from models.tcn_autoencoder import TCNAutoEncoder
from training.trainer import Trainer
from data.generator import generate_multi_data # gBottomup_R_unmerge.py에서 가져온 데이터 생성기

def set_seed(seed: int = 42):
    """재현성을 위해 랜덤 시드를 고정하는 함수."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str = './configs/default_config.yaml') -> Dict[str, Any]:
    """YAML 설정 파일을 불러오는 함수."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """메인 학습 프로세스를 실행하는 함수."""
    
    # 1. 설정 파일 로드
    print("Loading configuration...")
    config = load_config()
    print("Configuration loaded successfully:")
    print(yaml.dump(config, indent=2))

    # 2. 재현성을 위한 시드 설정
    set_seed()

    # 3. 학습용 시계열 데이터 생성
    print("Generating training data...")
    train_config = config['data']['train_data_generation']
    # gBottomup은 (n, d) 형태의 numpy 배열을 기대함
    all_data, _ = generate_multi_data(
        n=train_config['n_samples'],
        d=train_config['n_dims'],
        scenario=train_config['scenario']
    )
    print(f"Training data generated with shape: {all_data.shape}")

    # 4. TCN-AutoEncoder 모델 초기화
    print("Initializing the TCN-AutoEncoder model...")
    model_config = config['model']
    # 데이터 차원이 설정과 다를 경우 업데이트
    model_config['in_channels'] = all_data.shape[1] 
    
    model = TCNAutoEncoder(
        in_channels=model_config['in_channels'],
        embedding_dim=model_config['embedding_dim'],
        hidden_channels=model_config['hidden_channels'],
        depth=model_config['depth'],
        kernel_size=model_config['kernel_size']
    )
    print("Model initialized successfully.")

    # 5. Trainer 초기화
    print("Initializing the Trainer...")
    trainer = Trainer(model=model, all_data=all_data, config=config)
    print("Trainer initialized successfully.")

    # 6. 적응형 온라인 학습 프로세스 실행
    # 이 run 메소드는 내부적으로 gBottomup 탐색과 모델 학습을 순환하며 실행합니다.
    trainer.run()

    # 7. 학습된 최종 인코더 모델 저장
    print("Training complete. Saving the final encoder model...")
    save_path = config['data']['model_save_path']
    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 인코더의 가중치만 저장합니다.
    torch.save(trainer.model.encoder.state_dict(), save_path)
    print(f"Encoder model saved to: {save_path}")

if __name__ == '__main__':
    # 필요한 R 패키지 로드 (gBottomup 내부에서 rpy2를 사용하므로)
    print("Initializing R environment...")
    try:
        from utils import r_utils
        r_utils.init_r_packages()
        print("R environment initialized successfully.")
    except Exception as e:
        print(f"Could not initialize R environment. Please ensure R and gSeg are installed. Error: {e}")
        # R 환경 초기화 실패 시 종료
        exit()

    main()