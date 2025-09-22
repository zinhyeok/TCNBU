import torch
import numpy as np
import yaml
import os
from typing import Dict, Any

# 프로젝트의 다른 모듈들을 임포트합니다.
from models.tcn_autoencoder import TCNEncoder # AutoEncoder 전체가 아닌 Encoder만 필요
from models.g_bottomup import gBottomup
from data.generator import generate_multi_data # 테스트 데이터 생성을 위해 임시 사용
# from models.model_selector import ModelSelector # 최종 CP 선택 로직 (구현 필요)

def load_config(config_path: str = './configs/default_config.yaml') -> Dict[str, Any]:
    """YAML 설정 파일을 불러오는 함수."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_trained_encoder(config: Dict[str, Any]) -> TCNEncoder:
    """
    사전 훈련된 TCNEncoder 모델의 가중치를 불러옵니다.

    Args:
        config (Dict[str, Any]): 모델 아키텍처 및 가중치 경로 정보가 담긴 설정 딕셔너리.

    Returns:
        TCNEncoder: 가중치가 로드된 인코더 모델 객체.
    """
    model_config = config['model']
    encoder = TCNEncoder(
        in_channels=model_config['in_channels'],
        embedding_dim=model_config['embedding_dim'],
        hidden_channels=model_config['hidden_channels'],
        depth=model_config['depth'],
        kernel_size=model_config['kernel_size']
    )
    
    model_path = config['data']['model_save_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run main_train.py first.")
        
    print(f"Loading pre-trained encoder weights from: {model_path}")
    # CPU 환경에서도 로드할 수 있도록 map_location 설정
    encoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    encoder.eval() # 모델을 추론 모드로 설정
    
    return encoder

def main():
    """메인 변화점 탐지 프로세스를 실행하는 함수."""
    
    # 1. 설정 파일 로드
    print("Loading configuration...")
    config = load_config()

    # 2. 탐지할 새로운 데이터 생성 또는 로드
    # 여기서는 시연을 위해 데이터를 생성하지만, 실제로는 새로운 데이터를 로드해야 합니다.
    print("Generating new data for detection...")
    test_data, true_change_points = generate_multi_data(
        n=500, # 학습 데이터보다 짧은 길이로 설정
        d=config['model']['in_channels'],
        scenario='model15' # 다른 시나리오로 테스트 가능
    )
    print(f"Test data generated with shape: {test_data.shape}")
    print(f"True change points are at: {true_change_points}")

    # 3. 사전 훈련된 인코더 모델 로드
    encoder = load_trained_encoder(config)
    
    # 4. gBottomup 객체 초기화 (학습된 인코더 주입)
    print("Initializing gBottomup with the pre-trained encoder...")
    # 탐지 시에는 gBottomup의 'eliminate' 옵션을 사용하여 최종 CP를 선택하도록 설정
    # config['gbottomup']['eliminate'] = 'both' # 예: BIC 기반 선택
    g_bottomup_detector = gBottomup(encoder=encoder, config=config['gbottomup'])
    print("gBottomup detector initialized successfully.")

    # 5. 변화점 탐지 실행
    # 🔥 탐지 시에는 fit() 제너레이터를 끝까지 실행하되, 모델을 업데이트하지 않음
    print("Starting change point detection process...")
    g_bottomup_generator = g_bottomup_detector.fit(test_data)
    
    try:
        while True:
            # generator로부터 긍정 쌍을 받지만 사용하지 않음
            _ = next(g_bottomup_generator)
            # 업데이트된 인코더를 보내는 대신 None을 보냄
            g_bottomup_generator.send(None)
    except StopIteration:
        print("\ngBottomup detection process finished.")

    # 6. 최종 변화점 결과 처리
    # gBottomup의 fit 메소드가 최종 결과를 반환하도록 수정하거나,
    # merge_history를 후처리하여 최종 CP를 선택해야 합니다.
    # 여기서는 마지막 단계의 병합 후보들을 최종 후보로 간주하는 간단한 예시를 보여줍니다.
    if g_bottomup_detector.merge_history:
        # 마지막 병합 단계의 CP들을 후보로 선택
        last_step_stats = g_bottomup_detector.merge_history[-1]
        candidate_cps = sorted([stat.cp for stat in last_step_stats])
        
        # ModelSelector를 사용하여 최종 CP 선택 (예시)
        # model_selector = ModelSelector(g_bottomup_detector)
        # estimated_cps, _ = model_selector.stepwise_elimination(candidate_cps)
        
        # 여기서는 가장 G-값이 큰 (가장 변화가 뚜렷한) 3개를 선택하는 예시
        last_step_stats.sort(key=lambda s: s.G, reverse=True)
        estimated_cps = sorted([s.cp for s in last_step_stats[:3]])

    else:
        estimated_cps = []

    print("\n--- Detection Results ---")
    print(f"True Change Points:     {true_change_points}")
    print(f"Estimated Change Points:  {estimated_cps}")
    print("-------------------------")


if __name__ == '__main__':
    # R 환경 초기화
    print("Initializing R environment...")
    try:
        from utils import r_utils
        r_utils.init_r_packages()
        print("R environment initialized successfully.")
    except Exception as e:
        print(f"Could not initialize R environment. Error: {e}")
        exit()

    main()