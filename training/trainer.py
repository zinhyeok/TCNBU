import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.g_bottomup import gBottomup
from model.tcn_autoencoder import TCNAutoEncoder
from training.datapair import PairDataset
from training.losses import CombinedLoss
from typing import Dict, Any, List
import numpy as np
import os

class Trainer:
    """
    gBottomup 탐색과 TCN-AutoEncoder 학습을 '매 스텝'마다 연결하여
    단계별 적응형 학습을 수행하는 컨트롤러 클래스입니다.
    """
    def __init__(self, model: TCNAutoEncoder, all_data: np.ndarray, config: Dict[str, Any]):
        """
        Args:
            model (TCNAutoEncoder): 학습시킬 TCN-AutoEncoder 모델.
            all_data (np.ndarray): 전체 원본 시계열 데이터.
            config (Dict[str, Any]): 하이퍼파라미터 및 설정을 담은 딕셔너리.
        """
        self.model = model
        self.all_data = all_data
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # gBottomup 인스턴스 생성 (초기 인코더는 학습되지 않은 상태)
        self.g_bottomup = gBottomup(encoder=self.model.encoder, config=config['gbottomup'])
        
        # 학습 관련 설정
        train_config = config['training']
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_config.get('learning_rate', 1e-4))
        self.loss_fn = CombinedLoss(
            lambda_recon=train_config.get('lambda_recon', 0.5),
            temperature=train_config.get('temperature', 0.1)
        ).to(self.device)
        self.train_epochs = train_config.get('train_epochs', 5)
        self.batch_size = train_config.get('batch_size', 64)

        # 긍정 쌍을 누적하여 저장할 리스트
        self.accumulated_pairs: List[List[List[int]]] = []

    def _train_on_accumulated_pairs(self):
        """누적된 전체 긍정 쌍으로 모델을 학습(또는 미세 조정)합니다."""
        if not self.accumulated_pairs:
            print("No positive pairs to train on. Skipping model update.")
            return

        print(f"\n--- Training Step ---")
        print(f"Updating the model with {len(self.accumulated_pairs)} accumulated positive pairs...")
        
        dataset = PairDataset(
            self.accumulated_pairs, 
            self.all_data, 
            max_len=self.config['model'].get('max_len', 100)
        )
        # 데이터가 배치 크기보다 작을 경우 배치 크기를 조절
        effective_batch_size = min(self.batch_size, len(dataset))
        if effective_batch_size == 0: return
        
        dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.train_epochs):
            total_loss = 0
            for batch_hist, batch_fut in dataloader:
                batch_hist, batch_fut = batch_hist.to(self.device), batch_fut.to(self.device)
                
                self.optimizer.zero_grad()
                
                z_hist, recon_hist = self.model(batch_hist)
                z_fut, recon_fut = self.model(batch_fut)
                
                loss = self.loss_fn(z_hist, recon_hist, batch_hist, z_fut, recon_fut, batch_fut)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.train_epochs}, Average Loss: {avg_loss:.4f}")
        print(f"--- Model Updated ---")


    def run(self):
        """단계별 적응형 학습 및 탐색 프로세스를 시작합니다."""
        print("Starting bootstrapped adaptive learning process...")
        
        # gBottomup.fit()은 제너레이터이므로, for loop로 각 스텝을 순회합니다.
        g_bottomup_generator = self.g_bottomup.fit(self.all_data)
        
        step_count = 0
        for positive_pairs_this_step in g_bottomup_generator:
            step_count += 1
            is_first_step = (step_count == 1)
            
            print(f"\n[Step {step_count}] gBottomup provided {len(positive_pairs_this_step)} new positive pairs.")
            if is_first_step:
                print("This is the first step, based on raw data statistics.")
            else:
                print("Based on the latest embedding space statistics.")

            # 1. 이번 스텝에서 얻은 긍정 쌍을 누적 데이터셋에 추가
            self.accumulated_pairs.extend(positive_pairs_this_step)

            # 2. 누적된 전체 데이터로 모델 업데이트
            self._train_on_accumulated_pairs()
            
            # 3. 🔥 gBottomup에 업데이트된 인코더를 다시 주입 (send)
            #    제너레이터의 다음 루프(다음 스텝)는 이 새로운 인코더를 사용하게 됩니다.
            try:
                g_bottomup_generator.send(self.model.encoder)
            except StopIteration:
                # gBottomup이 마지막 스텝이었던 경우, send에서 StopIteration이 발생할 수 있음
                break

        print("\n========================================================")
        print("gBottomup process finished.")
        print("Adaptive learning and detection process complete.")
        print("========================================================")