import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.g_bottomup import gBottomup
from model.tcn_autoencoder import TCNAutoEncoder
from training.datapair import PairDataset
from training.losses import CombinedLoss
from typing import Dict, Any
import numpy as np

class Trainer:
    """
    gBottomup 탐색과 TCN-AutoEncoder 학습을 연결하여
    적응형 온라인 학습을 수행하는 컨트롤러 클래스입니다.
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

        # gBottomup 인스턴스 생성 및 초기 인코더 주입
        self.g_bottomup = gBottomup(encoder=self.model.encoder, config=config)
        
        # 학습 관련 설정
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-4))
        self.loss_fn = CombinedLoss(
            lambda_recon=config.get('lambda_recon', 0.5),
            temperature=config.get('temperature', 0.1)
        ).to(self.device)
        self.train_epochs = config.get('train_epochs', 5)
        self.batch_size = config.get('batch_size', 64)
        self.update_interval_steps = config.get('update_interval_steps', 5) # M 스텝

        self.accumulated_pairs = []

    def _train_on_pairs(self):
        """수집된 긍정 쌍으로 모델을 미세 조정(fine-tuning)합니다."""
        if not self.accumulated_pairs:
            print("No positive pairs to train on. Skipping model update.")
            return

        print(f"Fine-tuning the model with {len(self.accumulated_pairs)} accumulated positive pairs...")
        
        dataset = PairDataset(self.accumulated_pairs, self.all_data, max_len=self.config.get('max_len', 100))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.train_epochs):
            total_loss = 0
            for batch_hist, batch_fut in dataloader:
                batch_hist, batch_fut = batch_hist.to(self.device), batch_fut.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                z_hist, recon_hist = self.model(batch_hist)
                z_fut, recon_fut = self.model(batch_fut)
                
                # Loss calculation
                loss = self.loss_fn(z_hist, recon_hist, batch_hist, z_fut, recon_fut, batch_fut)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.train_epochs}, Average Loss: {avg_loss:.4f}")

    def run(self):
        """적응형 온라인 학습 및 탐색 프로세스를 시작합니다."""
        print("Starting adaptive online training and detection process...")
        
        g_bottomup_generator = self.g_bottomup.fit(self.all_data)
        
        step_count = 0
        try:
            while True:
                # 1. gBottomup을 M 스텝 실행하고 긍정 쌍 수집
                temp_pairs = []
                for _ in range(self.update_interval_steps):
                    # generator로부터 긍정 쌍을 받음
                    positive_pairs_from_step = next(g_bottomup_generator)
                    temp_pairs.extend(positive_pairs_from_step)
                    step_count += 1
                    print(f"gBottomup Step {step_count}: Collected {len(positive_pairs_from_step)} positive pairs.")

                # 2. 누적 데이터셋에 추가
                self.accumulated_pairs.extend(temp_pairs)

                # 3. 수집된 데이터로 모델 업데이트
                self._train_on_pairs()
                
                # 4. gBottomup에 업데이트된 인코더 주입 (yield를 통해)
                g_bottomup_generator.send(self.model.encoder)

        except StopIteration:
            print("\ngBottomup process finished.")
            # 최종 결과 처리
            final_merge_history = self.g_bottomup.merge_history
            print("Final merge history collected.")
            # 이 history를 사용하여 최종 변화점 선택 로직 수행
            # final_cps = ModelSelector(self.g_bottomup).select_final_cps()
            # print(f"Final change points estimated: {final_cps}")

        print("Adaptive online training and detection process complete.")