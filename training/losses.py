import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss_with_filtering(z_history: torch.Tensor, z_future: torch.Tensor, temperature: float = 0.1, similarity_threshold: float = 0.9):
    """
    배치 내 네거티브 샘플링과 유사도 기반 필터링을 적용한 InfoNCE 손실 함수.

    Args:
        z_history (torch.Tensor): History 세그먼트의 요약 임베딩 벡터 (Batch, embedding_dim).
        z_future (torch.Tensor): Future 세그먼트의 요약 임베딩 벡터 (Batch, embedding_dim).
        temperature (float): 코사인 유사도를 스케일링하는 온도 파라미터.
        similarity_threshold (float): '가짜 부정'을 필터링하기 위한 유사도 임계값.

    Returns:
        torch.Tensor: 계산된 InfoNCE 손실 값 (스칼라).
    """
    device = z_history.device
    batch_size = z_history.size(0)
    
    # 코사인 유사도 행렬 계산
    # (B, 1, D)와 (1, B, D) -> (B, B)
    sim_matrix = F.cosine_similarity(z_history.unsqueeze(1), z_future.unsqueeze(0), dim=-1)

    # 🔥 "가짜 부정" 필터링 로직
    # 긍정 쌍 (대각선 요소)의 유사도를 기준으로 동적 임계값 설정
    positive_similarities = torch.diag(sim_matrix)
    dynamic_threshold = positive_similarities.mean()
    
    # 자기 자신을 제외한 (off-diagonal) 부정 쌍 후보들 중에서
    # 임계값을 넘는 '가짜 부정'을 식별
    mask = torch.ones_like(sim_matrix, dtype=torch.bool, device=device)
    mask.fill_diagonal_(False) # 자기 자신(긍정 쌍)은 필터링 대상에서 제외
    
    false_negatives = (sim_matrix > dynamic_threshold) & mask
    
    # 유사도 행렬에서 '가짜 부정' 위치의 값을 매우 낮게 설정하여 손실 계산에서 제외
    sim_matrix.masked_fill_(false_negatives, -1e9)
    
    # 최종 손실 계산
    sim_matrix /= temperature
    labels = torch.arange(batch_size, device=device)
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


class CombinedLoss(nn.Module):
    """
    대조 학습 손실과 재구성 손실을 결합하는 클래스.
    """
    def __init__(self, lambda_recon: float = 0.5, temperature: float = 0.1):
        """
        Args:
            lambda_recon (float): 재구성 손실의 가중치.
            temperature (float): InfoNCE 손실의 온도 파라미터.
        """
        super().__init__()
        self.lambda_recon = lambda_recon
        self.temperature = temperature
        self.reconstruction_loss_fn = nn.MSELoss()

    def forward(self, 
                z_sequence_hist: torch.Tensor, x_recon_hist: torch.Tensor, x_original_hist: torch.Tensor,
                z_sequence_fut: torch.Tensor, x_recon_fut: torch.Tensor, x_original_fut: torch.Tensor
               ) -> torch.Tensor:
        """
        Args:
            z_sequence_hist, z_sequence_fut: 인코더가 출력한 포인트별 임베딩 시퀀스.
            x_recon_hist, x_recon_fut: 디코더가 출력한 재구성 시퀀스.
            x_original_hist, x_original_fut: 원본 시퀀스.

        Returns:
            torch.Tensor: 최종 결합 손실 값.
        """
        # 1. 재구성 손실 계산
        loss_recon_hist = self.reconstruction_loss_fn(x_recon_hist, x_original_hist)
        loss_recon_fut = self.reconstruction_loss_fn(x_recon_fut, x_original_fut)
        total_loss_recon = loss_recon_hist + loss_recon_fut

        # 2. 대조 학습 손실 계산
        # 포인트별 임베딩 시퀀스에 Temporal Pooling을 적용하여 요약 벡터 생성
        z_hist_summary = z_sequence_hist.mean(dim=1)
        z_fut_summary = z_sequence_fut.mean(dim=1)
        
        loss_contrastive = info_nce_loss_with_filtering(
            z_hist_summary, z_fut_summary, self.temperature
        )
        
        # 3. 두 손실을 가중합하여 최종 손실 반환
        final_loss = loss_contrastive + self.lambda_recon * total_loss_recon
        return final_loss