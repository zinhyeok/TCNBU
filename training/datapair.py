import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple
from itertools import chain

class PairDataset(Dataset):
    """
    gBottomup에서 수집된 긍정 쌍(positive pairs)을 처리하여
    PyTorch 모델 학습을 위한 (history, future) 텐서 쌍을 생성하는 클래스입니다.
    """
    def __init__(self, positive_pairs: List[List[List[int]]], all_data: np.ndarray, max_len: int = 100):
        """
        Args:
            positive_pairs (List[List[List[int]]]): 
                gBottomup에서 수집된 긍정 쌍의 리스트. 
                예: [[[0, 1], [2, 3, 4]], [[10, 11, 12], [13, 14]]]
            all_data (np.ndarray): 전체 원본 시계열 데이터 (Total_Length, Channels).
            max_len (int): 각 세그먼트를 패딩/트렁케이팅할 최대 길이.
        """
        super().__init__()
        self.positive_pairs = positive_pairs
        self.all_data = all_data
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.positive_pairs)

    def _pad_or_truncate(self, segment_data: np.ndarray) -> np.ndarray:
        """세그먼트 데이터를 고정된 길이(max_len)로 맞춥니다."""
        current_len = segment_data.shape[0]
        if current_len > self.max_len:
            # 길이가 길면 잘라냄
            return segment_data[:self.max_len, :]
        elif current_len < self.max_len:
            # 길이가 짧으면 0으로 패딩
            padding_size = self.max_len - current_len
            # (padding_size, num_channels) 형태의 패딩 생성
            padding = np.zeros((padding_size, segment_data.shape[1]), dtype=segment_data.dtype)
            return np.vstack([segment_data, padding])
        return segment_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        하나의 긍정 쌍을 가져와 전처리 후 텐서로 반환합니다.

        Args:
            idx (int): 가져올 긍정 쌍의 인덱스.

        Returns:
            history_tensor (torch.Tensor): (Length, Channels)
            future_tensor (torch.Tensor): (Length, Channels)
        """
        # 긍정 쌍 [history_indices_list, future_indices_list]
        pair = self.positive_pairs[idx]
        
        history_indices = pair[0]
        future_indices = pair[1]
        
        # 원본 데이터에서 해당 인덱스의 데이터를 추출
        history_data = self.all_data[history_indices, :]
        future_data = self.all_data[future_indices, :]
        
        # 패딩/트렁케이팅 적용
        history_padded = self._pad_or_truncate(history_data)
        future_padded = self._pad_or_truncate(future_data)
        
        # Numpy 배열을 PyTorch 텐서로 변환
        history_tensor = torch.from_numpy(history_padded).float()
        future_tensor = torch.from_numpy(future_padded).float()
        
        return history_tensor, future_tensor