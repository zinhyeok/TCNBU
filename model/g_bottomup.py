import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Any
from math import ceil, log2, floor
from itertools import chain

# 헬퍼 함수 및 클래스는 별도 파일로 분리하는 것이 좋지만, 여기서는 설명을 위해 함께 정의합니다.
# 실제 구현 시 utils/r_utils.py, models/model_selector.py 등으로 분리해야 합니다.
from utils.r_utils import compute_g_stat_from_graph  # R gSeg 호출 함수 (가정)
# from models.model_selector import ModelSelector # 최종 CP 선택 로직 (가정)


@dataclass
class CPStat:
    """Change Point와 G-통계량 쌍을 저장하는 데이터 클래스"""
    cp: int
    G: float
    window_indices: List[List[int]] # 어떤 윈도우에서 계산되었는지 추적

class gBottomup:
    """
    TCN 인코더와 유기적으로 결합하여 적응형 온라인 학습을 수행하는 gBottomup 알고리즘 클래스.
    """
    def __init__(self, encoder: torch.nn.Module, config: Dict[str, Any]):
        """
        Args:
            encoder (torch.nn.Module): TCN 인코더 모델 객체.
            config (Dict[str, Any]): 하이퍼파라미터 및 설정을 담은 딕셔너리.
        """
        self.encoder = encoder
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder.to(self.device)

        # 설정 값들을 클래스 속성으로 저장
        self.min_obs = config.get('min_obs', 4)
        self.merge_percentile = config.get('merge_percentile', 0.1)
        self.graph_type = config.get('graph_type', 'mst')
        self.k_max = config.get('k_max', 5) # 그래프 생성 시 최대 k값

        # 탐색 과정을 기록할 히스토리
        self.seg_history: List[List[List[int]]] = []
        self.merge_history: List[List[CPStat]] = []
        
        self.observations: Optional[np.ndarray] = None

    def set_encoder(self, encoder: torch.nn.Module):
        """외부(Trainer)에서 업데이트된 인코더를 주입받기 위한 메소드."""
        self.encoder = encoder
        self.encoder.to(self.device)

    def _embed_data(self, window_data: np.ndarray) -> np.ndarray:
        """주어진 데이터를 현재 인코더를 사용해 임베딩 공간으로 변환합니다."""
        self.encoder.eval()
        with torch.no_grad():
            # (Length, Channels) -> (1, Length, Channels)
            tensor_data = torch.from_numpy(window_data).float().unsqueeze(0).to(self.device)
            # 인코더는 (B, L, C_emb) 형태의 포인트별 임베딩을 반환
            embedded_tensor = self.encoder(tensor_data)
            # (1, L, C_emb) -> (L, C_emb)
            return embedded_tensor.squeeze(0).cpu().numpy()

    def _build_graph(self, observations: np.ndarray) -> np.ndarray:
        """
        (임베딩된) 관측값으로부터 k-MST 그래프를 구축하고 엣지 리스트를 반환합니다.
        (gBottomup_R_unmerge.py의 build_graph 로직을 단순화하여 통합)
        """
        # 이 부분은 gBottomup_R_unmerge.py의 build_graph 로직을 가져와 구현합니다.
        # 여기서는 핵심 로직을 보여주기 위해 개념적인 코드로 대체합니다.
        from sklearn.metrics.pairwise import euclidean_distances
        from scipy.sparse.csgraph import minimum_spanning_tree

        n_obs = observations.shape[0]
        k = min(self.k_max, (n_obs // 2) - 1, int(np.sqrt(n_obs)))
        k = max(1, k)
        
        dist_matrix = euclidean_distances(observations)
        
        # k-MST 생성 로직 (간략화)
        # 실제로는 누적 엣지를 관리해야 함
        mst = minimum_spanning_tree(dist_matrix)
        rows, cols = mst.nonzero()
        
        # R과 호환되도록 1-based 인덱싱으로 변환
        edges = np.array(list(zip(rows, cols)), dtype=int) + 1
        return edges

    def compute_g(self, window_indices: List[List[int]]) -> CPStat:
        """
        주어진 윈도우에 대해 임베딩 변환 후 G-통계량을 계산합니다.
        """
        # 1. 원본 데이터 추출
        flat_indices = list(chain.from_iterable(window_indices))
        window_data = self.observations[flat_indices, :]

        # 2. 임베딩 공간으로 변환 🔥
        embedded_data = self._embed_data(window_data)

        # 3. 임베딩 공간에서 그래프 생성
        edges = self._build_graph(embedded_data)
        
        # 4. G-통계량 계산 (R 헬퍼 함수 호출)
        n_obs = len(flat_indices)
        t_split = len(window_indices[0]) - 1 # 0-based 분할점
        
        g_stat = compute_g_stat_from_graph(n_obs, edges, t_split)

        # 5. 결과 반환
        change_point_index = flat_indices[t_split]
        return CPStat(cp=change_point_index, G=g_stat, window_indices=window_indices)

    def _select_merge_candidates(self, stats: List[CPStat]) -> List[int]:
        """G-통계량 순위를 기준으로 병합할 윈도우의 인덱스를 선택합니다."""
        if not stats:
            return []
        
        n = len(stats)
        k = ceil(n * self.merge_percentile) if n > 0 else 0
        if k == 0 and n > 1: k = 1 # 최소 1개는 병합
        
        # G-값 기준 오름차순 정렬하여 상위 k개 선택
        sorted_indices = sorted(range(n), key=lambda i: stats[i].G)
        return sorted_indices[:k]

    def fit(self, data: np.ndarray):
        """
        적응형 온라인 학습을 위한 fit 메소드.
        Trainer에 의해 제어되며, M 스텝마다 긍정 쌍을 yield합니다.
        """
        self.observations = data
        n, d = data.shape

        # 초기 세그먼트 생성
        seg_idx = [list(range(i, min(i + self.min_obs, n))) for i in range(0, n, self.min_obs)]
        self.seg_history.append([s.copy() for s in seg_idx])

        # 사전 Unmerge 단계 (필요 시 추가)
        # num_unmerge_steps = floor(log2(n))
        # for _ in range(num_unmerge_steps):
        #     seg_idx = self._perform_unmerge_pass(seg_idx)
        # self.seg_history.append([seg.copy() for seg in seg_idx])

        # 메인 탐색-학습 루프 (외부 Trainer가 제어)
        while len(seg_idx) > 1:
            # 1. 현재 세그먼트 구조로 슬라이딩 윈도우 생성
            # (gBottomup_R_unmerge.py의 sliding_windows 로직 사용)
            all_windows = self._create_sliding_windows(seg_idx)
            if not all_windows: break

            # 2. 모든 윈도우에 대해 G-통계량 계산
            current_stats = [self.compute_g(win) for win in all_windows]
            self.merge_history.append(current_stats)

            # 3. 병합할 후보 윈도우 선택
            candidate_indices = self._select_merge_candidates(current_stats)
            if not candidate_indices: break
            
            # 4. 선택된 후보들을 긍정 쌍으로 구성하여 yield
            positive_pairs = [current_stats[i].window_indices for i in candidate_indices]
            
            # 🔥 Trainer에게 긍정 쌍을 전달하고, 업데이트된 인코더를 기다림
            new_encoder = yield positive_pairs
            if new_encoder is not None:
                self.set_encoder(new_encoder)

            # 5. 세그먼트 병합 수행
            # (gBottomup_R_unmerge.py의 merge_segments_sequential 로직 사용)
            seg_idx = self._merge_segments(all_windows, candidate_indices)
            self.seg_history.append([s.copy() for s in seg_idx])
            
            # 후-병합 Unmerge 단계 (필요 시 추가)
            # seg_idx = self._perform_unmerge_pass(seg_idx)
            # self.seg_history.append([seg.copy() for seg in seg_idx])
            
        # 최종 변화점 선택 로직 (ModelSelector 등 사용)
        # final_cps = ModelSelector(self).select_final_cps()
        # return final_cps
        return self.merge_history # 임시로 전체 히스토리 반환

    # 아래는 gBottomup_R_unmerge.py에서 가져와야 할 헬퍼 메소드들입니다.
    # 설명을 위해 시그니처만 남겨둡니다.
    def _create_sliding_windows(self, seg_idx: List[List[int]]) -> List[List[List[int]]]:
        # ... gBottomup_R_unmerge.py의 sliding_windows 구현 ...
        windows = []
        n = len(seg_idx)
        for i in range(n - 1):
            windows.append([seg_idx[i], seg_idx[i+1]])
        return windows

    def _merge_segments(self, all_windows: List[List[List[int]]], indices_to_merge: List[int]) -> List[List[int]]:
        # ... gBottomup_R_unmerge.py의 merge_segments_sequential 구현 ...
        
        # 병합될 윈도우들을 G-값 순서대로 정렬 (gBottomup_R_unmerge.py 참조)
        # 여기서는 단순화를 위해 인덱스 순서대로 병합
        
        segments_to_merge_set = set()
        for i in indices_to_merge:
            win = all_windows[i]
            segments_to_merge_set.add(tuple(map(tuple, win)))
        
        merged_segments = []
        used_segments = set()
        
        # 1. 병합되지 않는 세그먼트 유지
        current_segments = list(chain.from_iterable(all_windows))
        final_segments = []
        
        # 이 부분은 merge_segments_sequential의 정확한 로직이 필요합니다.
        # 아래는 개념적인 구현입니다.
        
        all_base_segments = {tuple(s) for s in list(chain.from_iterable(all_windows))}
        
        newly_created = []
        removed = set()

        for i in indices_to_merge:
            window_to_merge = all_windows[i]
            
            seg_ A_tuple = tuple(window_to_merge[0])
            seg_B_tuple = tuple(window_to_merge[1])

            if seg_A_tuple in removed or seg_B_tuple in removed:
                continue

            merged = sorted(list(set(chain.from_iterable(window_to_merge))))
            newly_created.append(merged)
            removed.add(seg_A_tuple)
            removed.add(seg_B_tuple)

        final_segments = [list(s) for s in all_base_segments if s not in removed]
        final_segments.extend(newly_created)
        
        return sorted(final_segments, key=lambda x: x[0])


    def _perform_unmerge_pass(self, seg_idx: List[List[int]]) -> List[List[int]]:
        # ... gBottomup_R_unmerge.py의 _perform_unmerge_pass 구현 ...
        # 이 함수 내부에서도 self.compute_g를 호출해야 함
        return seg_idx