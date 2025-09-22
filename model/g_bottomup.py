import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
import logging
from datetime import datetime
import copy
import pandas as pd
import subprocess
import json
import sys, os
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable, Set
from sklearn.metrics.pairwise import euclidean_distances
from itertools import chain, repeat, combinations
from heapq import nsmallest
from math import ceil
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.visualize import plot_tree_layout_custom, plot_segment_tree
from data.dataGenerator import generate_single_data, generate_multi_data
from scipy.sparse import find
import torch
from model.model_selector import ModelSelector  

from math import ceil, floor, log2
from utils.r_utils import compute_g_stat_from_graph
@dataclass
class CPStat:          # (cp, G) 한 쌍
    cp: int
    G: float

class gBottomup:
    def __init__(self, encoder: torch.nn.Module,  config: dict, start_with=0, num_cp=3, alpha=0.05, isnan=0,
                c=2, isFullTree=True, eliminate='both',  visualize=False, model_timestamp=None, logger_timestamp=None):
        """
        Parameters:
        ----------
            encoder: torch.nn.Module
                TCN-AutoEncoder의 인코더 부분
            config: dict
                gBottomup 관련 하이퍼파라미터 딕셔너리
            start_with: int
                시작 인덱스, 데이터의 시작 index, python 계열 0, R 1
            model_type: str
                모델 종류, base, local, max, step 
            num_cp: int
                BLR 위해서 사용되는 CP의 개수
            min_obs: int
                초기 세그먼트 크기
            merge_percentile: float
                병합할 window를 선택하기 위한 G 통계량의 백분위수 (0~1 사이의 값)
                예: 0.1이면 하위 10% G 통계량을 가진 window를 선택하여 병합
            alpha: int or None
                유의수준 (기본값 0.05), threshold 계산에 사용   
            isnan: int
                NaN을 inf로 변환할지 여부
                - 0: NaN을 0으로 변환
                - 1: NaN을 inf로 변환
            c: int
                BIC 계산 시 사용되는 상수
            isFullTree: bool
                전체 트리를 끝까지 만들지, early stop 허용할지 여부
            eliminate: str
                - 'forward': forward elimination
                - 'backward': backward elimination
                - 'both': forward + backward(backward elimination 후 forward elimination)
                - 'none': 최종 1개만 뱉음

            visualize: bool
                병합과정 시각화 여부
            
            model_timestamp: str
                모델 이름에 포함될 타임스탬프
            
            logger_timestamp: str
                로거 이름에 포함될 타임스탬프
        """
        np.seterr(divide='ignore', invalid='ignore')
        self.encoder = encoder
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder.to(self.device)
        self.n_thread = self.config.get('num_threads', os.cpu_count())


        self.min_obs = config.get('min_obs', 4)
        self.merge_percentile = config.get('merge_percentile', 0.1)

        self.start_with = start_with
        self.num_cp = num_cp
        self.alpha = alpha
        self.isnan = isnan
        self.isFullTree = isFullTree
        self.eliminate = eliminate

        self.visualize = visualize
        
        self.seg_history: List[List[List[int]]] = []

        self.model_timestamp = model_timestamp if model_timestamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger_timestamp = logger_timestamp

        if self.alpha is not None:
            df = 2  # 자유도 설정 (chi2 분포에 사용)
            self.CRITICAL = chi2.ppf(1 - self.alpha, df)
        
        self.c = c  # BIC 계산 시 사용되는 상수
        self.merge_history: List[List[CPStat]] = []
        self.actual_merges_history: List[List[int]] = []
        self._is_first_step = True


    def sliding_windows(self, seg_idx):
        """
        seg_idx: list of segments index
        return:
            list of windows with segments indices
            - [[0, 1], [2, 3], ...] 형태로 각 window에 포함된 segment의 인덱스 리스트
        """
        windows = []
        seg_idx = [seg for seg in seg_idx if len(seg) > 0]
        n = len(seg_idx)
        stride = 1
        visited = set()

        for start_idx in range(n - 1):
            current_window = []
            obs_count = 0
            segment_count = 0
            window_segment_indices = []

            for idx in range(start_idx, n, stride):
                cur_idx = seg_idx[idx]
                current_window.append(cur_idx)
                window_segment_indices.append(idx)  # ✅ segment의 위치 인덱스를 추적

                obs_count += len(cur_idx)
                segment_count += 1

                if segment_count >= 2:
                    windows.append(current_window.copy())
                    visited.update(window_segment_indices)  # ✅ 관측값이 아닌 segment index 기록
                    break

        remaining = [seg_idx[i] for i in range(n) if i not in visited]

        remaining = [seg_idx[i] for i in range(n) if i not in visited]
        flat_remaining = list(chain.from_iterable(remaining))  # [198, 199, ...]

        if flat_remaining:
            if windows:
                # 마지막 윈도우의 마지막 segment에 붙이기
                windows[-1][-1].extend(flat_remaining)
            else:
                # 윈도우가 아예 없었다면 새로 시작
                windows.append([flat_remaining])
        try: 
            windows = sorted(windows, key=lambda w: w[0][0] if w and len(w[0]) > 0 else float('inf'))
        except:
            pass
        
        return windows  # 윈도우가 없으면 빈 리스트 반환
    
    def get_t(self, window_indices):
        """
        주어진 segment 리스트로부터 t, n1, n2를 계산한다.
        Parameters:
            index_lst: list of segments
                - [[1], [2], [3,4,5], [6], [7]] 형태의 segment의 index 리스트
        
        Returns:
            t: int
                - group1의 마지막 observation index (0-based)
            n1: int
                - group1의 관측치 수
            n2: int
                - group2의 관측치 수
        """
        index_lst= window_indices.copy()  # 원본 리스트를 변경하지 않도록 복사
        # Step 1:segment 경계 위치 구하기 (누적합)
        segment_sizes = [len(segment) for segment in index_lst]

        cumulative_sizes = np.cumsum(segment_sizes) 
        n = int(cumulative_sizes[-1])
        t = int(cumulative_sizes[0])-1  # index로 맞춤, 0 based라 1뺌
        n1 = int(cumulative_sizes[0])  # group1의 관측치 수
        n2 = n - n1
        return t, n1, n2

            
    def build_graph(self, observations, graph_type='mst', eliminate=True, min_window_length=None):
        """
        관측값으로부터 k-graph를 구축하고 그래프를 반환한다.

        Parameters:
            observations: list of vectors (n x d)
                - 관측값 리스트
            k: int
                - 몇 번 MST를 결합할지

        Returns:
            edges: list of (i, j)
                - k-graph 누적된 edge 리스트
            weights: list of float
                - rank_type이 주어졌을 때 edge별 가중치 리스트 (없으면 None)
        """
        observations = np.array(observations)
        n = observations.shape[0]      
        #k값의 최대값 설정
        # k는 n//2-1보다 작거나 같아야 함, 완전그래프를 만들지 못하도록 제한(mst의 경우 N/2로 최대가 제한됨, nng는 n-1개)
        if min_window_length is not None:
            step_n = min(n, min_window_length)  # 최소 윈도우 길이보다 작으면 그 길이로 제한
        else:
            step_n = n  # 현재 윈도우 길이로 설정
        
        # k값 설정
        # 30 상한
        if graph_type == 'mst':
            # 여러 k 후보 값들을 계산
            k_candidate1 = int(np.sqrt(step_n))
            k_candidate2 = (step_n) // 2 - 1
            k = min(30, k_candidate1, k_candidate2)

        # eliminate 시에는 segment 크기에 상관없이 최대 5로 제한
        if eliminate:
            k_candidate1 = int(np.sqrt(step_n))
            k_candidate2 = (step_n) // 2 - 1
            k = min(5, k_candidate1, k_candidate2)

        k = max(1, k)  # k는 최소 1로 설정

        # Step 1: 거리 기반 유사도 행렬 계산
        dist = euclidean_distances(observations) 
        # dist = squareform(pdist(observations, metric='mahalanobis'))
        sim = -dist

        used = np.zeros((n, n))  # 사용된 edge 기록
        edge_to_level = dict()   # (i,j): level 기록

        all_edges = []  #반환할 edge 리스트, 누적 edge 

        for level in range(1, k+1):
            # Step 2: 현재까지 사용되지 않은 edge만 고려
            effective_dist = np.where(used == 0, dist, 1e8)

            if graph_type == 'mst':
                # Step 3: MST 만들기 (최대 유사도 MST)
                # 무방향그래프, 단 symmetric은 아님
                mst = minimum_spanning_tree(effective_dist)  # 최소 거리를 최대 유사도로 변환
                mst = mst.toarray()
                graph = mst
                

            elif graph_type == 'nng':
                nng = kneighbors_graph(effective_dist, 1, mode='connectivity', include_self=False).toarray()
                n = nng.shape[0]
                # 양방향인 엣지 찾아서 한 방향만 retain
                for i in range(n):
                    for j in range(i + 1, n):  # i < j만 확인하면 중복 없음
                        if nng[i, j] == 1 and nng[j, i] == 1:
                            nng[j, i] = 0  # j→i 방향 제거
                graph = nng


            # Step 4: 현재 그래프 edge 추출 및 기록(K구현위해서)
            rows, cols, _ = find(graph)

            # 0이 아닌 엣지들만 순회
            for i, j in zip(rows, cols):
                all_edges.append((i, j))
                edge_to_level[(i,j)] = level
                used[i,j] = 1
                used[j,i] = 1
                        

        E = np.array(all_edges, dtype=int)  # (i, j) 형태의 edge 리스트로 변환
        E += 1  # R과 호환되도록 1-based 인덱싱으로 변환
        return E

    ############# G 통계량 계산 #############   
    def compute_G(self, window_indices, eliminate=False, min_window_length=None):
        """
        주어진 window(segments 묶음)로부터 G 통계량 계산
        Parameters:
            window: List[List[int]]
                - 실제 값을 담고 있는 window
                (예: [[1], [2], [3,4,5], [6], [7]])
            window_indices: List[List[int]]
                - segment의 index 리스트
                (예: [[0], [1], [2,3,4], [5], [6]])
        Returns:
            G: float
                - statistic 값 
        중요한점은 Segment마다 index가 바뀌어야함(바뀌는중)
        """

        #Step 1: t, n1, n2, cp, window 구성
        # t: group1의 마지막 인덱스, n1: group1의 관측값 수, n2: group2의 관측값 수
        # window_indices_flat: 모든 segment의 인덱스를 1차원으로 flatten

        t, n1, n2 = self.get_t(window_indices)  # t, n1, n2 계산 
        n = n1 + n2
        window_indices_flat = list(chain.from_iterable(window_indices))  # flatten
        window_data = self.observations[window_indices_flat, :]
        cp = window_indices_flat[t]

        if self._is_first_step:
            # 첫 스텝: 원본 데이터(raw data)로 그래프 생성
            observation = window_data
        else:
            # 이후 스텝: 임베딩 공간으로 변환 후 그래프 생성
            observation = self._embed_data(window_data)

        #Step 2: K-MST 그래프 생성, Edge 리스트 E 생성
        E = self.build_graph(observations=observation, graph_type='mst', eliminate=eliminate, min_window_length=min_window_length)


        #Step 3: R gseg패키지 실행해서 G 통계량 계산
        St = compute_g_stat_from_graph(n, E, t)

        if not self.isnan:
            #error handling
            St = np.nan_to_num(St, nan=0, posinf=0, neginf=0)
        else: 
            St = np.nan_to_num(St, nan=np.inf, posinf=np.inf, neginf=np.inf) # NaN을 inf로 변환

        return CPStat(cp=cp, G=St)


    def merge_segments_sequential(self,
            all_windows: List[List[List[int]]],
            all_stats: List[CPStat],
            candidate_indices: List[int]
            ) -> List[List[int]]:
            """
            규칙에 따라 순차적으로, 겹침 없이 윈도우를 병합합니다.

            1. 병합 후보들을 G 통계량 오름차순으로 정렬합니다.
            2. G값이 가장 낮은 후보부터 순서대로 병합을 시도합니다.
            3. 한 번 병합에 사용된 기본 세그먼트는 해당 단계의 다른 병합에 재사용될 수 없습니다.
            
            Args:
                all_windows: 현재 단계의 모든 윈도우 리스트.
                all_stats: 현재 단계의 모든 CPStat 객체 리스트.
                candidate_indices: 병합 후보가 되는 윈도우들의 인덱스 리스트 (하위 10% 등).

            Returns:
                다음 단계로 넘어갈 새로운 세그먼트 리스트.
            """
            
            # 1. 병합 후보를 (G값, 윈도우) 형태로 만들어 오름차순 정렬
            merge_candidates = []
            for i in candidate_indices:
                # 튜플: (G 통계량, 병합될 윈도우 구조)
                merge_candidates.append((all_stats[i].G, all_windows[i]))
            
            # G값을 기준으로 오름차순 정렬
            merge_candidates.sort(key=lambda x: x[0])

            # 2. 현재 단계의 모든 고유한 기본(base) 세그먼트를 set으로 준비
            all_base_segments_set: Set[Tuple[int, ...]] = set()
            for window in all_windows:
                for segment in window:
                    all_base_segments_set.add(tuple(sorted(segment)))

            # 3. 순차적 병합 수행
            # 이번 단계에서 병합에 사용되어 사라질 세그먼트들을 추적
            segments_to_remove: Set[Tuple[int, ...]] = set()
            # 병합의 결과로 새로 생성될 세그먼트들을 저장
            newly_created_segments: List[List[int]] = []

            for _, window_to_merge in merge_candidates:
                # 현재 후보 윈도우를 구성하는 기본 세그먼트들 (튜플 형태)
                candidate_base_segs = {tuple(sorted(seg)) for seg in window_to_merge}

                # 이 후보의 세그먼트 중 하나라도 이미 병합에 사용되었다면 건너뛰기
                # isdisjoint: 두 set이 겹치는 원소가 하나도 없으면 True 반환
                if not candidate_base_segs.isdisjoint(segments_to_remove):
                    continue

                # --- 유효한 병합이므로 수행 ---
                # 1) 이 후보의 기본 세그먼트들을 '제거 목록'에 추가하여 이후 재사용 방지
                segments_to_remove.update(candidate_base_segs)
                
                # 2) 실제로 세그먼트들을 하나로 합쳐서 '신규 생성 목록'에 추가
                merged_list = [node for seg in window_to_merge for node in seg]
                newly_created_segments.append(sorted(list(set(merged_list))))

            # 4. 최종 세그먼트 리스트 조합
            # 기존의 모든 세그먼트에서, 이번에 병합에 사용된 세그먼트들을 뺀다
            final_segments_set = all_base_segments_set - segments_to_remove
            
            # set을 다시 list of lists로 변환
            final_segments = [list(seg) for seg in final_segments_set]
            # 새로 생성된 병합 세그먼트들을 추가
            final_segments.extend(newly_created_segments)
            
            # 시작점을 기준으로 정렬하여 반환
            return sorted(final_segments, key=lambda x: x[0])
    

    def select_cps_by_connected_rule(self) -> List[int]:
        if not self.merge_history:
            return []

        # 1. G > λ 조건을 만족하는 모든 'Seed CP' 찾기
        seed_cps = set()
        for stats_list in self.merge_history:
            for stat in stats_list:
                if stat.G > self.CRITICAL:
                    seed_cps.add(stat.cp)

        final_cps = set(seed_cps)              
        return sorted(list(final_cps))
    
    def select_merge_indices_rank(
        self,
        cp_stats_now: List, # CPStat 타입으로 가정
        merge_percentile: float | None = None,
    ) -> List[int]:
        """
        G-값의 '순위'를 기준으로 병합할 cp의 '인덱스'를 고른다.
        (동점 처리 규칙 추가: 커트라인에 해당하는 동점 값은 모두 포함)

        Parameters
        ----------
        cp_stats_now : List[CPStat]
            이번 step의 CP-통계량
        merge_percentile : float, optional
            0~1 사이 : 하위 몇 %를 병합할지. (예: 0.20 → 하위 20 %)
            None일 경우 기본값(self.DEFAULT_MERGE_PERCENTILE)을 사용합니다.

        Returns
        -------
        List[int]
            병합 대상 change-point의 원본 리스트 인덱스 (오름차순)
        """
        if not cp_stats_now:
            return []

        # -------- 0) 매개변수 유효성 검사 및 기본값 설정 --------
        if merge_percentile is None:
            merge_percentile = self.merge_percentile
        
        # -------- 1) 몇 개를 자를지 결정 ---------------------------------
        n = len(cp_stats_now)
        k = ceil(n * merge_percentile) if n > 0 else 0
        if k == 0:
            return []
        
        # k가 전체 개수보다 클 수 없도록 보정
        k = min(k, n)

        # -------- 2) G 기준 전체 정렬 및 커트라인(G-값) 결정 --------
        # enumerate를 사용하여 (원본 인덱스, CPStat 객체) 쌍을 만듭니다.
        # G-값을 기준으로 전체 리스트를 오름차순 정렬합니다.
        sorted_stats_with_indices = sorted(
            enumerate(cp_stats_now), 
            key=lambda x: x[1].G
        )

        # k번째 항목의 G-값을 커트라인으로 설정합니다. (0-based index이므로 k-1)
        cutoff_g_value = sorted_stats_with_indices[k-1][1].G

        # -------- 3) 커트라인과 동점자를 모두 포함하여 후보 선정 --------
        # G-값이 커트라인보다 작거나 같은 모든 항목을 후보로 선택합니다.
        # sorted()는 안정적(stable)이므로, G-값이 같을 경우 원래 순서를 유지합니다.
        candidates_with_stats = []
        for index, stat in sorted_stats_with_indices:
            if stat.G <= cutoff_g_value:
                candidates_with_stats.append((index, stat))
            else:
                # 리스트가 정렬되어 있으므로, 커트라인보다 큰 값이 나오면 중단해도 됩니다.
                break

        # -------- 4) Full-tree 여부/CRITICAL 값 반영 ----------------------
        if not self.isFullTree:
            final_candidates = [
                index for index, stat in candidates_with_stats
                if stat.G < self.CRITICAL
            ]
        else:
            final_candidates = [index for index, stat in candidates_with_stats]

        # -------- 5) 최종 결과 반환 --------------------------------------
        return sorted(final_candidates)

    def select_merge_indices_local_min(self, cp_stats_now: List[CPStat], window_size: int = 3) -> List[int]:
        """
        슬라이딩 윈도우를 사용하여 G-통계량의 지역 최소값(local minima)에 해당하는 인덱스를 선택합니다.

        Args:
            cp_stats_now (List[CPStat]): 현재 병합 단계의 CPStat 객체 리스트.
            window_size (int): 최소값을 찾기 위한 슬라이딩 윈도우의 크기.

        Returns:
            List[int]: 병합 대상으로 선택된 CPStat 객체의 원본 인덱스 리스트 (정렬된 유니크 값).
        """
        # 입력값이 비어있거나 window_size가 부적절한 경우 빈 리스트 반환
        if not cp_stats_now or window_size <= 0:
            return []

        num_stats = len(cp_stats_now)
        # 윈도우 크기가 전체 리스트 크기보다 클 경우, 전체에서 최소값을 찾도록 조정
        if window_size > num_stats:
            window_size = num_stats
            
        local_min_indices = set()

        # 슬라이딩 윈도우를 이동시키며 반복
        for i in range(num_stats - window_size + 1):
            window = cp_stats_now[i : i + window_size]

            # 현재 윈도우에서 G-통계량의 최소값을 찾음
            min_g_in_window = min(stat.G for stat in window)

            # 윈도우 내에서 최소값과 일치하는 모든 CPStat을 찾음
            for j, stat in enumerate(window):
                if stat.G == min_g_in_window:
                    # 원본 리스트에서의 인덱스(i + j)를 결과 집합에 추가
                    local_min_indices.add(i + j)

        # 정렬된 리스트로 변환하여 반환
        return sorted(list(local_min_indices))
    
    ############# Unmerge 로직 #############

    def _calculate_g_for_unmerge(self, segments_to_test: List[List[int]], min_len: None) -> float:
        """
        Unmerge 로직을 위한 G-통계량 계산 헬퍼 메소드.
        세그먼트 인덱스 리스트를 받아 G-통계량 값을 반환한다.
        Parameters:
            segments_to_test: List[List[int]]
                - G-통계량을 계산할 세그먼트 인덱스 리스트
            min_len: int
                - k-MST 그래프 생성 시 고려할 최소 윈도우 길이
        Returns:
            G: float
                - 계산된 G-통계량 값
        """
        # 데이터가 없는 빈 세그먼트는 계산에서 제외
        valid_segments = [seg for seg in segments_to_test if seg]
        if len(valid_segments) < 2:
            return 0.0 if not self.isnan else np.inf

        # compute_G 메소드에 필요한 형태로 데이터 가공
        window_indices_flat = list(chain.from_iterable(valid_segments))
        window_data = [self.observations[i] for i in window_indices_flat]
        
        # compute_G는 CPStat 객체를 반환하므로 .G 속성을 추출
        stat = self.compute_G(valid_segments, min_window_length=min_len)
        return stat.G

    def _perform_unmerge_pass(self, seg_idx: List[List[int]]) -> List[List[int]]:
        """
        Unmerge 로직을 안정화될 때까지 반복 수행.
        분리/재병합 효용 점수가 양수인 모든 세그먼트에 대해, 점수가 가장 높은 순으로 순차적으로 unmerge를 진행한다.
        한 번의 unmerge 후 변경된 세그먼트 구조를 즉시 반영하여 다시 최적의 unmerge 대상을 찾는다.
        """
        # 안정화될 때까지 루프 반복
        while True:
            unmerge_candidates = []
            
            # 1. 현재 seg_idx 기준으로 모든 unmerge 후보와 점수 계산
            #    양 옆에 이웃이 있는 세그먼트(인덱스 1부터 N-2까지)를 순회
            for i in range(1, len(seg_idx) - 1):
                seg_A, seg_B, seg_C = seg_idx[i-1], seg_idx[i], seg_idx[i+1]
                
                if len(seg_B) < 2: continue

                split_point = len(seg_B) // 2
                seg_B1, seg_B2 = seg_B[:split_point], seg_B[split_point:]
                
                # min_len 계산
                min_len = min(len(seg_A) + len(seg_B1), len(seg_B2) + len(seg_C), len(seg_B1) + len(seg_B2))
                # min_len = None
                # G-통계량 계산
                g_keep = self._calculate_g_for_unmerge([seg_B1, seg_B2], min_len)
                g_split_left = self._calculate_g_for_unmerge([seg_A, seg_B1], min_len)
                g_split_right = self._calculate_g_for_unmerge([seg_B2, seg_C], min_len)
                
                # "min of difference" 규칙을 점수로 사용
                score_left = g_keep - g_split_left
                score_right = g_keep - g_split_right
                unmerge_score = min(score_left, score_right)
                
                if unmerge_score > 0:
                    unmerge_candidates.append((unmerge_score, i))
            
            # 2. Unmerge 대상이 더 이상 없는지 확인
            if not unmerge_candidates:
                # 안정화 상태에 도달했으므로 루프 종료
                break
                
            # 3. 점수가 가장 높은 최선 대상을 찾아 Unmerge 실행
            unmerge_candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_idx = unmerge_candidates[0]
            
            # 새로운 세그먼트 리스트 생성
            seg_A, seg_B, seg_C = seg_idx[best_idx - 1], seg_idx[best_idx], seg_idx[best_idx + 1]
            
            split_point = len(seg_B) // 2
            new_seg_left = seg_A + seg_B[:split_point]
            new_seg_right = seg_B[split_point:] + seg_C
            
            # seg_idx를 업데이트하여 다음 루프 이터레이션에 반영
            seg_idx = seg_idx[:best_idx-1] + [new_seg_left, new_seg_right] + seg_idx[best_idx+2:]
            
            print(f"Unmerge performed at index {best_idx} -> new indices {best_idx-1}, {best_idx}. Score: {best_score:.4f}")

        # 모든 unmerge가 완료되어 안정화된 seg_idx 반환
        return seg_idx
    
    def set_encoder(self, encoder: torch.nn.Module):
            """외부(Trainer)에서 업데이트된 인코더를 주입받기 위한 메소드."""
            self.encoder = encoder
            self.encoder.to(self.device)
            print("[gBottomup] New encoder received and updated.")

    def _embed_data(self, window_data: np.ndarray) -> np.ndarray:
        """주어진 데이터를 현재 인코더를 사용해 임베딩 공간으로 변환합니다."""
        self.encoder.eval()
        with torch.no_grad():
            tensor_data = torch.from_numpy(window_data).float().unsqueeze(0).to(self.device)
            embedded_tensor = self.encoder(tensor_data)
            return embedded_tensor.squeeze(0).cpu().numpy()

    def fit(self, data: np.ndarray):
        self.observations = data
        n, _ = data.shape
        self._is_first_step = True
        
        seg_idx = [list(range(i, min(i + self.min_obs, n))) for i in range(0, n, self.min_obs)]
        self.seg_history.append([s.copy() for s in seg_idx])

        while len(seg_idx) > 1:
            window_indices = self.sliding_windows(seg_idx)
            if not window_indices: break

            min_window_length = min(sum(len(seg) for seg in w) for w in window_indices)
            cp_stats_now = [self.compute_G(win_idx, False, min_window_length) for win_idx in window_indices]
            self.merge_history.append(cp_stats_now)

            candidate_indices = self.select_merge_indices_rank(cp_stats_now)
            if not candidate_indices: break
            
            # 긍정 쌍을 yield하고 외부의 신호를 기다림
            positive_pairs_this_step = [window_indices[i] for i in candidate_indices]
            new_encoder = yield positive_pairs_this_step
            if new_encoder is not None:
                self.set_encoder(new_encoder)

            # 세그먼트 병합 및 다음 스텝 준비
            seg_idx = self.merge_segments_sequential(window_indices, cp_stats_now, candidate_indices)
            self.seg_history.append([s.copy() for s in seg_idx])
            self._is_first_step = False

    def detect(self, data: np.ndarray, selection_method: str = 'backward') -> List[int]:
        """
        최종 변화점 탐지 및 선택을 수행하는 메소드.

        학습된 인코더를 사용하여 Bottom-up 프로세스를 끝까지 실행하고,
        ModelSelector를 통해 최적의 변화점을 반환합니다.

        Args:
            data (np.ndarray): 탐지를 수행할 데이터.
            selection_method (str): 'forward', 'backward', 'stepwise' 중 모델 선택 방법.

        Returns:
            List[int]: 최종적으로 선택된 변화점 리스트.
        """
        print(f"\n--- Starting final detection with '{selection_method}' selection ---")
        
        # 1. fit 제너레이터를 학습 없이(send(None)) 끝까지 실행하여 merge_history를 채웁니다.
        detection_generator = self.fit(data)
        try:
            while True:
                next(detection_generator)
                detection_generator.send(None) # 학습을 진행하지 않음
        except StopIteration:
            print("Full merge history collected.")

        # 2. 내부적으로 ModelSelector를 생성하고 실행합니다.
        selector = ModelSelector(self)

        if self.alpha is not None: 
            cp_candidate = self.select_cps_by_connected_rule()
            G_value = None
        else: 
            cp_estimate = []
            G_value = None

        cp_candidate = sorted((list(set(cp_candidate))))  # 혹시 모를 중복 제거를 위해 set으로 변환
        cp_cand_ori = cp_candidate.copy()
        merge_summary_dict = {
            step + 1: [(stat.cp, round(stat.G, 4)) for stat in stats_list]
            for step, stats_list in enumerate(self.merge_history)
            }

        if selection_method == 'forward':
        # Forward Selection with ep-BIC
            cp_estimate, G_value = selector.forward_selection(candidate_cps=cp_candidate)
        elif selection_method == 'backward':
            # Backward Elimination. 초기 후보는 마지막 merge 단계의 모든 cp
            cp_estimate, G_value = selector.backward_elimination(candidate_cps=cp_candidate)
        elif selection_method == 'both' or selection_method == 'stepwise':
            cp_estimate, G_value = selector.stepwise_elimination(candidate_cps=cp_candidate)
        elif selection_method == 'topk':
            cp_estimate_now = set()
            # Top-K Selection. k는 self.num_cp
            if self.num_cp is None or self.num_cp <= 0:
                raise ValueError("num_cp must be a positive integer for 'topk' elimination.")
            else:
                cp_estimate_now = set()
                for stats_list in self.merge_history[::-1]:  # 마지막 단계부터 거꾸로
                    topk_stats = [stat.cp for stat in stats_list]
                    cp_estimate_now.update(topk_stats)
                    if len(cp_estimate_now) >= self.num_cp:
                        break
            cp_estimate = sorted(list(cp_estimate_now))
            G_value = None
        else:
            cp_estimate= cp_estimate
            G_value = None

        cp_candidate = cp_cand_ori.copy()
        cp_estimate = [cp + self.start_with for cp in cp_estimate]
        cp_candidate = [cp + self.start_with for cp in cp_candidate]
        return cp_estimate, G_value, cp_candidate