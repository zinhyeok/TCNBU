# model_selector.py

import numpy as np
from itertools import combinations
from typing import List, Tuple, Optional
from scipy.stats import chi2
# 타입 힌팅을 위해 gBottomup 클래스를 임포트합니다.
# 실제 실행 시 순환 참조 문제가 발생하지 않도록 TYPE_CHECKING을 사용할 수 있습니다.
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from gBottomup import gBottomup, CPStat

class ModelSelector:
    """
    gBottomup 알고리즘의 결과를 바탕으로 최적의 변경점(change-point) 조합을 선택하는 클래스.
    Forward, Backward, Stepwise Elimination 전략을 제공합니다.
    """
    def __init__(self, g_bottomup_instance):
        """
        ModelSelector를 초기화합니다.

        Args:
            g_bottomup_instance: fit() 메서드가 완료된 gBottomup 클래스의 인스턴스.
                                 이 인스턴스를 통해 compute_G, merge_history 등의 정보에 접근합니다.
        """
        self.model = g_bottomup_instance
        self.data = self.model.observations # fit 메서드에서 사용된 데이터에 접근
        

    def _find_segment_bounds(self, cp: int, cps: List[int], n: int) -> Tuple[int, int]:
        """
        주어진 change-point(cp)가 속한 세그먼트의 경계 (L, R)를 반환합니다.
        """
        all_cps = sorted(set([cp] + cps + [-1, n - 1]))
        idx = all_cps.index(cp)
        L = all_cps[idx - 1] + 1 if idx > 0 else 0
        R = all_cps[idx + 1] if idx + 1 < len(all_cps) else n - 1
        return L, R
    
    def select_cps_by_alpha(self, all_cp_estimates_, alpha,
                             # --- 구간 확장을 위한 파라미터 추가 ---
                             expansion_step=10, 
                             max_expansion_steps=3):
        """
        주어진 cp_estimate가 실제 cp인지 alpha를 이용해서 검정합니다.
        초기 검정에서 유의하지 않으면, 구간을 확장하며 추가 검정을 수행합니다.
        """
        cp_lst = [cp for cp, freq in all_cp_estimates_]
        n = self.data.shape[0]
        critical_value = chi2.ppf(1 - alpha, 2)
        selected_cps = []

        for cp, freq in all_cp_estimates_:
            # --- 1. 초기 구간 설정 및 검정 (기존 코드와 유사) ---
            L, R = self._find_segment_bounds(cp, cp_lst, n)
            
            # gBottomup 인스턴스의 compute_G 메서드 호출
            window = [self.data[L : cp + 1], self.data[cp + 1 : R + 1]]
            idx = [list(range(L, cp + 1)), list(range(cp + 1, R + 1))]
            
            cp_stat = self.model.compute_G(
                window, idx,
                eliminate=True,
                min_window_length=R - L + 1 # 현재 윈도우 길이 전달
            )

            # --- 2. 유의성 플래그 및 구간 확장 로직 추가 ---
            is_significant = cp_stat.G > critical_value

            if not is_significant:
                # --- 유의하지 않을 경우에만 구간 확장 루프 실행 ---
                current_L, current_R = L, R
                
                for _ in range(max_expansion_steps):
                    # 구간 확장 (데이터 경계를 넘지 않도록 max/min 사용)
                    current_L = max(0, current_L - expansion_step)
                    current_R = min(n - 1, current_R + expansion_step)
                    
                    # 더 이상 확장할 수 없으면 중단
                    if current_L == 0 and current_R == n - 1:
                        break
                    
                    # 확장된 구간으로 재검정
                    window_expanded = [self.data[current_L : cp + 1], self.data[cp + 1 : current_R + 1]]
                    idx_expanded = [list(range(current_L, cp + 1)), list(range(cp + 1, current_R + 1))]

                    cp_stat_expanded = self.model.compute_G(
                        window_expanded, idx_expanded,
                        eliminate=True,
                        min_window_length=current_R - current_L + 1
                    )
                    
                    if cp_stat_expanded.G > critical_value:
                        is_significant = True
                        break # 유의하게 나오면 더 이상 확장할 필요 없음

            # --- 3. 최종 결정 ---
            if is_significant:
                selected_cps.append(cp)

        return sorted(selected_cps)

    def total_score(self, cp_list: list[int]) -> float:
        """주어진 cp 리스트에 대한 페널티 포함 목적 함수 점수를 계산합니다."""
        if not cp_list:
            return 0.0
        n = self.data.shape[0]
        cp_list = sorted(cp_list)
        
        # min_len 계산 시 cp_list가 비어있으면 에러가 발생하므로 위에서 처리
        min_len = min((R - L + 1) for c in cp_list for L, R in [self._find_segment_bounds(c, cp_list, n)])
        
        sum_G = 0.0
        for cp in cp_list:
            L, R = self._find_segment_bounds(cp, cp_list, n)
            window = [self.data[L : cp + 1], self.data[cp + 1 : R + 1]]
            idx = [list(range(L, cp + 1)), list(range(cp + 1, R + 1))]

            # gBottomup 인스턴스의 compute_G 메서드 호출
            cp_stat = self.model.compute_G(
                idx,
                eliminate=True,
                min_window_length=min_len
            )
            sum_G += cp_stat.G

        m = len(cp_list)
        boundaries = np.array([0] + sorted(cp_list) + [n])
        boundaries = sorted(list(set(boundaries)))  # 중복 제거
        segment_lengths = np.diff(boundaries)
        length_reward_term = np.sum(np.log(segment_lengths))
        
        # gBottomup 인스턴스의 c 상수 사용
        penalty = (self.model.c * m * np.log(n) + length_reward_term)
        return sum_G - penalty


    def forward_selection(self, candidate_cps: Optional[List[int]] = None) -> Tuple[List[int], float]:
        """
        Forward Selection을 수행하여 BIC 점수를 최대화하는 최적의 변경점 조합을 찾습니다.

        - candidate_cps가 제공되면, 해당 후보군 내에서 표준 Forward Selection을 수행합니다.
        - candidate_cps가 제공되지 않으면(None 또는 빈 리스트), 모델의 병합 이력을 기반으로
        계층적 Forward Selection을 수행합니다.

        Args:
            candidate_cps (Optional[List[int]], optional):
                고려할 변경점 후보 리스트. Defaults to None.

        Returns:
            Tuple[List[int], float]: 선택된 최적의 변경점 리스트와 그 때의 최고 점수.
        """
        # ==============================================================================
        # 시나리오 1: candidate_cps 리스트가 명시적으로 주어진 경우
        # ==============================================================================
        if candidate_cps:
            # 1. 변수 초기화
            selected_cps: List[int] = []
            remaining_cps = set(candidate_cps)
            
            # 2. 초기 점수 계산 (변경점이 하나도 없는 모델)
            best_score: float = self.total_score([])
            
            # 3. 더 이상 점수 향상이 없을 때까지 반복
            while True:
                best_candidate_in_step: Optional[int] = None
                best_score_in_step: float = best_score

                # 4. 남은 후보군을 하나씩 추가하며 최상의 후보 탐색
                for cp_to_add in remaining_cps:
                    test_cps = selected_cps + [cp_to_add]
                    current_score = self.total_score(test_cps)

                    if current_score > best_score_in_step:
                        best_score_in_step = current_score
                        best_candidate_in_step = cp_to_add
                
                # 5. 최적 후보를 모델에 추가할지 결정
                if best_candidate_in_step is not None:
                    best_score = best_score_in_step
                    selected_cps.append(best_candidate_in_step)
                    remaining_cps.remove(best_candidate_in_step)
                else:
                    # 점수 향상이 없으면 종료
                    break
            
            # 6. 최종 결과 반환
            return sorted(selected_cps), best_score

        # # ==============================================================================
        # # 시나리오 2: candidate_cps가 주어지지 않은 경우 (None 또는 빈 리스트)
        # # ==============================================================================
        # else:
        #     print("INFO: No candidate_cps provided. Running hierarchical forward selection.")
        #     patience = 2
        #     selected_cps: List[int] = []
        #     current_best_cps: List[int] = []
        #     best_score: float = self.total_score([])
            
        #     cumulative_candidates = set()
        #     not_improved_count = 0

        #     # merge_history를 역순으로 탐색
        #     for step in range(len(self.model.merge_history) - 1, len(self.model.merge_history) // 2, -1):
        #         # 현재 step의 후보군을 누적 후보군에 추가
        #         step_candidates = set(stat.cp for stat in self.model.merge_history[step])
        #         cumulative_candidates.update(step_candidates)
                
        #         # 새로 테스트할 후보들 = (지금까지 누적된 모든 후보) - (이전 step의 최적 조합)
        #         new_candidates_to_test = list(cumulative_candidates - set(current_best_cps))
                
        #         best_score_in_step = self.total_score(current_best_cps)
        #         best_next_cps_in_step = current_best_cps

        #         if new_candidates_to_test:
        #             # 새로 추가된 후보들을 하나씩만 더해보며 greedy하게 탐색
        #             for combo in combinations(new_candidates_to_test, 1):
        #                 test_cp_lst = sorted(list(set(current_best_cps).union(combo)))
        #                 new_score = self.total_score(test_cp_lst)

        #                 if new_score > best_score_in_step:
        #                     best_score_in_step = new_score
        #                     best_next_cps_in_step = test_cp_lst
                
        #         current_best_cps = best_next_cps_in_step
                
        #         # 전체 최고 점수와 비교하여 조기 종료 여부 결정
        #         if best_score_in_step > best_score:
        #             best_score = best_score_in_step
        #             selected_cps = current_best_cps
        #             not_improved_count = 0
        #         else:
        #             not_improved_count += 1
                
        #         if not_improved_count >= patience:
        #             break
                    
        #     return selected_cps, best_score
    
    def backward_elimination(self, candidate_cps: list[int] | None = None, min_cpnum: int = 0) -> tuple[list[int], float]:
        """Backward Elimination을 수행합니다."""
        n = self.data.shape[0]

        cur_cps = candidate_cps

        # 초기 점수 계산
        cur_score = self.total_score(cur_cps)
        
        # 최종 결과를 저장할 변수 (초기 상태로 초기화)
        best_cps, best_score = cur_cps[:], cur_score

        # 변화점 수가 min_cpnum보다 많을 동안 반복
        while len(cur_cps) > min_cpnum:
            
            # 이번 단계에서 제거할 최적의 변화점과 그 때의 점수를 찾음
            cp_to_remove, new_score = self._find_best_cp_to_remove(cur_cps, n)
            
            # 제거할 변화점을 찾지 못했다면 루프 중단
            if cp_to_remove is None:
                break
                
            # 변화점 리스트에서 해당 cp 제거
            cur_cps.remove(cp_to_remove)
            cur_score = new_score

            # 현재 점수가 역대 최고 점수보다 좋으면 결과 업데이트
            if cur_score > best_score:
                best_score = cur_score
                best_cps = cur_cps[:] # 리스트 복사
            else:
                break

        return best_cps, best_score
    
    def _find_best_cp_to_remove(self, cp_list: list[int], n: int) -> tuple[int | None, float]:
        """제거했을 때 점수가 가장 높은 (또는 손실이 가장 적은) 변화점을 찾습니다."""
        base_score = self.total_score(cp_list)
        best_gain = 0
        cp_to_remove = None
        best_cand_score = base_score

        for cp in cp_list:
            cand = [c for c in cp_list if c != cp]
            segs = [0] + cand + [n]
            

            cand_score = self.total_score(cand)
            gain = cand_score - base_score # 점수 변화량
            
            if gain > best_gain:
                best_gain = gain
                cp_to_remove = cp
                best_cand_score = cand_score

        return cp_to_remove, best_cand_score
    
    def _find_best_cp_to_add(self, current_cps: list[int], candidates_to_add: set[int]) -> tuple[int | None, float]:
        """추가했을 때 점수가 가장 높은 변화점을 찾습니다."""
        base_score = self.total_score(current_cps)
        best_gain = 0
        cp_to_add = None
        best_cand_score = base_score

        for cp in candidates_to_add:
            cand = sorted(current_cps + [cp])
            
            cand_score = self.total_score(cand)
            gain = cand_score - base_score # 점수 변화량
            
            if gain > best_gain:
                best_gain = gain
                cp_to_add = cp
                best_cand_score = cand_score

        return cp_to_add, best_cand_score

    def stepwise_elimination(self, candidate_cps: list[int] | None = None) -> tuple[list[int], float]:
        """
        단계적 소거법(Stepwise Selection)을 수행하여 최적의 변경점 조합을 찾습니다.

        이 구현은 다음 방식으로 동작합니다:
        1. 모든 후보를 포함한 전체 모델로 초기화합니다.
        2. 각 단계에서 다음 두 가지를 모두 탐색합니다.
        a. (Backward) 현재 모델에서 변경점 하나를 '제거'하는 최적의 경우
        b. (Forward) 현재 모델에 변경점 하나를 '추가'하는 최적의 경우
        3. 위 a, b 두 경우 중 점수를 더 크게 향상시키는 단 하나의 행동을 선택하여 모델을 업데이트합니다.
        4. 모델에 더 이상 변화(점수 향상)가 없을 때까지 위 과정을 반복합니다.

        Args:
            cps (list[int] | None, optional): 전체 변경점 후보 리스트.

        Returns:
            tuple[list[int], float]: 찾은 최적의 변경점 리스트와 그 때의 최고 점수.
        """
        # 1. 후보군 설정
        if candidate_cps is None:
            all_candidates = set(st.cp for step_hist in self.model.merge_history for st in step_hist)
        else:
            all_candidates = set(candidate_cps)

        # 2. 변수 초기화 
        current_cps = sorted(list(all_candidates))
        best_score = self.total_score(current_cps)

        while True:
            # 이번 이터레이션에서 찾은 최적의 행동 정보를 저장할 변수
            # (action_type: 'remove' or 'add', cp_to_change: 대상 cp, next_cps: 변경 후 cp 리스트)
            best_action_in_iter = {'action_type': None, 'cp_to_change': None, 'score': best_score, 'next_cps': current_cps}

            # --- 3. Backward 탐색: 제거할 최적의 후보 찾기 ---
            for cp_to_remove in current_cps:
                cand_cps = [c for c in current_cps if c != cp_to_remove]
                cand_score = self.total_score(cand_cps)

                if cand_score > best_action_in_iter['score']:
                    best_action_in_iter['score'] = cand_score
                    best_action_in_iter['action_type'] = 'remove'
                    best_action_in_iter['cp_to_change'] = cp_to_remove
                    best_action_in_iter['next_cps'] = cand_cps
            
            # --- 4. Forward 탐색: 추가할 최적의 후보 찾기 ---
            candidates_to_add = all_candidates - set(current_cps)
            for cp_to_add in candidates_to_add:
                cand_cps = sorted(current_cps + [cp_to_add])
                cand_score = self.total_score(cand_cps)
                
                # 현재까지 찾은 최적의 행동(제거 포함)보다 더 나은지 확인
                if cand_score > best_action_in_iter['score']:
                    best_action_in_iter['score'] = cand_score
                    best_action_in_iter['action_type'] = 'add'
                    best_action_in_iter['cp_to_change'] = cp_to_add
                    best_action_in_iter['next_cps'] = cand_cps

            # --- 5. 모델 업데이트 및 종료 조건 확인 ---
            # 이번 이터레이션에서 점수를 향상시키는 행동을 찾았다면 모델 업데이트
            if best_action_in_iter['action_type'] is not None:
                current_cps = best_action_in_iter['next_cps']
                best_score = best_action_in_iter['score']
                # print(f"Action: {best_action_in_iter['action_type']} {best_action_in_iter['cp_to_change']}, New Score: {best_score:.2f}") # 디버깅용
            else:
                # 점수 향상이 없었으므로 더 이상 진행하지 않고 종료
                break
                
        return current_cps, best_score