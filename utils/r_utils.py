import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
import numpy as np
import os
import contextlib

# R 객체를 저장할 글로벌 변수
R_OBJECTS = {}

def init_r_packages():
    """
    R 환경을 초기화하고 필요한 패키지를 로드하여 글로벌 변수에 저장합니다.
    이 함수는 프로그램 시작 시 딱 한 번만 호출되어야 합니다.
    """
    if R_OBJECTS: # 이미 초기화되었으면 실행하지 않음
        return

    print("Initializing R environment and loading packages...")
    with localconverter(ro.default_converter + numpy2ri.converter):
        R_OBJECTS['base'] = importr('base')
        R_OBJECTS['stats'] = importr('stats')
        R_OBJECTS['ade4'] = importr('ade4')
        R_OBJECTS['gSeg'] = importr('gSeg')
    print("R packages loaded successfully.")

def compute_g_stat_from_graph(n_obs: int, E, t_split: int) -> float:
    """
    주어진 그래프 엣지로부터 G-통계량을 계산하는 R 헬퍼 함수.
    
    Args:
        n_obs (int): 관측치의 총 개수.
        edges (np.ndarray): (num_edges, 2) 형태의 1-based 엣지 리스트.
        t_split (int): 0-based 분할점 인덱스.

    Returns:
        float: 계산된 G-통계량.
    """
    if 'gSeg' not in R_OBJECTS:
        raise RuntimeError("R environment is not initialized. Call init_r_packages() first.")

    gSeg = R_OBJECTS['gSeg']

    
    with localconverter(ro.default_converter + numpy2ri.converter) as cv:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                res = gSeg.gseg1(n_obs, E,
                                statistics="g",
                                pval_appr=False,
                                skew_corr=False)
                
                
        slst_r = res[0][0][2]
        St = slst_r[t_split]
    return St