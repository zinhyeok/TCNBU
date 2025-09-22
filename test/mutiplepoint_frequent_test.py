import numpy as np
import json
import subprocess
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
import os
from test.logger_setup import setup_module_logger
from model.gBottomup_R import gBottomup
from itertools import product
from typing import Dict, List, Tuple
from sklearn.metrics import adjusted_rand_score
from data.dataGenerator import generate_multi_data
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.vectors import IntVector, FloatVector
import traceback
import contextlib
import psutil
import rpy2.robjects as robjects
import gc

# 지표
# Adjusted Rand Index (ARI)
# n_cps: 예측한 change-point의 개수
# cp 위치 찍기(이건 향후 진행)

# === Logging setup ===
from test.logger_setup import setup_module_logger

logger_multi = setup_module_logger("project.multiGraphstep")
logger_multi.info("Multi-point frequent test started")
project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(project_root, ".."))  # utilize/ 기준이면 한 단계 위로

# === Experiment Runner ===
def setup_r_environment():
    """
    gSegMulti와 의존성을 모두 포함하는 R 환경을 설정하고,
    gSegMulti 함수 객체를 반환합니다.
    """
    r = robjects.r
    
    # R 스크립트들이 있는 디렉터리 경로
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # gSegMulti/R 디렉터리와 gSegMulti.R 파일이 있는 상위 디렉터리
    gseg_r_dir = os.path.join(base_dir, "gSegMulti", "R")
    gsegmulti_file = os.path.join(gseg_r_dir, "gSegMulti.R")

    # 1. gSegMulti의 의존성 파일들 먼저 로드 (gSBS, gBE 등)
    print("Sourcing R dependency files...")
    r_files = [os.path.join(gseg_r_dir, f) for f in os.listdir(gseg_r_dir) if f.endswith('.R')]
    for f in r_files:
        r.source(f)
    
    # 2. gSegMulti.R 파일 로드
    print(f"Sourcing main file: {gsegmulti_file}")
    r.source(gsegmulti_file)
    
    print("R sourcing complete.")
    
    # 3. R 전역 환경에서 gSegMulti 함수를 Python 객체로 가져오기
    gsegmulti_func = robjects.globalenv['gSegMulti']
    
    return gsegmulti_func

# 스크립트 시작 시 R 함수를 한 번만 로드합니다.
GSEGMULTI_R_FUNC = setup_r_environment()


def run_gsegMulti_rpy2(data_np, alpha=0.05, method='SBS', model_selection=True):
    """
    rpy2를 사용하여 gSegMulti를 실행하고 결과를 Python 객체로 반환합니다.
    (R 함수 호출을 변환 컨텍스트 안에서 실행하도록 수정)
    """
    try:
        with robjects.conversion.localconverter(
            robjects.default_converter + numpy2ri.converter
        ) as cv:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    res_r = GSEGMULTI_R_FUNC(data_np, search_type=method, alpha=alpha, model_selection=model_selection)

        # R 결과를 Python 객체로 변환
        if model_selection:
            # model_selection=TRUE일 때, 결과 리스트의 두 번째 원소($tauhat)가 최종 CP임
            # res_r[0]은 $tautilde (후보 CP), res_r[1]은 $tauhat (최종 CP)
            try:
                cps = res_r[1] 
            except (IndexError, TypeError):
                return []
        else:
            # model_selection=FALSE일 때, 결과 리스트는 $tautilde 하나만 있음
            try:
                cps = res_r[0]
            except (IndexError, TypeError):
                return []

        # R의 NA/NULL 값 처리 및 인덱스 변환
        if cps is robjects.NULL or not hasattr(cps, '__iter__'):
            return []
        
        # NA 값을 필터링하고 0-based 인덱스로 조정
        final_cps = [int(cp) - 1 for cp in cps]
        return final_cps

    except Exception as e:
        print(f"An error occurred during R execution: {e}")
        return []


def evaluate(tauhat, true_tau, tol=2):
    if tauhat is None:
        return 0
    # tauhat이 list인 경우, 첫 번째 요소를 사용
    if isinstance(tauhat, list):
        tauhat = tauhat[0]

    return int(abs(tauhat - true_tau) <= tol)

def count_matches(predicted, true_cps, tol=2):
    # predicted가 None이면 종료
    if predicted is None:
        return 0
    # predicted가 스칼라일 경우 리스트로 변환
    if isinstance(predicted, (int, np.integer, float, np.floating)):
        predicted = [predicted]
    elif not isinstance(predicted, (list, np.ndarray)):
        return 0
    if len(predicted) == 0:
        return 0

    # true_cps가 None이거나 정수 하나일 경우 리스트로 변환
    if true_cps is None:
        return 0
    if isinstance(true_cps, (int, np.integer, float, np.floating)):
        true_cps = [true_cps]
    elif not isinstance(true_cps, (list, np.ndarray)):
        return 0
    if len(true_cps) == 0:
        return 0

    matched = 0
    try:
        for p_cp in predicted:
            if any(abs(t_cp - p_cp) <= tol for t_cp in true_cps):
                matched += 1
    except Exception as e:
        return 0

    return min(matched, len(true_cps))

def count_false_positives(predicted, true_cps, tol=2):
    # 입력 유효성 검사
    if predicted is None or (isinstance(predicted, (list, np.ndarray)) and len(predicted) == 0):
        return 0
    if true_cps is None:
        true_cps = []
    
    # 스칼라 → 리스트 변환
    if isinstance(predicted, (int, np.integer, float, np.floating)):
        predicted = [predicted]
    elif not isinstance(predicted, (list, np.ndarray)):
        return 0
    if isinstance(true_cps, (int, np.integer, float, np.floating)):
        true_cps = [true_cps]
    elif not isinstance(true_cps, (list, np.ndarray)):
        return 0

    false_positives = 0
    try:
        for p_cp in predicted:
            if all(abs(t_cp - p_cp) > tol for t_cp in true_cps):
                false_positives += 1
    except Exception:
        return 0

    return false_positives


def _cps_to_labels(cps, n):
    """
    change-point 리스트 → 길이 n의 구간 레이블 벡터
    cps: None 또는 정수/리스트
    """
    if cps is None:
        cps = []
    if isinstance(cps, (int, np.integer, float, np.floating)):
        cps = [int(cps)]
    try:
        cps = sorted(int(x) for x in cps)
    except:
        cps = []

    # 경계 정의
    boundaries = [0] + cps + [n]
    labels = np.empty(n, dtype=int)
    for i in range(len(boundaries)-1):
        labels[boundaries[i]:boundaries[i+1]] = i
    return labels

def compute_ari(predicted, true_cps, n):
    """
    Adjusted Rand Index (ARI) between 예측·실제 cps
    - None 이면 한 덩어리(= change-point 없음)로 간주
    """
    true_labels = _cps_to_labels(true_cps, n)
    pred_labels = _cps_to_labels(predicted, n)
    return adjusted_rand_score(true_labels, pred_labels)


def calculate_set_metrics(true_taus, candidate_taus, tol):
    true_taus = sorted(list(true_taus)) if true_taus is not None else []
    candidate_taus = sorted(list(candidate_taus)) if candidate_taus is not None else []
    if not true_taus and not candidate_taus: return 1.0, 1.0, 1.0
    if not true_taus: return 1.0, 0.0, 0.0
    if not candidate_taus: return 0.0, 1.0, 0.0
    tp = min(len(true_taus), sum(1 for tt in true_taus if any(abs(tt - ct) <= tol for ct in candidate_taus)))
    fp = sum(1 for ct in candidate_taus if not any(abs(ct - tt) <= tol for tt in true_taus))
    fn = len(true_taus) - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, f1


def run_gBottomUp(model, data):
    cp_list, stat, cp_cand = model.fit(data)
    if not cp_list:
        return {"tauhat": [], "stat": [], "cp_candidate": []}
    # sorted_cp = [cp for cp, s in sorted(zip(cp_list, stat), key=lambda x: x[1], reverse=True)]
    # sorted_stat = sorted(stat, reverse=True)
    # cp_list = sorted_cp[0]
    return {"tauhat": cp_list, "stat": stat, "cp_candidate": cp_cand}


# --------------------------------------------------------------------------- #
# Model builder / caching
# --------------------------------------------------------------------------- #
def build_model(alpha: float, c: int, num_cp: int, pre_merge: bool, eliminate: str, min_obs:int,
                *,  timestamp: str) -> Tuple[str, gBottomup]:
    name = (f"alpha_{alpha}c_{c}_{eliminate}_minobs{min_obs}_numcp{num_cp}")
    gbottomup_params = {
        'num_cp': num_cp,
        'alpha': alpha,
        'min_obs': min_obs,
        'pre_merge': pre_merge,
        'merge_percentile': 0.01,
        'isFullTree': True,
        'c': c,
        'eliminate': eliminate
    }
    model = gBottomup(**gbottomup_params)
    return name, model


# === Experiment Runner for model list ===
def experiment_all_models(d, scenario, frequency, runs=100, tol=5, timestamp=None):
    # 전역 파라미터 설정
    globals()["d"] = d  # 필요 시 사용할 수 있도록
    # ---- scenario‑specific delta / sigma ---------------------------------- #

    base_dir = os.getcwd()
    input_csv = os.path.join(base_dir, "data.csv")
    output_json = os.path.join(base_dir, "z_result.json")


    # --- parameter grid -------------------- #
    alpha_R         = 0.05
    alpha          = [0.05]
    eliminate_lst  = ['backward']
    num_cp_lst = [3]  # num_cp는 gBottomup_test.py에서 설정
    c_lst          = [2]
    pre_merge_lst = [True]
    # min_obs_lst = [10]
    min_obs_lst = [10]

    grid = [g for g in product(alpha, c_lst, num_cp_lst, pre_merge_lst, eliminate_lst, min_obs_lst)]

    tol = tol


    # ---- 결과 저장을 위한 리스트 ------------------------------------------ #
    results = []
    model_bank: Dict[str, gBottomup] = dict(
        build_model(*g, timestamp=timestamp) for g in grid
    )
    model_namelst = list(model_bank.keys())
    del model_bank

    # 각 시나리오에 대해 반복
    process = psutil.Process(os.getpid())
    for seed in tqdm(range(runs), desc=f"Running multi-cp for d={d} scenario={scenario} frequency={frequency}"):
        
        # ---- build model bank ------------------------------------------------- #
        model_bank: Dict[str, gBottomup] = dict(
            build_model(*g, timestamp=timestamp) for g in grid
        )
        
        # R 메모리 사용량 확인
        mem_info = process.memory_info()
        print(f"\n[Seed {seed}] - Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
        r_gc = robjects.r['gc']
        
        # 데이터 생성
        data, true_cps = generate_multi_data(scenario=scenario, n=200, frequency=frequency, d=d, seed=seed)
        n_samples = data.shape[0]

        # S 기반 baseline 실행
        np.savetxt(input_csv, data, delimiter=",")
        tauhat_data = run_gsegMulti_rpy2(data, alpha=alpha_R, method='SBS', model_selection=True)

        if isinstance(tauhat_data, int):
            # If it's an int, it means there's 1 CP.
            G_multi_n_cps = 1
            G_multi_tau = [tauhat_data]
        elif tauhat_data is None:
            # Handle cases where it might be None
            G_multi_n_cps = 0
            G_multi_tau = []
        else:
            # Otherwise, it should be an iterable.
            G_multi_n_cps = len(tauhat_data)
            G_multi_tau = sorted(tauhat_data)

        gmulti_ari = compute_ari(G_multi_tau, true_cps, n_samples)
        gmulti_recall, _, __ = calculate_set_metrics(true_cps, G_multi_tau, tol)

        for name, model in model_bank.items():
            res = run_gBottomUp(model, data)
            tauhat_data = res["tauhat"]
            cp_candidate = res["cp_candidate"]
            if isinstance(tauhat_data, int):
                # If it's an int, it means there's 1 CP.
                G_bottomup_n_cps = 1
                G_bottomup_tau = [tauhat_data]
                cp_candidate = [cp_candidate]
            elif tauhat_data is None:
                # Handle cases where it might be None
                G_bottomup_n_cps = 0
                G_bottomup_tau = []
            else:
                # Otherwise, it should be an iterable.
                G_bottomup_n_cps = len(tauhat_data)
                G_bottomup_tau = sorted(tauhat_data)
                cp_candidate = sorted(cp_candidate)

            gBottomup_ari = compute_ari(G_bottomup_tau, true_cps, n_samples)
            gBottomup_recall, _, __ = calculate_set_metrics(true_cps, cp_candidate, tol)
            results.append({
            "scenario": scenario,
            "seed": seed,
            "model": name,
            "True_tau": true_cps,
            "G_multi_tau": G_multi_tau,
            "G_bottomup_candidates": cp_candidate,
            "G_bottomup_tau": G_bottomup_tau,
            "G_multi_n_cps": G_multi_n_cps,
            "G_bottomup_n_cps": G_bottomup_n_cps,
            "G_multi_recall": gmulti_recall,
            "G_bottomup_recall": gBottomup_recall,
            "G_multi_ari": gmulti_ari,
            "G_bottomup_ari": gBottomup_ari
            })

        del model_bank  # 메모리 정리
        gc.collect()  # Python 메모리 정리
        r_gc()  # R 메모리 정리

    # 결과를 DataFrame(시나리오에 관해 저장, 모델별 평균 결과확인)
    df = pd.DataFrame(results)
    for model_name in model_namelst:
        df_model = df[df["model"] == model_name]
        summary = {
            "G_multi_ncps": df_model["G_multi_n_cps"].mean(),
            "G_bottomup_ncps": df_model["G_bottomup_n_cps"].mean(),
            "G_multi_recall": df_model["G_multi_recall"].mean(),
            "G_bottomup_recall": df_model["G_bottomup_recall"].mean(),
            "G_multi_ari": df_model["G_multi_ari"].mean(),
            "G_bottomup_ari": df_model["G_bottomup_ari"].mean()
        }
        logger_multi.info(f"\n====== {scenario}-d{d}-frequency{frequency}======")
        logger_multi.info(f"Model: {model_name}")
        logger_multi.info(f"summary: {summary}")
    return df


def save_multipoint_frequent_GraphStep_exp(runs=100, timestamp=None):
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"results/multi_point_GraphStep/{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    d_list = [500]
    frequency_lst = [50]
    tol = 5
    scenario_lst = ['modelsingle1-1', 'modelsingle2-1', 'modelsingle3-1']
    for d in d_list:
        for scenario in scenario_lst:
            for frequency in frequency_lst:
                # 데이터 생성 및 모델 실행
                df = experiment_all_models(d=d, scenario=scenario, frequency=frequency, runs=runs, tol=tol, timestamp=timestamp)
                save_name = f"multiple_test_{scenario}_d{d}_frequency_{frequency}_{timestamp}.csv"
                df.to_csv(os.path.join(output_dir, save_name), index=False)
                logger_multi.info(f"Saved {save_name} to {output_dir}")