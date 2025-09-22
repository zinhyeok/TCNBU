import numpy as np
import json
import subprocess
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
import os
from test.logger_setup import setup_module_logger
from model.gBottomup import gBottomupGraphstep
from itertools import product
from typing import Dict, List, Tuple
from util.dataGenerator import generate_single_data
# === Logging setup ===

logger_single  = setup_module_logger("project.single")
logger_single.info("Single-point test started")
project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(project_root, ".."))  # utilize/ 기준이면 한 단계 위로


# === Experiment Runner ===

def run_gseg_R(input_csv, output_json, alpha=0.05):
    # 절대경로로 변환
    base_dir = os.path.abspath(os.path.dirname(__file__))  # e.g., .../g_seg/utilize
    script_path = os.path.join(base_dir, "run_gseg_Z.R")
    input_path = os.path.abspath(input_csv)
    output_path = os.path.abspath(output_json)

    # Rscript 호출
    subprocess.run(
        ["Rscript", script_path, input_path, output_path, str(alpha)],
        check=True
    )
    # 결과 읽기
    with open(output_path, "r") as f:
        return json.load(f)
    
    
def run_gsegMulti_R(input_csv, output_json, alpha=0.05, method='SBS'):
    # 절대경로로 변환
    base_dir = os.path.abspath(os.path.dirname(__file__))  # e.g., .../g_seg/utilize
    script_path = os.path.join(base_dir, "run_gsegmulti.R")
    input_path = os.path.abspath(input_csv)
    output_path = os.path.abspath(output_json)

    # Rscript 호출
    subprocess.run(
        ["Rscript", script_path, input_path, output_path, str(alpha), method],
        check=True
    )
    # 결과 읽기
    with open(output_path, "r") as f:
        return json.load(f)
    

def evaluate(tauhat, true_tau, tol=20):
    if tauhat is None:
        return 0
    # tauhat이 list인 경우, 첫 번째 요소를 사용
    if isinstance(tauhat, list):
        tauhat = tauhat[0]

    return int(abs(tauhat - true_tau) <= tol)

def count_matches(predicted, true_cps, tol=20):
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

def count_false_positives(predicted, true_cps, tol=20):
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


def run_gButtomUp(model, data):
    cp_list, stat = model.fit(data)
    if not cp_list:
        return {"tauhat": None, "stat": stat}
    # sorted_cp = [cp for cp, s in sorted(zip(cp_list, stat), key=lambda x: x[1], reverse=True)]
    # sorted_stat = sorted(stat, reverse=True)
    # cp_list = sorted_cp[0]
    return {"tauhat": cp_list, "stat": stat}

# --------------------------------------------------------------------------- #
# Model builder / caching
# --------------------------------------------------------------------------- #
def build_model(c: int, r: str, step_th: int,
                base_merge: bool, eliminate: str,
                isFullTree: bool, min_obs:int,
                *, alpha: float, timestamp: str) -> Tuple[str, gBottomupGraphstep]:
    name = (f"Full{isFullTree}_{r}_{step_th}_"
            f"basemerge{base_merge}_c_{c}_{eliminate}_minobs{min_obs}")
    model = gBottomupGraphstep(
        rank_type=r,
        step_th=step_th,
        base_merge=base_merge,
        alpha=alpha,
        c=c,
        isFullTree=isFullTree,
        eliminate=eliminate,
        min_obs=min_obs,
        model_timestamp=name,
        logger_timestamp=timestamp,
        visualize=False,
    )
    return name, model

# === Experiment Loop for model lst ===
def experiment_all_models(n=200, d=50, scenario="mean", runs=100, tol=20, timestamp=None):
    
    base_dir = os.getcwd()
    input_csv = os.path.join(base_dir, "data.csv")
    output_json = os.path.join(base_dir, "z_result.json")

    alpha          = 0.05
    rank_types     = ["base"]
    min_obs_lst = [4]
    step_th_lst    = [-1]
    eliminate_lst  = ["both", "forward", "backward"]
    c_lst          = [0.5, 2]
    isFullTree_lst = [False]
    base_merge_lst = [True, False]

    grid = [g for g in product(c_lst, rank_types, step_th_lst,
                               base_merge_lst, eliminate_lst,
                               isFullTree_lst, min_obs_lst)]
    
    # ---- build model bank ------------------------------------------------- #
    model_bank: Dict[str, gBottomupGraphstep] = dict(
        build_model(*g, alpha=alpha, timestamp=timestamp) for g in grid
    )
    logger_single.info(f"Built {len(model_bank)} gBottomupGraphstep models.")
    tol = tol    
    # ---- 결과 저장을 위한 리스트 ------------------------------------------ #
    results = []
    model_lst = set()  # 중복 방지를 위해 set 사용    
    
    for seed in tqdm(range(runs), desc=f"{scenario}-d{d}"):
        data, true_cps = generate_single_data(n=n, d=d, scenario=scenario, seed=seed)
    
        np.savetxt(input_csv, data, delimiter=",")

        z_result = run_gseg_R(input_csv, output_json, alpha)
        z_acc = count_matches(z_result["tauhat"], true_cps, tol)
        z_false_positives = count_false_positives(z_result["tauhat"], true_cps, tol)

        gmulti_result_sbs = run_gsegMulti_R(input_csv, output_json, alpha, 'SBS')
        gmulti_acc_sbs = count_matches(gmulti_result_sbs["tauhat"], true_cps, tol)
        gmulti_false_positives_sbs = count_false_positives(gmulti_result_sbs["tauhat"], true_cps, tol)

        gmulti_result_wbs = run_gsegMulti_R(input_csv, output_json, alpha, 'WBS')
        gmulti_acc_wbs = count_matches(gmulti_result_wbs["tauhat"], true_cps, tol)
        gmulti_false_positives_wbs = count_false_positives(gmulti_result_wbs["tauhat"], true_cps, tol)
        
        for name, model in model_bank.items():
                model_lst.add(name)  # 모델 이름을 set에 추가
                res = run_gButtomUp(model, data)
                results.append({
                "scenario": scenario,
                "seed": seed,
                "model": name,
                "Z_tau": z_result["tauhat"],
                "Z_acc": z_acc,
                "Z_false_positives": z_false_positives,
                "G_multiSBS_tau": gmulti_result_sbs["tauhat"],
                "G_multiSBS_acc": gmulti_acc_sbs,
                "G_multiSBS_false_positives": gmulti_false_positives_sbs,
                "G_multiWBS_tau": gmulti_result_wbs["tauhat"],
                "G_multiWBS_acc": gmulti_acc_wbs,
                "G_multiWBS_false_positives": gmulti_false_positives_wbs,
                "G_bottomup_tau": res["tauhat"],
                "G_bottomup_acc": count_matches(res["tauhat"], true_cps, tol),
                "G_bottomup_fp": count_false_positives(res["tauhat"], true_cps, tol)
                })

    # 데이터 별 결과를 DataFrame으로 변환
    df = pd.DataFrame(results)
    for model_name in model_lst:
        df_model = df[df["model"] == model_name]
        summary = {
            "Z_acc_all": df_model["Z_acc"].mean(),
            "G_multiSBS_acc_all": df_model["G_multiSBS_acc"].mean(),
            "G_multiWBS_acc_all": df_model["G_multiWBS_acc"].mean(),
            "G_bottomup_acc": df_model["G_bottomup_acc"].mean(),
            "Z_false_positives_all": df_model["Z_false_positives"].mean(),
            "G_multiSBS_false_positives_all": df_model["G_multiSBS_false_positives"].mean(),
            "G_multiWBS_false_positives_all": df_model["G_multiWBS_false_positives"].mean(),
            "G_bottomup_false_positives": df_model["G_bottomup_fp"].mean(),
        }
        logger_single.info(f"\n====== {scenario}-d{d}-n{n} ======")
        logger_single.info(f"Model: {model_name}")
        logger_single.info(f"summary: {summary}")
        
    return df

# === 전체 실험 실행 ===
def save_singlepoint_exp(runs=100, timestamp=None):
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = os.getcwd()
    output_dir = os.path.join(project_root, f"results/single_point/{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    scenario_lst = ["mean", "scale", "both"]
    d_lst = [2000, 500]
    n = 200

    for scenario in scenario_lst:
        for d in d_lst:
            logger_single.info(f"Running {scenario}-d{d} with all models")
            df = experiment_all_models(
                n=n,
                d=d,
                scenario=scenario,
                runs=runs,
                tol=20,
                timestamp=timestamp
            )
            save_name = f"single_test_{scenario}_d{d}_n{n}_{timestamp}.csv"
            df.to_csv(os.path.join(output_dir, save_name), index=False)
            logger_single.info(f"Saved {save_name} to {output_dir}")
