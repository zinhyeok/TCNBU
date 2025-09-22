import os
import shutil
import pandas as pd

# 경로 설정
csv_path = os.path.abspath("results/single_point/20250514_124613/mean_d500_rank-base_minobs-10.csv")  # 사용자가 업로드한 CSV
image_folder = os.path.abspath("fig/20250514_125452")                # 해당 실험 케이스의 이미지 폴더
output_root = os.path.join(image_folder,"classified_result")        # 결과 저장 폴더

# 분류 기준
group_map = {
    (1, 1): "Z1_G1",
    (1, 0): "Z1_G0",
    (0, 1): "Z0_G1",
    (0, 0): "Z0_G0",
}

# CSV 로드
df = pd.read_csv(csv_path)

# 이미지 목록 정렬 (CSV와 순서 맞춰야 하므로)
image_list = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

# 일치하는 수만큼만 처리
min_len = min(len(df), len(image_list))

# 분류 및 복사
for i in range(min_len):
    row = df.iloc[i]
    group_key = (row["Z_acc"], row["G_acc"])
    group_folder = group_map.get(group_key, "Unknown")

    src_path = os.path.join(image_folder, image_list[i])
    dst_folder = os.path.join(output_root, group_folder)
    os.makedirs(dst_folder, exist_ok=True)
    shutil.copy2(src_path, os.path.join(dst_folder, image_list[i]))

print(f"✅ 분류 완료: {min_len}개 이미지가 group_map 기준으로 분류됨.")
