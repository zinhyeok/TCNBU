import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
from datetime import datetime

def visualize_graph(edges, tau, k=None):
    """
    엣지 리스트와 임계점 tau를 받아 그래프를 시각화합니다.
    tau 이전 노드는 하늘색, 이후 노드는 분홍색으로 표시됩니다.
    
    Parameters:
        edges (list of tuples): (node1, node2) 형태의 엣지 리스트
        tau (int): 노드를 구분할 임계점
        k (int): 그래프의 k-값
    """
    save_path = os.path.join("fig/", f"graph_tau_{tau}.png")

    # 그래프 생성 및 엣지 추가
    G = nx.Graph()
    G.add_edges_from(edges)

    # 노드 그룹 구분
    group_a = list(range(tau))
    group_b = list(set(G.nodes()) - set(group_a))

    # 노드 색상 설정
    node_colors = ['skyblue' if node in group_a else 'lightcoral' for node in G.nodes()]

    # 시각화
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            edge_color='gray', node_size=500)
    
    plt.title(f"Graph Visualization with τ = {tau}, k={k}")
    # plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    plt.close()  # 메모리 누수 방지용


def plot_tree_layout_custom(merge_history, model_timestamp=None, file_name="merge_tree.png", title="gButtomUp Tree Layout", size=(36,20)):
    # GUI 없이 실행 가능하도록 설정 (headless-safe)
    matplotlib.use("Agg")

    G = nx.DiGraph()
    node_labels = {}
    node_colors = []
    node_id = 0

    seg_to_node = {}
    node_step = {}

    merge_history = [step_tuple[0] for step_tuple in merge_history]

    # 초기 segment 노드 생성
    for seg in merge_history[0][0]:
        flat_seg = seg[0] if isinstance(seg[0], list) else seg
        label = f"{flat_seg[0]}-{flat_seg[-1]}" if len(flat_seg) > 1 else f"{flat_seg[0]}"
        G.add_node(node_id)
        node_labels[node_id] = label
        seg_to_node[tuple(flat_seg)] = node_id
        node_step[node_id] = 0
        node_colors.append(0)
        node_id += 1

    # 병합 단계 처리
    for step_idx in range(1, len(merge_history)):
        segments_prev = merge_history[step_idx - 1][0]
        segments_curr = merge_history[step_idx][0]

        prev_set = set(map(lambda s: tuple(s[0] if isinstance(s[0], list) else s), segments_prev))
        curr_set = set(map(lambda s: tuple(s[0] if isinstance(s[0], list) else s), segments_curr))
        new_merges = curr_set - prev_set

        for new_seg in new_merges:
            new_seg = new_seg[0] if isinstance(new_seg[0], list) else new_seg
            merged_from = []
            for old_seg in segments_prev:
                old_flat = old_seg[0] if isinstance(old_seg[0], list) else old_seg
                if set(old_flat).issubset(set(new_seg)):
                    merged_from.append(tuple(old_flat))

            if len(merged_from) >= 2:
                label = f"{new_seg[0]}-{new_seg[-1]}"
                G.add_node(node_id)
                node_labels[node_id] = label
                seg_to_node[tuple(new_seg)] = node_id
                node_step[node_id] = step_idx
                node_colors.append(step_idx)

                for sub in merged_from:
                    if sub in seg_to_node:
                        G.add_edge(seg_to_node[sub], node_id)

                node_id += 1

    # 트리 레이아웃 계산 및 y축 반전
    step_to_nodes = {}
    for node, step in node_step.items():
        step_to_nodes.setdefault(step, []).append(node)

    # 2. 좌표 할당
    pos = {}
    step_y_gap = 100   # 계층 높이 간격
    step_x_gap = 80    # 같은 계층 내 노드 간 간격
    def extract_start(label):
        parts = label.split("-")
        return int(parts[0]) if parts else int(label)

    for step, nodes in step_to_nodes.items():
        nodes = sorted(nodes, key=lambda node: extract_start(node_labels[node]))
        n = len(nodes)
        x_start = -((n - 1) * step_x_gap) / 2
        for i, node in enumerate(nodes):
            x = x_start + i * step_x_gap
            y = -step * step_y_gap
            pos[node] = (x, y)


    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    # #y축반전 option
    # # for k in pos:
    # #     x, y = pos[k]
    # #     pos[k] = (x, -y)  # root가 아래로

    # 시각화
    plt.figure(figsize=size, constrained_layout=True)
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(node_colors), max(node_colors))

    # nx.draw(G, pos, with_labels=False, arrows=False,
    #         node_color=node_colors, cmap=cmap, node_size=1500)
    nx.draw(G, pos, with_labels=False, arrows=False,
        node_color='lightgray', node_size=1500)
    
    custom_labels = {
    node: f"{node_labels[node]}\n(step {node_step[node]})"
    for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=custom_labels, font_size=9)

    # step 범례 표시
    unique_steps = sorted(set(node_colors))
    patches = [mpatches.Patch(color=cmap(norm(s)), label=f"Step {s}") for s in unique_steps]

    # plt.legend(handles=patches, title="Merge Step", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.title(title)
    if model_timestamp is None:
        model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    timestamp = model_timestamp
    folder_path = os.path.join("fig/", f"{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, file_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=300) 
    plt.close()



def plot_segment_tree(seg_history,
                      model_timestamp=None,
                      file_name="merge_tree.png",
                      title="gBottomUp – Segment Merge Tree",
                      size=(32, 18)):
    """
    seg_history : List[ List[List[int]] ]
        └─ seg_history[step] = [segment1, segment2, ...]
        각 segment 는 관측치 인덱스(또는 range) 리스트
    """
    # ────────────────────────────── 준비 (headless-safe)
    matplotlib.use("Agg")

    G            = nx.DiGraph()
    node_labels  = {}
    seg2node_id  = {}
    node_id      = 0

    # ────────────────────────────── 0) 초기 세그먼트 노드
    for seg in seg_history[0]:
        label = f"{seg[0]}-{seg[-1]}" if len(seg) > 1 else f"{seg[0]}"
        G.add_node(node_id)
        node_labels[node_id] = label
        seg2node_id[tuple(seg)] = node_id
        node_id += 1

    # ────────────────────────────── 1) step 별 병합 간선
    for step in range(1, len(seg_history)):
        prev_segs = [tuple(s) for s in seg_history[step-1]]
        curr_segs = [tuple(s) for s in seg_history[step]]

        prev_set  = set(prev_segs)
        curr_set  = set(curr_segs)

        # 새롭게 등장한(= 병합돼 만들어진) 세그먼트
        new_segs  = curr_set - prev_set

        for new_seg in new_segs:
            # 이 new_seg 를 구성한 직전 step 의 세그먼트들 찾기
            children = [pseg for pseg in prev_segs
                        if set(pseg).issubset(new_seg)]

            if len(children) < 2:          # 1개면 단순 이름 변경이므로 skip
                continue

            # 부모(merged) 노드 생성
            label = f"{new_seg[0]}-{new_seg[-1]}" if len(new_seg) > 1 else f"{new_seg[0]}"
            G.add_node(node_id)
            node_labels[node_id] = label
            seg2node_id[new_seg] = node_id

            # 간선 연결
            for child in children:
                G.add_edge(seg2node_id[child], node_id)

            node_id += 1

    # ────────────────────────────── 2) layout  (graphviz ‘dot’)
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    # ────────────────────────────── 3) 시각화 (흑백)
    plt.figure(figsize=size, constrained_layout=True)
    nx.draw(G, pos,
            with_labels=False, arrows=False,
            node_color="lightgray",   # → 단색
            node_size=1600,
            edgecolors="k", linewidths=0.6)

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    plt.title(title, fontsize=14)

    # ────────────────────────────── 4) 파일 저장
    if model_timestamp is None:
        model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = os.path.join("fig", model_timestamp)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, file_name), dpi=300, bbox_inches="tight")
    plt.close()
