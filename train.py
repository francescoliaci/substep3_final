#!/usr/bin/env python3
# ============================================================
# Substep 3 – Step Matching + Realized Graph Saving
#
# You ALREADY have:
#   (A) Substep-1 output dict (.npy):
#       key   = "<video_id>_<start>_<end>"  e.g. "10_16_360p_0.00_4.80"
#       value = np.ndarray shape (1792,)
#   (B) Precomputed text embeddings folder:
#       <recipe_name>_node_embeddings.npy  (N_nodes, 256)
#       <recipe_name>_meta.json            (optional)
#
# This script provides TWO MODES:
#   1) TRAIN mode: learns a projection Linear(1792 -> 256) using weak supervision
#      (temporal order ~ graph order).
#   2) MATCH mode: loads the trained projection, runs clustering + Hungarian,
#      fuses matched node features, and saves realized graphs as .npz.
#
# Usage examples (Colab):
#   # Train projection
#   python substep3_train_and_match.py \
#     --mode train \
#     --final_features_path /content/drive/MyDrive/AMLproject/output_subtask_1.npy \
#     --graph_dir /content/drive/MyDrive/AMLproject/task_graphs \
#     --csv_path /content/drive/MyDrive/AMLproject/activity_idx_step_idx.csv \
#     --text_emb_dir /content/drive/MyDrive/AMLproject/text_embeddings \
#     --proj_out /content/drive/MyDrive/AMLproject/visual_proj.pt \
#     --epochs 5 --lr 1e-3
#
#   # Run matching + save realized graphs
#   python substep3_train_and_match.py \
#     --mode match \
#     --final_features_path /content/drive/MyDrive/AMLproject/output_subtask_1.npy \
#     --graph_dir /content/drive/MyDrive/AMLproject/task_graphs \
#     --csv_path /content/drive/MyDrive/AMLproject/activity_idx_step_idx.csv \
#     --text_emb_dir /content/drive/MyDrive/AMLproject/text_embeddings \
#     --proj_in /content/drive/MyDrive/AMLproject/visual_proj.pt \
#     --output_dir /content/drive/MyDrive/AMLproject/graph_realizations_2
# ============================================================

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


# -----------------------------
# Helpers
# -----------------------------
def safe_recipe_filename(recipe_name: str) -> str:
    # matches your existing convention (<name_without_spaces>.json)
    return recipe_name.lower().replace(" ", "")


def load_substep1_dict(final_features_path: str) -> Dict[str, np.ndarray]:
    data = np.load(final_features_path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError("Substep-1 file did not load as a dict. Check the .npy content.")
    return data


def group_by_video(data: Dict[str, np.ndarray]) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[Tuple[float, float]]]]:
    video_to_steps = defaultdict(list)
    video_to_times = defaultdict(list)

    for key, feat in data.items():
        parts = key.split("_")
        if len(parts) < 3:
            continue
        video_id = "_".join(parts[:-2])

        try:
            start_t = float(parts[-2])
            end_t = float(parts[-1])
        except Exception:
            start_t, end_t = np.nan, np.nan

        video_to_steps[video_id].append(np.asarray(feat, dtype=np.float32))
        video_to_times[video_id].append((start_t, end_t))

    return dict(video_to_steps), dict(video_to_times)


def load_recipe_map(csv_path: str) -> Dict[int, str]:
    df = pd.read_csv(csv_path)
    return (
        df[["activity_idx", "activity_name"]]
        .drop_duplicates()
        .set_index("activity_idx")["activity_name"]
        .to_dict()
    )


def load_task_graph(graph_dir: str, recipe_name: str) -> Tuple[List[int], List[str], List[List[int]]]:
    graph_path = os.path.join(graph_dir, f"{safe_recipe_filename(recipe_name)}.json")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(graph_path)

    with open(graph_path, "r") as f:
        graph = json.load(f)

    node_ids = sorted(graph["steps"].keys(), key=int)
    node_texts = [graph["steps"][i] for i in node_ids]
    node_ids_int = [int(i) for i in node_ids]

    # remap edges to 0..N-1
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids_int)}
    edges_remapped = [[id_to_idx[int(u)], id_to_idx[int(v)]] for u, v in graph["edges"]]

    # sanity
    num_nodes = len(node_ids_int)
    if edges_remapped:
        m = int(np.max(edges_remapped))
        if m >= num_nodes:
            raise ValueError(f"Bad edge remap: max {m} >= num_nodes {num_nodes} for recipe {recipe_name}")

    return node_ids_int, node_texts, edges_remapped


def load_text_node_embeddings(text_emb_dir: str, recipe_name: str) -> np.ndarray:
    # Your folder likely has filenames without spaces OR with original recipe_name.
    # We try a few common patterns.
    candidates = [
        os.path.join(text_emb_dir, f"{recipe_name}_node_embeddings.npy"),
        os.path.join(text_emb_dir, f"{safe_recipe_filename(recipe_name)}_node_embeddings.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return np.load(p).astype(np.float32)

    raise FileNotFoundError(
        "Could not find node embeddings for recipe_name. Tried:\n" + "\n".join(candidates)
    )


def topological_order(num_nodes: int, edges_remapped: List[List[int]]) -> List[int]:
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges_remapped)
    return list(nx.topological_sort(G))


def cluster_windows_to_pseudosteps(
    V_proj_norm: torch.Tensor,
    times: List[Tuple[float, float]],
    num_nodes: int,
    cluster_sim_threshold: float = 0.94,
    max_ratio: float = 2.0,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    V_proj_norm: (S_windows, 256), already normalized and temporally unordered
    times: list of (start,end) length S_windows
    Returns:
      V_pseudo: (K, 256) normalized
      pseudo_times: np.ndarray dtype=object length K
    """
    times_arr = np.array(times, dtype=object)
    starts = np.array([t[0] for t in times_arr], dtype=np.float32)

    order = np.argsort(starts)
    V_sorted = V_proj_norm[order]
    times_sorted = times_arr[order]

    clusters = []
    current = [0]

    for i in range(1, V_sorted.shape[0]):
        sim = torch.dot(V_sorted[i - 1], V_sorted[i]).item()
        if sim >= cluster_sim_threshold:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)

    # force reasonable number of pseudo-steps
    while len(clusters) > max_ratio * max(num_nodes, 1):
        merged = []
        i = 0
        while i < len(clusters):
            if i < len(clusters) - 1:
                merged.append(clusters[i] + clusters[i + 1])
                i += 2
            else:
                merged.append(clusters[i])
                i += 1
        clusters = merged

    pseudo_feats = []
    pseudo_times = []
    for c in clusters:
        feats = V_sorted[c]  # (len(c), 256)
        pseudo_feats.append(feats.mean(dim=0))

        st = times_sorted[c[0]][0]
        et = times_sorted[c[-1]][1]
        pseudo_times.append((st, et))

    V_pseudo = torch.stack(pseudo_feats, dim=0)
    V_pseudo = F.normalize(V_pseudo, dim=1)
    return V_pseudo, np.array(pseudo_times, dtype=object)


def hungarian_with_time_and_topology(
    V_pseudo: torch.Tensor,          # (K, 256) normalized
    pseudo_times: np.ndarray,        # length K
    T_nodes: torch.Tensor,           # (N, 256) normalized
    edges_remapped: List[List[int]],
    lambda_time: float = 3.0,
) -> np.ndarray:
    """
    Returns matched_pairs as array of shape (min(K,N), 2):
      (pseudo_step_index, node_index) in ORIGINAL compact indices.
    """
    K = V_pseudo.shape[0]
    N = T_nodes.shape[0]

    # order pseudo-steps by time
    starts = np.array([t[0] for t in pseudo_times], dtype=np.float32)
    order_steps = np.argsort(starts)
    V_ord = V_pseudo[order_steps]  # (K, 256)

    # topological order for nodes
    node_order = topological_order(N, edges_remapped)
    T_ord = T_nodes[node_order]    # (N, 256)

    sim = V_ord @ T_ord.T
    cost = -sim.detach().cpu().numpy()

    # diagonal penalty
    ii = np.arange(K)[:, None] / max(K - 1, 1)
    jj = np.arange(N)[None, :] / max(N - 1, 1)
    cost = cost + lambda_time * ((ii - jj) ** 2)

    row_ind, col_ind = linear_sum_assignment(cost)

    # map back
    row_ind = order_steps[row_ind]  # pseudo-step idx in [0..K-1]
    col_ind = np.array([node_order[c] for c in col_ind], dtype=np.int32)  # node idx in [0..N-1]

    return np.stack([row_ind, col_ind], axis=1)


# -----------------------------
# Models
# -----------------------------
class NodeFusion(nn.Module):
    """
    Simple fusion: concat([text, visual]) -> linear -> 256
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, text_feat: torch.Tensor, visual_feat: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([text_feat, visual_feat], dim=-1))


# -----------------------------
# Train projection
# -----------------------------
def train_visual_projection(
    video_to_steps: Dict[str, List[np.ndarray]],
    video_to_times: Dict[str, List[Tuple[float, float]]],
    id_to_recipe: Dict[int, str],
    graph_dir: str,
    text_emb_dir: str,
    device: str,
    epochs: int = 5,
    lr: float = 1e-3,
    cluster_sim_threshold: float = 0.94,
    max_ratio: float = 2.0,
    lambda_time: float = 0.0,  # during training you can skip penalty; order is enforced by targets
) -> nn.Module:
    """
    Weak supervision: after clustering and ordering, assume diagonal alignment:
      pseudo-step i matches node i (up to min(K,N)).
    Use cross-entropy on similarity matrix.
    """
    VISUAL_DIM = 1792
    TEXT_DIM = 256

    visual_proj = nn.Linear(VISUAL_DIM, TEXT_DIM).to(device)
    opt = torch.optim.Adam(visual_proj.parameters(), lr=lr)

    visual_proj.train()

    video_ids = list(video_to_steps.keys())

    for ep in range(epochs):
        total_loss = 0.0
        n_used = 0

        for video_id in tqdm(video_ids, desc=f"Train epoch {ep+1}/{epochs}"):
            # parse recipe_id
            try:
                recipe_id = int(video_id.split("_")[0])
            except Exception:
                continue
            if recipe_id not in id_to_recipe:
                continue
            recipe_name = id_to_recipe[recipe_id]

            # load graph
            try:
                node_ids_int, node_texts, edges_remapped = load_task_graph(graph_dir, recipe_name)
            except Exception:
                continue
            num_nodes = len(node_ids_int)
            if num_nodes < 2:
                continue

            # load text embeddings
            try:
                T_np = load_text_node_embeddings(text_emb_dir, recipe_name)  # (N,256)
            except Exception:
                continue

            T = torch.from_numpy(T_np).float().to(device)
            T = F.normalize(T, dim=1)

            # visual windows
            step_feats = video_to_steps[video_id]
            times = video_to_times[video_id]
            if len(step_feats) < 2:
                continue

            V = torch.from_numpy(np.stack(step_feats, axis=0)).float().to(device)  # (S,1792)
            Vp = visual_proj(V)                                                   # (S,256)
            Vp = F.normalize(Vp, dim=1)

            # cluster -> pseudo steps (still 256)
            V_pseudo, pseudo_times = cluster_windows_to_pseudosteps(
                Vp, times, num_nodes=num_nodes,
                cluster_sim_threshold=cluster_sim_threshold,
                max_ratio=max_ratio,
            )

            K = V_pseudo.shape[0]
            N = T.shape[0]
            M = min(K, N)
            if M < 2:
                continue

            # order by time and topo, then take first M
            starts = np.array([t[0] for t in pseudo_times], dtype=np.float32)
            order_steps = np.argsort(starts)
            V_ord = V_pseudo[order_steps][:M]  # (M,256)

            node_order = topological_order(N, edges_remapped)
            T_ord = T[node_order][:M]          # (M,256)

            # similarity (M,M)
            S = V_ord @ T_ord.T

            targets = torch.arange(M, device=device)
            loss = F.cross_entropy(S, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            n_used += 1

        avg = total_loss / max(n_used, 1)
        print(f"[TRAIN] epoch {ep+1}: avg_loss={avg:.4f} videos_used={n_used}")

    return visual_proj.eval()


# -----------------------------
# Match + Save realized graphs
# -----------------------------
@torch.no_grad()
def match_and_save(
    video_to_steps: Dict[str, List[np.ndarray]],
    video_to_times: Dict[str, List[Tuple[float, float]]],
    id_to_recipe: Dict[int, str],
    graph_dir: str,
    text_emb_dir: str,
    output_dir: str,
    visual_proj: nn.Module,
    device: str,
    cluster_sim_threshold: float = 0.94,
    max_ratio: float = 2.0,
    lambda_time: float = 3.0,
):
    os.makedirs(output_dir, exist_ok=True)

    fusion = NodeFusion(256).to(device).eval()

    saved_count = 0
    skipped_no_recipe = 0
    skipped_no_graph = 0
    skipped_no_text = 0
    skipped_empty = 0

    for video_id, step_feats in tqdm(video_to_steps.items(), desc="Matching videos"):
        if len(step_feats) == 0:
            skipped_empty += 1
            continue

        # map to recipe
        try:
            recipe_id = int(video_id.split("_")[0])
        except Exception:
            skipped_no_recipe += 1
            continue
        if recipe_id not in id_to_recipe:
            skipped_no_recipe += 1
            continue
        recipe_name = id_to_recipe[recipe_id]

        # graph
        try:
            node_ids_int, node_texts, edges_remapped = load_task_graph(graph_dir, recipe_name)
        except Exception:
            skipped_no_graph += 1
            continue
        num_nodes = len(node_ids_int)
        if num_nodes == 0:
            skipped_no_graph += 1
            continue

        # text embeddings
        try:
            T_np = load_text_node_embeddings(text_emb_dir, recipe_name)  # (N,256)
        except Exception:
            skipped_no_text += 1
            continue
        T = torch.from_numpy(T_np).float().to(device)
        T = F.normalize(T, dim=1)  # (N,256)

        # visual windows -> project
        V = torch.from_numpy(np.stack(step_feats, axis=0)).float().to(device)  # (S,1792)
        Vp = visual_proj(V)                                                   # (S,256)
        Vp = F.normalize(Vp, dim=1)

        # cluster -> pseudo steps
        times = video_to_times[video_id]
        V_pseudo, pseudo_times = cluster_windows_to_pseudosteps(
            Vp, times, num_nodes=num_nodes,
            cluster_sim_threshold=cluster_sim_threshold,
            max_ratio=max_ratio,
        )

        # Hungarian match
        matched_pairs = hungarian_with_time_and_topology(
            V_pseudo, pseudo_times, T, edges_remapped, lambda_time=lambda_time
        )

        # fuse matched nodes
        node_features = T.clone()
        for r, c in matched_pairs:
            node_features[c] = fusion(T[c], V_pseudo[r])

        # save
        out_path = os.path.join(output_dir, f"{video_id}_graph.npz")
        np.savez(
            out_path,
            node_features=node_features.detach().cpu().numpy(),     # (N,256)
            edges=np.array(edges_remapped, dtype=np.int32),
            node_texts=np.array(node_texts, dtype=object),
            matched_pairs=matched_pairs.astype(np.int32),
            step_times=np.array(pseudo_times, dtype=object),
            original_node_ids=np.array(node_ids_int, dtype=np.int32),
        )
        saved_count += 1

    print("\n✅ Substep 3 completed.")
    print("Saved graphs:", saved_count)
    print("Skipped (no recipe in CSV):", skipped_no_recipe)
    print("Skipped (no graph json found):", skipped_no_graph)
    print("Skipped (no text embeddings):", skipped_no_text)
    print("Skipped (empty):", skipped_empty)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "match"], required=True)

    ap.add_argument("--final_features_path", required=True)
    ap.add_argument("--graph_dir", required=True)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--text_emb_dir", required=True)

    ap.add_argument("--output_dir", default=None, help="Required for mode=match")
    ap.add_argument("--proj_out", default=None, help="Where to save projection (mode=train)")
    ap.add_argument("--proj_in", default=None, help="Projection to load (mode=match)")

    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--cluster_sim_threshold", type=float, default=0.94)
    ap.add_argument("--max_ratio", type=float, default=2.0)
    ap.add_argument("--lambda_time", type=float, default=3.0)

    args = ap.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    # load inputs
    print("Loading Substep-1 dict:", args.final_features_path)
    data = load_substep1_dict(args.final_features_path)
    print("Segments:", len(data))

    video_to_steps, video_to_times = group_by_video(data)
    print("Videos found:", len(video_to_steps))

    id_to_recipe = load_recipe_map(args.csv_path)
    print("Recipes in CSV:", len(id_to_recipe))

    if args.mode == "train":
        visual_proj = train_visual_projection(
            video_to_steps=video_to_steps,
            video_to_times=video_to_times,
            id_to_recipe=id_to_recipe,
            graph_dir=args.graph_dir,
            text_emb_dir=args.text_emb_dir,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            cluster_sim_threshold=args.cluster_sim_threshold,
            max_ratio=args.max_ratio,
            lambda_time=0.0,
        )

        if not args.proj_out:
            raise ValueError("--proj_out is required in mode=train")
        os.makedirs(os.path.dirname(args.proj_out) or ".", exist_ok=True)
        torch.save(visual_proj.state_dict(), args.proj_out)
        print("✅ Saved projection to:", args.proj_out)

    elif args.mode == "match":
        if not args.output_dir:
            raise ValueError("--output_dir is required in mode=match")
        if not args.proj_in:
            raise ValueError("--proj_in is required in mode=match")

        visual_proj = nn.Linear(1792, 256).to(device)
        visual_proj.load_state_dict(torch.load(args.proj_in, map_location=device))
        visual_proj.eval()

        match_and_save(
            video_to_steps=video_to_steps,
            video_to_times=video_to_times,
            id_to_recipe=id_to_recipe,
            graph_dir=args.graph_dir,
            text_emb_dir=args.text_emb_dir,
            output_dir=args.output_dir,
            visual_proj=visual_proj,
            device=device,
            cluster_sim_threshold=args.cluster_sim_threshold,
            max_ratio=args.max_ratio,
            lambda_time=args.lambda_time,
        )


if __name__ == "__main__":
    main()




