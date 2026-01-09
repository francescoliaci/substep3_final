# ============================================================
# Substep 3 – Task-Graph Encoding + Step Matching (UPDATED)
# Works with your Substep-1 output_subtask_1.npy dict:
#   key   = "<video_id>_<start>_<end>"   e.g. "10_16_360p_0.00_4.80"
#   value = np.ndarray shape (1792,)
#
# What it does:
# 1) Loads the dict of pseudo-step embeddings (sliding windows)
# 2) Groups them per video_id
# 3) Encodes task-graph node texts using EgoVLP text encoder (1792-D)
# 4) Visual step embeddings are already 1792-D (EgoVLP-aligned) 
# 5) Runs Hungarian matching (1-to-1) between visual steps and graph nodes
# 6) Fuses matched node features with their matched visual features
# 7) Saves a realized graph per video as .npz (ready for Substep 4)
# ============================================================
import sys 
sys.path.append("/content")

import os
import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from transformers import DistilBertTokenizer

# ------------------------------------------------------------
# PATHS (EDIT THESE TO YOUR SETUP)
# ------------------------------------------------------------

FINAL_FEATURES_PATH = "/content/drive/MyDrive/AMLproject/output_subtask_1.npy"
GRAPH_DIR           = "/content/drive/MyDrive/AMLproject/task_graphs"
CSV_PATH            = "/content/drive/MyDrive/AMLproject/activity_idx_step_idx.csv"
OUTPUT_DIR          = "/content/drive/MyDrive/AMLproject/graph_realizations_2"
EGOVLP_CKPT         = "/content/drive/MyDrive/AMLproject/pretrained/EgoVLP_PT_BEST.pth"

# EgoVLP repo path (so we can import FrozenInTime)
# Example:
#   /content/EgoVLP/
#       model/model.py
# If your folder name differs, update accordingly.
EGOVLP_REPO_PATH    = "/content/EgoVLP"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# BASIC SANITY CHECKS (HELPFUL WHEN RUNNING ON COLAB)
# ------------------------------------------------------------

print("DEVICE:", DEVICE)
print("FINAL_FEATURES_PATH exists:", os.path.exists(FINAL_FEATURES_PATH))
print("GRAPH_DIR exists:", os.path.exists(GRAPH_DIR))
print("CSV_PATH exists:", os.path.exists(CSV_PATH))
print("OUTPUT_DIR:", OUTPUT_DIR)

# ------------------------------------------------------------
# LOAD CSV MAPPING: activity_idx -> activity_name
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

id_to_recipe = (
    df[["activity_idx", "activity_name"]]
    .drop_duplicates()
    .set_index("activity_idx")["activity_name"]
    .to_dict()
)

# ------------------------------------------------------------
# LOAD EGOVLP TEXT ENCODER (FrozenInTime)
# ------------------------------------------------------------

from EgoVLP.model.model import FrozenInTime 
# from EgoVLP.base.base_model import BaseModel (in model.py)

# NOTE: We are using DistilBERT because FrozenInTime (EgoVLP) supports it
text_params = {
    "model": "distilbert-base-uncased",
    "pretrained": True,
}

# Create model: only text branch is used here
model = FrozenInTime(
    text_params=text_params,
    target_dim=1792,
    load_checkpoint=EGOVLP_CKPT
).to(DEVICE)

model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

@torch.no_grad()
def encode_text(texts):
    """
    Encode a list of strings (node step descriptions) into EgoVLP-aligned embeddings.
    Returns: torch.Tensor of shape (N, 1792), L2-normalized.
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    feats = model.compute_text(enc)     # (N, 1792)
    feats = F.normalize(feats, dim=1)   # cosine space
    return feats

# ------------------------------------------------------------
# NODE FUSION MODULE
# ------------------------------------------------------------

class NodeFusion(nn.Module):
    """
    Fuse a node text embedding with a matched visual step embedding.
    Input:  text_feat   (1792,)
            visual_feat (1792,)
    Output: fused node_feat (1792,)
    """
    def __init__(self, dim=1792):
        super().__init__()
        # keep it simple; avoid ReLU to preserve cosine geometry
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, text_feat, visual_feat):
        x = torch.cat([text_feat, visual_feat], dim=-1)
        return self.proj(x)

fusion = NodeFusion(1792).to(DEVICE)
fusion.eval()

# ------------------------------------------------------------
# LOAD SUBSTEP 1 OUTPUT (DICT OF WINDOW FEATURES)
# ------------------------------------------------------------

print("\nLoading Substep-1 dict from:", FINAL_FEATURES_PATH)
data = np.load(FINAL_FEATURES_PATH, allow_pickle=True).item()

print("Type:", type(data))
print("Number of segments:", len(data))

# Peek at a few items
for i, (k, v) in enumerate(data.items()):
    print("Example key:", k, "shape:", np.array(v).shape)
    if i == 2:
        break

# ------------------------------------------------------------
# DETERMINE VISUAL DIM (YOUR FILE SHOWS 1792)
# ------------------------------------------------------------

# Find the first vector and infer its dimensionality robustly
_first_key = next(iter(data.keys()))
VISUAL_DIM = int(np.array(data[_first_key]).shape[0])
TEXT_DIM = 1792

print("\nDetected VISUAL_DIM from file:", VISUAL_DIM)

# ------------------------------------------------------------
# GROUP FEATURES BY VIDEO
# Your keys are like: "10_16_360p_0.00_4.80"
# We must extract video_id = "10_16_360p" (everything except last 2 parts).
# ------------------------------------------------------------

video_to_steps = defaultdict(list)
video_to_times = defaultdict(list)  # store (start, end) for optional debugging

for key, feat in data.items():
    parts = key.split("_")
    if len(parts) < 3:
        # Not enough parts to parse <video>_<start>_<end>
        continue

    # video_id = all parts except last two (start, end)
    video_id = "_".join(parts[:-2])

    # parse start/end if possible (optional)
    try:
        start_t = float(parts[-2])
        end_t   = float(parts[-1])
    except:
        start_t, end_t = None, None

    video_to_steps[video_id].append(feat)
    video_to_times[video_id].append((start_t, end_t))

print("\n🎥 Videos found:", len(video_to_steps))
print("Example video_id:", next(iter(video_to_steps.keys())))

# ------------------------------------------------------------
# MAIN LOOP: MATCH STEPS TO TASK GRAPH
# ------------------------------------------------------------

# Optional counters for logging
saved_count = 0
skipped_no_graph = 0
skipped_no_recipe = 0
skipped_empty = 0

for video_id, step_feats in tqdm(video_to_steps.items(), desc="Matching videos"):

    if len(step_feats) == 0:
        skipped_empty += 1
        continue

    # --------------------------------------------------------
    # MAP VIDEO -> RECIPE ID
    # Your video_id begins with activity_idx (e.g., "10_16_360p")
    # so recipe_id = int(video_id.split("_")[0]) should be correct.
    # --------------------------------------------------------
    try:
        recipe_id = int(video_id.split("_")[0])
    except:
        skipped_no_recipe += 1
        continue

    if recipe_id not in id_to_recipe:
        skipped_no_recipe += 1
        continue

    recipe_name = id_to_recipe[recipe_id]

    # Your graphs are stored as <recipe_name_without_spaces>.json
    # If your graph filenames differ, adjust this.
    safe_name = recipe_name.lower().replace(" ", "")
    graph_path = os.path.join(GRAPH_DIR, f"{safe_name}.json")

    if not os.path.exists(graph_path):
        skipped_no_graph += 1
        continue

    # --------------------------------------------------------
    # LOAD TASK GRAPH JSON
    # Expected structure:
    #   graph["steps"] : dict { node_id(str/int) : text(str) }
    #   graph["edges"] : list of edges
    # --------------------------------------------------------
    with open(graph_path, "r") as f:
        graph = json.load(f)

    node_ids   = sorted(graph["steps"].keys(), key=int)
    node_texts = [graph["steps"][i] for i in node_ids]

    # --------------------------------------------------------
    # MAP ORIGINAL NODE IDS -> COMPACT INDICES (0..N-1)
    # --------------------------------------------------------

    node_ids_int = [int(i) for i in node_ids]

    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids_int)}
    idx_to_id = {idx: node_id for idx, node_id in enumerate(node_ids_int)}
    # remap edges
    edges_remapped = [
        [id_to_idx[int(u)], id_to_idx[int(v)]]
        for u, v in graph["edges"]
    ]
    # sanity check
    num_nodes = len(node_ids_int)
    if len(edges_remapped) > 0:
        m = np.max(edges_remapped)
        if m >= num_nodes:
            raise ValueError(f"Bad remap: max edge idx {m} >= num_nodes {num_nodes} for {video_id}")

    # --------------------------------------------------------
    # TEXT ENCODING (TASK GRAPH NODES) -> (num_nodes, 256)
    # --------------------------------------------------------
    T = encode_text(node_texts)

    # --------------------------------------------------------
    # VISUAL STEP EMBEDDINGS
    # step_feats is a list of (1792,) numpy arrays.
    # We stack -> (num_steps, 1792)
    # Project -> (num_steps, 256)
    # Normalize -> cosine similarity space
    # --------------------------------------------------------
    V_np = np.stack(step_feats).astype(np.float32)  # (S, VISUAL_DIM)

    V = torch.from_numpy(V_np).to(DEVICE)           # (S, 1792)
    V = F.normalize(V, dim=1)                       # (S, 1792)

    # --------------------------------------------------------
    # HUNGARIAN MATCHING (1-to-1)
    # sim: (S, N)
    # Hungarian solves min cost, so we use cost = -sim (maximize sim)
    # --------------------------------------------------------
    sim  = V @ T.T                                   # (S, N)
    cost = -sim.detach().cpu().numpy()               # (S, N)

    row_ind, col_ind = linear_sum_assignment(cost)   # pairs: (visual_step_idx, node_idx)

    # --------------------------------------------------------
    # FUSE MATCHED NODES
    # Start from node_features = T (text-only)
    # For matched node c, replace with fusion(text_node, matched_visual_step)
    # --------------------------------------------------------
    node_features = T.clone()

    for r, c in zip(row_ind, col_ind):
        node_features[c] = fusion(T[c], V[r])

    # --------------------------------------------------------
    # SAVE GRAPH REALIZATION PER VIDEO
    # --------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, f"{video_id}_graph.npz")

    np.savez(
        out_path,
        # node features aligned with indices 0..N-1
        node_features=node_features.detach().cpu().numpy(),  # (N, 1792)
        # edges now use compact indices
        edges=np.array(edges_remapped, dtype=np.int32),
        # text aligned with node_features
        node_texts=np.array(node_texts, dtype=object),
        # hungarian output already uses compact indices (OK)
        matched_pairs=np.array(list(zip(row_ind, col_ind)), dtype=np.int32),
        # optional but useful
        step_times=np.array(video_to_times[video_id], dtype=object),
        # DEBUG 
        original_node_ids=np.array(node_ids_int, dtype=np.int32)
    )   

    saved_count += 1

print("\n✅ Substep 3 completed.")
print("Saved graphs:", saved_count)
print("Skipped (no recipe in CSV):", skipped_no_recipe)
print("Skipped (no graph json found):", skipped_no_graph)
print("Skipped (empty):", skipped_empty)

# ------------------------------------------------------------
# QUICK SANITY CHECK: LOAD ONE SAVED FILE AND PRINT SUMMARY
# ------------------------------------------------------------
saved_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith("_graph.npz")]
print("\nFiles saved in OUTPUT_DIR:", len(saved_files))

if len(saved_files) > 0:
    example = saved_files[0]
    x = np.load(os.path.join(OUTPUT_DIR, example), allow_pickle=True)

    print("\n--- Example realized graph:", example, "---")
    print("node_features shape:", x["node_features"].shape)
    print("edges shape:", x["edges"].shape)
    print("num node_texts:", len(x["node_texts"]))
    print("matched_pairs shape:", x["matched_pairs"].shape)
    print("first 5 matched pairs (visual_step_idx, node_idx):", x["matched_pairs"][:5])
