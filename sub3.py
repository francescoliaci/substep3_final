# ============================================================
# Substep 3 – Task-Graph Encoding + Step Matching (FIXED)
# Compatible with sliding-window Substep 1 output
# ============================================================

import os
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from transformers import DistilBertTokenizer

# ------------------------------------------------------------
# PATHS 
# ------------------------------------------------------------

FINAL_FEATURES_PATH = "/content/drive/MyDrive/AMLproject/output_subtask_1.npy"
GRAPH_DIR           = "/content/drive/MyDrive/AMLproject/task_graphs"
CSV_PATH            = "/content/drive/MyDrive/AMLproject/activity_idx_step_idx.csv"
OUTPUT_DIR          = "/content/drive/MyDrive/AMLproject/graph_realizations_2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD CSV (VIDEO -> RECIPE NAME)
# ------------------------------------------------------------

import pandas as pd

df = pd.read_csv(CSV_PATH)

id_to_recipe = (
    df[["activity_idx", "activity_name"]]
    .drop_duplicates()
    .set_index("activity_idx")["activity_name"]
    .to_dict()
)

# ------------------------------------------------------------
# LOAD EGOVLP TEXT ENCODER
# ------------------------------------------------------------

from model.model import FrozenInTime

text_params = {
    "model": "distilbert-base-uncased",
    "pretrained": True,
}

model = FrozenInTime(
    text_params=text_params,
    projection_dim=256
).to(DEVICE)

model.eval()

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

@torch.no_grad()
def encode_text(texts):
    """
    Encode a list of step descriptions into EgoVLP text embeddings.
    Output shape: (num_nodes, 256)
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    feats = model.compute_text(enc)
    return F.normalize(feats, dim=1)

# ------------------------------------------------------------
# NODE FUSION MODULE
# ------------------------------------------------------------

class NodeFusion(nn.Module):
    """
    Fuses task-graph node (text) features with matched visual features
    """
    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, text_feat, visual_feat):
        x = torch.cat([text_feat, visual_feat], dim=-1)
        return self.proj(x)

fusion = NodeFusion(256).to(DEVICE)
fusion.eval()

# ------------------------------------------------------------
# LOAD SUBSTEP 1 OUTPUT (SLIDING-WINDOW FEATURES)
# ------------------------------------------------------------

print("Loading Substep 1 features...")
data = np.load(FINAL_FEATURES_PATH, allow_pickle=True).item()

# ------------------------------------------------------------
# GROUP FEATURES BY VIDEO
# ------------------------------------------------------------

video_to_steps = defaultdict(list)

for key, feat in data.items():
    # expected key format: videoID_start_end_0
    video_id = key.split("_")[0]
    video_to_steps[video_id].append(feat)

print(f"Videos found: {len(video_to_steps)}")

# ------------------------------------------------------------
# MAIN LOOP: MATCH STEPS TO TASK GRAPH
# ------------------------------------------------------------

for video_id, step_feats in tqdm(video_to_steps.items()):

    # --------------------------------------------------------
    # MAP VIDEO -> RECIPE
    # --------------------------------------------------------

    try:
        recipe_id = int(video_id.split("_")[0])
    except:
        continue

    if recipe_id not in id_to_recipe:
        continue

    recipe_name = id_to_recipe[recipe_id]
    safe_name = recipe_name.lower().replace(" ", "")
    graph_path = os.path.join(GRAPH_DIR, f"{safe_name}.json")

    if not os.path.exists(graph_path):
        continue

    # --------------------------------------------------------
    # LOAD TASK GRAPH
    # --------------------------------------------------------

    with open(graph_path, "r") as f:
        graph = json.load(f)

    node_ids   = sorted(graph["steps"].keys(), key=int)
    node_texts = [graph["steps"][i] for i in node_ids]

    # --------------------------------------------------------
    # TEXT ENCODING (TASK GRAPH NODES)
    # --------------------------------------------------------

    T = encode_text(node_texts)  # (num_nodes, 256)

    # --------------------------------------------------------
    # VISUAL STEP EMBEDDINGS
    # --------------------------------------------------------
    # Each step is a 1280-D fused vector:
    # [ EgoVLP (256) | Omnivore (1024) ]
    # We keep ONLY EgoVLP to stay in aligned space

    V = torch.tensor(
        np.stack(step_feats),
        dtype=torch.float32,
        device=DEVICE
    )

    V = V[:, :256]          # keep EgoVLP part
    V = F.normalize(V, 1)   # cosine space

    # --------------------------------------------------------
    # HUNGARIAN MATCHING
    # --------------------------------------------------------

    sim  = V @ T.T
    cost = -sim.detach().cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(cost)

    # --------------------------------------------------------
    # FUSE MATCHED NODES
    # --------------------------------------------------------

    node_features = T.clone()

    for r, c in zip(row_ind, col_ind):
        node_features[c] = fusion(T[c], V[r])

    # --------------------------------------------------------
    # SAVE GRAPH REALIZATION
    # --------------------------------------------------------

    out_path = os.path.join(OUTPUT_DIR, f"{video_id}_graph.npz")

    np.savez(
        out_path,
        node_features=node_features.detach().cpu().numpy(),
        edges=np.array(graph["edges"]),
        node_texts=node_texts,
        matched_pairs=np.array(list(zip(row_ind, col_ind)))
    )

print("✅ Substep 3 completed successfully")
