# clustering.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
INPUT_NPY = "output_subtask_1.npy"
OUT_DIR = "step_embeddings"

CLUSTER_SIM_THRESHOLD = 0.94
MAX_RATIO = 2.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# LOAD WINDOW FEATURES
# -----------------------------
data = np.load(INPUT_NPY, allow_pickle=True).item()

video_to_feats = defaultdict(list)
video_to_times = defaultdict(list)

for key, feat in data.items():
    parts = key.split("_")
    if len(parts) < 3:
        continue

    video_id = "_".join(parts[:-2])

    try:
        start = float(parts[-2])
        end = float(parts[-1])
    except:
        continue

    video_to_feats[video_id].append(feat)
    video_to_times[video_id].append((start, end))

print(f"🎥 Found {len(video_to_feats)} videos")

# -----------------------------
# CLUSTER PER VIDEO
# -----------------------------
for video_id in tqdm(video_to_feats, desc="Clustering videos"):

    feats = np.stack(video_to_feats[video_id]).astype(np.float32)
    times = np.array(video_to_times[video_id], dtype=object)

    if len(feats) == 0:
        continue

    # ---- sort by time ----
    starts = np.array([t[0] for t in times], dtype=np.float32)
    order = np.argsort(starts)

    feats = torch.from_numpy(feats[order]).to(DEVICE)
    feats = F.normalize(feats, dim=1)
    times = times[order]

    # ---- temporal clustering ----
    clusters = []
    current = [0]

    for i in range(1, feats.shape[0]):
        sim = torch.dot(feats[i - 1], feats[i]).item()
        if sim >= CLUSTER_SIM_THRESHOLD:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)

    # ---- merge if too many clusters ----
    while len(clusters) > MAX_RATIO * len(clusters):
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

    # ---- build pseudo-steps ----
    step_feats = []
    step_times = []

    for c in clusters:
        step_feats.append(feats[c].mean(dim=0).cpu().numpy())
        step_times.append((times[c[0]][0], times[c[-1]][1]))

    step_feats = np.stack(step_feats)              # (K, 1792)
    step_times = np.array(step_times, dtype=object)  # (K, 2)

    # ---- save SINGLE FILE ----
    out_path = os.path.join(OUT_DIR, f"{video_id}_steps.npz")

    np.savez(
        out_path,
        video_id=video_id,
        step_features=step_feats,
        step_times=step_times
    )

print("✅ Clustering completed. One file per video saved.")
