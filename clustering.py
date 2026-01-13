import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances

# ============================================================
# PATHS (LOCAL)
# ============================================================

INPUT_FEATURES = "output_subtask_1.npy"
GRAPH_DIR     = Path("task_graphs")
CSV_PATH      = "activity_idx_step_idx.csv"
OUTPUT_DIR    = Path("video_embeddings")

# hyperparameters
FINE_THRESHOLD = 0.15

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD CSV MAPPING: activity_idx -> activity_name
# ============================================================

df = pd.read_csv(CSV_PATH)

id_to_recipe = (
    df[["activity_idx", "activity_name"]]
    .drop_duplicates()
    .set_index("activity_idx")["activity_name"]
    .to_dict()
)

# ============================================================
# UTILS
# ============================================================

def parse_key(k: str):
    """
    '9_4_360p_211.20_216.00'
      -> recipe_id='9'
      -> video_id='4'
      -> start, end
    """
    parts = k.split("_")
    recipe_id = parts[0]
    video_id  = parts[1]
    start = float(parts[-2])
    end   = float(parts[-1])
    return recipe_id, video_id, start, end


def load_features(path):
    return np.load(path, allow_pickle=True).item()


def group_by_recipe_video(data):
    groups = defaultdict(list)

    for k, feat in data.items():
        recipe_id, video_id, start, end = parse_key(k)
        rv_id = f"{recipe_id}_{video_id}"
        groups[rv_id].append((start, end, k, feat))

    for rv_id in groups:
        groups[rv_id].sort(key=lambda x: x[0])

    return groups


def get_task_graph(recipe_id: int):
    """
    Returns (graph_json, safe_name) or (None, None)
    """
    if recipe_id not in id_to_recipe:
        return None, None

    recipe_name = id_to_recipe[recipe_id]
    safe_name = recipe_name.lower().replace(" ", "")
    graph_path = GRAPH_DIR / f"{safe_name}.json"

    if not graph_path.exists():
        return None, None

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    return graph, safe_name


# ============================================================
# STEP 1 — FINE TEMPORAL SEGMENTATION
# ============================================================

def temporal_clustering(windows, threshold):
    """
    Merge ONLY consecutive windows if cosine distance is small
    """
    if len(windows) == 1:
        return [windows]

    feats = normalize(np.stack([w[3] for w in windows]))

    clusters = []
    current = [windows[0]]

    for i in range(1, len(windows)):
        d = cosine_distances(
            feats[i - 1][None],
            feats[i][None]
        )[0, 0]

        if d < threshold:
            current.append(windows[i])
        else:
            clusters.append(current)
            current = [windows[i]]

    clusters.append(current)
    return clusters


# ============================================================
# STEP 2 — MERGE TO TARGET (INCLUDING START + END)
# ============================================================

def cluster_embedding(cluster):
    feats = np.stack([w[3] for w in cluster])
    return feats.mean(axis=0)


def reduce_clusters_to_target(clusters, target_steps):
    """
    Merge most similar ADJACENT clusters until target reached
    """
    if len(clusters) <= target_steps:
        return clusters

    clusters = list(clusters)

    while len(clusters) > target_steps:
        embs = normalize(
            np.stack([cluster_embedding(c) for c in clusters])
        )

        dists = [
            cosine_distances(
                embs[i][None],
                embs[i + 1][None]
            )[0, 0]
            for i in range(len(embs) - 1)
        ]

        i = int(np.argmin(dists))
        merged = clusters[i] + clusters[i + 1]
        clusters = clusters[:i] + [merged] + clusters[i + 2:]

    return clusters


# ============================================================
# SAVE
# ============================================================

def save_recipe_video(rv_id, clusters):
    step_embeddings = []
    step_intervals  = []
    window_keys     = []

    for c in clusters:
        feats = np.stack([w[3] for w in c])
        step_embeddings.append(feats.mean(axis=0))
        step_intervals.append([c[0][0], c[-1][1]])
        window_keys.append([w[2] for w in c])

    step_embeddings = np.stack(step_embeddings)
    step_intervals  = np.array(step_intervals, dtype=np.float32)

    out_path = OUTPUT_DIR / f"{rv_id}_360p_steps.npz"
    np.savez(
        out_path,
        step_embeddings=step_embeddings,
        step_intervals=step_intervals,
        window_keys=np.array(window_keys, dtype=object),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    data = load_features(INPUT_FEATURES)
    groups = group_by_recipe_video(data)

    print(f"Loaded {len(data)} sliding windows")
    print(f"Found {len(groups)} recipe_videos\n")

    for rv_id, windows in groups.items():
        recipe_id = int(rv_id.split("_")[0])

        graph, _ = get_task_graph(recipe_id)
        if graph is None:
            print(f"[WARN] Missing task graph for recipe {recipe_id}")
            continue

        # total nodes INCLUDING START + END
        target_steps = len(graph["steps"])

        # --------------------------------------------------
        # Step 1: fine temporal segmentation
        # --------------------------------------------------
        fine_clusters = temporal_clustering(
            windows,
            threshold=FINE_THRESHOLD
        )

        # --------------------------------------------------
        # Step 2: merge to EXACT number of graph nodes
        # --------------------------------------------------
        clusters = reduce_clusters_to_target(
            fine_clusters,
            target_steps=target_steps
        )

        # --------------------------------------------------
        # Add START / END anchors if missing
        # --------------------------------------------------
        if len(clusters) == target_steps - 2:
            clusters = (
                [[windows[0]]] +
                clusters +
                [[windows[-1]]]
            )

        print("=" * 80)
        print(f"Recipe-Video: {rv_id}")
        print(f"Original windows: {len(windows)}")
        print(f"Graph nodes (incl. START/END): {target_steps}")
        print(f"Final clustered steps: {len(clusters)}")

        save_recipe_video(rv_id, clusters)


if __name__ == "__main__":
    main()
