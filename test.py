import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ============================================================
# PATH: MODIFICA QUI
# ============================================================
GRAPH_PATH = "./graph_realizations/1_7_360p_graph.npz"

# ============================================================
# LOAD GRAPH
# ============================================================
g = np.load(GRAPH_PATH, allow_pickle=True)

node_features   = g["node_features"]        # (N, 1792)
edges           = g["edges"]                # (E, 2)
node_texts      = g["node_texts"]           # (N,)
matched_pairs   = g["matched_pairs"]        # (K, 2)
step_times      = g["step_times"]
original_ids    = g["original_node_ids"]

# ============================================================
# PRINT BASIC INFO
# ============================================================
print("=" * 80)
print("📦 Realized graph file:", GRAPH_PATH)
print("=" * 80)

N = node_features.shape[0]
E = edges.shape[0]
K = matched_pairs.shape[0]

print(f"Number of nodes        : {N}")
print(f"Number of edges        : {E}")
print(f"Matched visual steps   : {K}")
print(f"Node feature dimension : {node_features.shape[1]}")

print("\n--- Node feature stats ---")
print("min :", node_features.min())
print("max :", node_features.max())
print("mean:", node_features.mean())

# ============================================================
# PRINT ALL NODE TEXTS
# ============================================================
print("\n" + "=" * 80)
print("🧾 TASK GRAPH NODES (compact_id → original_id → text)")
print("=" * 80)

for i, text in enumerate(node_texts):
    print(f"[{i:02d}] (orig {original_ids[i]:02d}) {text}")

# ============================================================
# PRINT ALL MATCHINGS (VERY IMPORTANT)
# ============================================================
print("\n" + "=" * 80)
print("🔗 VISUAL STEP → TASK NODE MATCHINGS")
print("=" * 80)

for idx, (visual_idx, node_idx) in enumerate(matched_pairs):
    t_start, t_end = step_times[visual_idx]
    node_text = node_texts[node_idx]
    orig_id   = original_ids[node_idx]

    print(
        f"[{idx:02d}] "
        f"Visual step {visual_idx:02d} "
        f"(t={t_start:.2f}-{t_end:.2f})  "
        f"→ Node {node_idx:02d} (orig {orig_id:02d}) "
        f": {node_text}"
    )

# ============================================================
# BUILD NETWORKX GRAPH
# ============================================================
G = nx.DiGraph()

for i, text in enumerate(node_texts):
    G.add_node(
        i,
        label=f"{i}: {text[:40]}{'...' if len(text) > 40 else ''}"
    )

for u, v in edges:
    G.add_edge(int(u), int(v))

# ============================================================
# DAG VERTICAL LAYOUT (TRUE TOPOLOGICAL ORDER)
# ============================================================
def dag_vertical_layout(G):
    topo = list(nx.topological_sort(G))
    depth = {}

    for n in topo:
        preds = list(G.predecessors(n))
        if not preds:
            depth[n] = 0
        else:
            depth[n] = max(depth[p] for p in preds) + 1

    levels = {}
    for n, d in depth.items():
        levels.setdefault(d, []).append(n)

    pos = {}
    for d, nodes in levels.items():
        for i, n in enumerate(nodes):
            pos[n] = (i, -d)   # top → bottom

    return pos

pos = dag_vertical_layout(G)

# ============================================================
# NODE COLORS (matched vs not matched)
# ============================================================
matched_nodes = set(matched_pairs[:, 1].tolist())

node_colors = [
    "tab:orange" if n in matched_nodes else "lightgray"
    for n in G.nodes()
]

# ============================================================
# DRAW GRAPH
# ============================================================
plt.figure(figsize=(14, 10))

nx.draw(
    G,
    pos,
    with_labels=True,
    labels=nx.get_node_attributes(G, "label"),
    node_color=node_colors,
    node_size=1500,
    edge_color="gray",
    arrowsize=18,
    font_size=9,
)

plt.title(
    "Realized Task Graph (Vertical DAG)\n"
    "Orange = matched nodes, Gray = unused nodes"
)

plt.tight_layout()
plt.show()

print("\n✅ Done.")
