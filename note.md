
### For each recipe video, you will:
1. Take the ground-truth task graph (text steps).
2. Encode those text steps with the text encoder (EgoVLP or PE) → node features.
3. Take the visual step embeddings from Substep 1.
4. Use Hungarian matching to align each visual step with at most one graph node.
5. For matched pairs, combine text+visual features with a learnable projection to get updated node features.

The result is a graph where nodes now represent how this specific video executed each step (a “realization” of the task graph).

# 1st step
### For one recipe video, you need:
Input:
From CaptainCook4D you should have something like:
- A list of steps for the recipe, e.g.:
    step_texts = [
        "Whisk the yogurt until smooth",
        "Peel the cucumber",
        "Chop the cucumber into small pieces",
        "Add salt",
        "Add cumin",
        ...
    ]
- The graph structure, e.g. as an edge list:
    edges = [
        (0, 1),  # step 0 → step 1
        (1, 2),
        (2, 3),
        (3, 4),
        ...
    ]

For the same video, from Substep 1 you should already have:
    visual_step_features = [v1, v2, ..., vK]  # list or tensor of shape (K, D)

---
# 2nd step
EgoVLP produces multimodal embeddings, meaning the text encoder and video encoder map their inputs into the same semantic vector space.

When you run egovlp_model.encode_text(step_texts), each step description becomes a D-dimensional vector that represents its meaning (e.g., “chop”, “add”, “mix”).
When you extract video features from Substep 1, each visual step also becomes a D-dimensional vector in that same space.

The reason they are aligned is that EgoVLP is trained with a contrastive objective:
True (video, text) pairs are pushed close together in the embedding space.
Mismatched pairs are pushed far apart.
As a result:
A video of “chopping carrots” is close to the text “chop carrots”.
A video of “adding salt” is far from the text “flip the pancake”.

---
# 3rd - 4th step
This step connects:

visual step embeddings from Substep 1:
visual_step_feats → shape (K, D)

text node embeddings from the task graph:
node_text_feats → shape (N, D)

Our goal is to compute a K × N similarity matrix and then perform 1-to-1 assignment between visual steps and task-graph nodes using the Hungarian algorithm.

> Visual steps and text steps are both D-dimensional vectors normalized to unit length 
> (we normalized the text embeddings already, 
> ! and you should also normalize the visual step embeddings).

first of all, we compute a _cosine similarity matrix_
Cosine similarity is a measure of how similar two vectors are based on the angle between them, not on their magnitude.
    cosine_sim(a,b) = a * b / (|a|*|b|)
It returns a value between:
    +1 → vectors point in the same direction → very similar
    0 → vectors are orthogonal → unrelated
    –1 → vectors point in opposite directions → very different

The Hungarian algorithm (more below) solves:
a minimum-cost assignment problem

But we want to maximize similarity, not minimize it.
So we convert similarity → cost just taking the negative of the similarity matrix

The _Hungarian algorithm_ finds:
>    The optimal one-to-one matching between two sets, minimizing total cost.

the output looks like this:
visual step 0 → node 7  
visual step 1 → node 1  
visual step 2 → node 3  
visual step 3 → node 5  
...

matches = [
    (v_idx_0, n_idx_0),
    (v_idx_1, n_idx_1),
    ...
]

---
# 5th step
we want to fuse:
node_text_feats[n_idx]   (text meaning)
visual_step_feats[v_idx] (video execution)

and produce a new node embedding:
updated_node_feats[n_idx]

This updated node is what you will later feed to the GNN classifier (Substep 4).

“update the features of the matched nodes with a learnable projection of the node features and the visual features.”
this means:
1. Concatenate:[text_embedding ; visual_embedding]
2. Pass through a small neural network (MLP)
3. Get a new embedding in the same dimension D

Now:
updated_node_feats: shape (N, D)

This is the realized task-graph node feature matrix.
These go into your GNN in Substep 4.





---
# about Substep 1
For each recording:
1. Load the 1-second clip features (.npz file).
2. Run (or load) ActionFormer predictions → list of (start, end) step intervals.
3. For each interval, average the clip features whose timestamps fall inside.
4. Get a tensor visual_step_feats of shape (K, D) → one embedding per localized step.

For a single recording (say ID "10_7"), you need:

Chunk features file — e.g.
features_output/10_7_360p_224_1s_1s.npz

ActionFormer predictions for that recording — e.g. in a JSON like:

{
  "10_7": [
    {"start": 3.2, "end": 8.9},
    {"start": 8.9, "end": 16.0},
    ...
  ]
}


