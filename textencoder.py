import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer

# --------------------------------------------------
# PATH: assicurati che questo punti alla repo EgoVLP
# --------------------------------------------------
import sys
sys.path.append("/content/EgoVLP")  # cambia se serve

from EgoVLP.model.model import FrozenInTime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# 1. Inizializza TEXT ENCODER EgoVLP
# --------------------------------------------------
text_params = {
    "model": "distilbert-base-uncased",
    "pretrained": True,
}

model = FrozenInTime(
    text_params=text_params,
    projection_dim=256
).to(DEVICE)

model.eval()

# ⚠️ SE hai un checkpoint EgoVLP, caricalo QUI
# ckpt = torch.load("egovlp_ckpt.pt", map_location=DEVICE)
# model.load_state_dict(ckpt["state_dict"], strict=False)

# --------------------------------------------------
# 2. Tokenizer
# --------------------------------------------------
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

# --------------------------------------------------
# 3. Testi di esempio (METTI QUI I TUOI)
# --------------------------------------------------
texts = [
    "crack the eggs into a bowl",
    "whisk the eggs thoroughly",
    "heat the pan with olive oil"
]

# --------------------------------------------------
# 4. Encoding testuale
# --------------------------------------------------
with torch.no_grad():
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    text_feats = model.compute_text(enc)     # (N, 256)
    text_feats = F.normalize(text_feats, dim=1)

# --------------------------------------------------
# 5. STAMPA RISULTATI
# --------------------------------------------------
print("Text embeddings shape:", text_feats.shape)
print("Text embeddings dtype:", text_feats.dtype)

print("\nFirst embedding (first 10 dims):")
print(text_feats[0][:10])

print("\nAll embeddings:")
print(text_feats)
