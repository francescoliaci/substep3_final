import os, sys
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer

# adjust to your repo
sys.path.append("/content")

from EgoVLP.model.model import FrozenInTime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# REQUIRED PARAMS (even if you only use text)
# -----------------------------
video_params = {
    "model": "SpaceTimeTransformer",
    "pretrained": False,          # ok; ckpt will overwrite
    "num_frames": 4,
    "time_init": "zeros",
    "attention_style": "frozen-in-time",
    "arch_config": "base_patch16_224",
    "vit_init": "imagenet-21k",
}

text_params = {
    "model": "distilbert-base-uncased",
    "pretrained": True,
}

# -----------------------------
# PUT YOUR EGOVLP CHECKPOINT HERE
# -----------------------------
EGOVLP_CKPT = "/content/drive/MyDrive/AMLproject/pretrained/EgoVLP_PT_BEST.pth"

model = FrozenInTime(
    video_params=video_params,
    text_params=text_params,
    projection_dim=256,
    load_checkpoint=EGOVLP_CKPT,   # <-- this is the key
    projection="minimal",
).to(DEVICE)

model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

texts = [
    "crack the eggs into a bowl",
    "whisk the eggs thoroughly",
    "heat the pan with olive oil",
]

with torch.no_grad():
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    t = model.compute_text(enc)          # (N, 256)
    t = F.normalize(t, dim=1)

print("shape:", t.shape, "dtype:", t.dtype)
print("first vector first 10 dims:", t[0, :10])
print("cos sims:")
print("01:", float((t[0] @ t[1]).item()))
print("02:", float((t[0] @ t[2]).item()))
print("12:", float((t[1] @ t[2]).item()))
