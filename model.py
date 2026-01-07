import os
import torch
import torch.nn as nn
from transformers import AutoModel

from base import BaseModel

class FrozenInTime(BaseModel):
    """
    TEXT-ONLY version of FrozenInTime.
    Video branch completely disabled.
    """

    def __init__(self,
                 text_params,
                 projection_dim=256,
                 projection='minimal',
                 load_checkpoint=None):
        super().__init__()

        self.text_params = text_params

        if not text_params.get('pretrained', True):
            raise NotImplementedError(
                "Huggingface text models require pretrained init."
            )

        # -------------------------
        # Text encoder
        # -------------------------
        model_name = text_params['model']

        if model_name.startswith('distilbert'):
            self.text_model = AutoModel.from_pretrained(
                'distilbert-base-uncased',
                cache_dir='pretrained/distilbert-base-uncased'
            )
        else:
            self.text_model = AutoModel.from_pretrained(model_name)

        self.text_model.train()

        # -------------------------
        # Projection head (text)
        # -------------------------
        hidden_size = self.text_model.config.hidden_size

        if projection == 'minimal':
            self.txt_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_size, projection_dim),
            )
        elif projection == '':
            self.txt_proj = nn.Identity()
        else:
            raise NotImplementedError

        # -------------------------
        # Optional checkpoint load
        # -------------------------
        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.load_state_dict(state_dict, strict=False)

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------
    def forward(self, data, return_embeds=True):
        """
        data = {
            "text": {
                "input_ids": Tensor,
                "attention_mask": Tensor
            }
        }
        """
        text_data = data['text']
        text_embeddings = self.compute_text(text_data)

        return text_embeddings

    # ------------------------------------------------
    # Text encoding
    # ------------------------------------------------
    def compute_text(self, text_data):
        if self.text_params['model'].startswith('bert'):
            outputs = self.text_model(
                input_ids=text_data['input_ids'],
                attention_mask=text_data['attention_mask']
            )
            text_embeddings = outputs.pooler_output

        elif self.text_params['model'].startswith('distilbert'):
            outputs = self.text_model(**text_data)
            text_embeddings = outputs.last_hidden_state[:, 0, :]

        else:
            raise NotImplementedError(
                f"Text model {self.text_params['model']} not supported"
            )

        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_text_tokens(self, text_data):
        """
        Returns token-level embeddings (optional utility).
        """
        outputs = self.text_model(**text_data)
        token_embeddings = outputs.last_hidden_state
        token_embeddings = self.txt_proj(token_embeddings)
        return token_embeddings


if __name__ == "__main__":
    pass
