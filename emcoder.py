import torch
import torch.nn as nn
from safetensors.torch import load_file
from pydantic import BaseModel, model_validator, field_validator


class ModelConfig(BaseModel):
    vocab_size: int
    max_seq_len: int
    
    d_model: int
    n_head: int
    n_layers: int
    d_ffn: int
    
    dropout: float

    num_labels: int
    id2label: dict[int, str]
    label2id: dict[str, int]

    base_encoder_path: str

    @field_validator("id2label", mode="before")
    @classmethod
    def coerce_keys_to_int(cls, v):
        return {int(k): val for k, val in v.items()}

    @model_validator(mode='after')
    def check_consistency(self):
        if len(self.id2label) != self.num_labels:
            raise ValueError("num_labels does not match id2label dictionary len")
        return self




class EmCoderCore(nn.Module):
    """The core encoder architecture of EmCoder, without the classification head."""
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )
        self.pos_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
        )

        self.embed_norm = nn.LayerNorm(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_ffn,
            dropout=config.dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.n_layers
        )
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)



class EmCoder(nn.Module):
    """The full EmCoder model, including the classification head."""
    def __init__(self, encoder: EmCoderCore, config: ModelConfig):
        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_labels)
        )
    

    def _set_mc_dropout(self, active: bool = True):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(active)


    @classmethod
    def from_pretrained(cls, emcoder_path: str):
        """Loads the EmCoder model from the specified directory."""
        # Use model_config.json to initialize same parameterers as in training
        with open(f"{emcoder_path}/model_config.json", "r") as f:
            model_config = ModelConfig.model_validate_json(f.read())

        encoder = EmCoderCore(model_config)
        model = cls(encoder, model_config)
        
        state_dict = load_file(f"{emcoder_path}/model.safetensors")
        model.load_state_dict(state_dict, strict=True)
        return model


    @staticmethod
    def _masked_mean_pooling(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)  # (B, S, 1)
        masked_features = features * mask  # (B, S, D)
        sum_masked_features = masked_features.sum(dim=1)  # (B, D)
        count_tokens = torch.clamp(mask.sum(dim=1), min=1e-9)  # (B, 1)
        return sum_masked_features / count_tokens  # (B, D)


    def mc_forward(self, x: torch.Tensor, mask: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Performs Monte Carlo Dropout inference to quantify epistemic uncertainty."""
        self._set_mc_dropout(active=True)

        B, S = x.shape
        x_stacked = x.repeat(n_samples, 1) # (n_samples * B, S)
        mask_stacked = mask.repeat(n_samples, 1)

        features = self.encoder(x_stacked, mask_stacked)
        pooled = self._masked_mean_pooling(features, mask_stacked)
        logits = self.classifier(pooled) # (n_samples * B, num_labels)

        return logits.view(n_samples, B, -1)


    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Standard forward pass without MC Dropout."""
        features = self.encoder(x, mask)

        pooled = self._masked_mean_pooling(features, mask)
        return self.classifier(pooled)