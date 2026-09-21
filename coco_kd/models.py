# Adapted from effl-lab/MaskedKD models_student.py / models_teacher.py.
# Original: Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
# See vendor/MaskedKD/UPSTREAM.json for the exact source revision.
"""Torch-only DeiT adaptation preserving official parameter names and masking.

No timm registry or obsolete timm imports. Position embeddings are added before
token gathering; CLS is always retained. Attention is explicitly computed, as
in upstream, so the last-layer CLS attention is available during training.
"""
import torch
from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x)), attn


class Block(nn.Module):
    def __init__(self, dim, heads, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        return x + self.drop_path(self.mlp(self.norm2(x))), attn


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=16, stride=16)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DeiT(nn.Module):
    def __init__(self, dim, depth, heads, drop_path=0.0, num_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, dim))
        self.pos_drop = nn.Dropout(0)
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([Block(dim, heads, rate) for rate in rates])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x, indices=None, return_attention=False):
        if x.shape[-2:] != (224, 224):
            raise ValueError("This experiment requires 224 x 224 input")
        patches = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), patches), dim=1)
        x = x + self.pos_embed
        if indices is not None:
            patches = x[:, 1:].gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            x = torch.cat((x[:, :1], patches), dim=1)
        x = self.pos_drop(x)
        for block in self.blocks:
            x, attention = block(x)
        logits = self.head(self.norm(x)[:, 0])
        if return_attention:
            # Upstream losses.py: attn.mean(dim=1)[:, 0, 1:]
            return logits, attention.mean(dim=1)[:, 0, 1:]
        return logits


URLS = {
    "student": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    "teacher": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
}


def build_model(role, cfg, pretrained=False):
    if role not in URLS:
        raise ValueError(role)
    if cfg.model_scale == "debug":
        if pretrained:
            raise ValueError("Debug models cannot load ImageNet weights")
        dim, depth, heads = (48, 2, 3) if role == "teacher" else (24, 2, 3)
    else:
        dim, depth, heads = (384, 12, 6) if role == "teacher" else (192, 12, 3)
    model = DeiT(dim, depth, heads, cfg.drop_path, num_classes=cfg.num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(URLS[role], map_location="cpu", check_hash=True, weights_only=True)
        state = {k: v for k, v in checkpoint["model"].items() if not k.startswith("head.")}
        result = model.load_state_dict(state, strict=False)
        if set(result.missing_keys) != {"head.weight", "head.bias"} or result.unexpected_keys:
            raise RuntimeError(f"Unexpected pretrained checkpoint mismatch: {result}")
    return model
