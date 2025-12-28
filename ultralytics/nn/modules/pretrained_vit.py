import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load

# Compatibility: PyTorch<2.0 does not expose F.scaled_dot_product_attention (or has a different internal signature).
# DINOv2 expects torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
# is_causal=False, scale=None) -> attn_output
if not hasattr(F, "scaled_dot_product_attention"):

    def _sdpa_fallback(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        """Scaled dot-product attention fallback for torch<2.0.

        Notes:
            - DINOv2 often runs under AMP. A naive FP16 softmax attention can overflow and produce NaNs.
            - We compute attention weights in FP32 with a stable softmax, then cast back.
        """

        # query/key/value: (..., L, E) and (..., S, E) typically (B, heads, seq, head_dim)
        orig_dtype = query.dtype
        q = query.float()
        k = key.float()
        v = value.float()

        d = q.size(-1)
        scale_factor = (1.0 / (d**0.5)) if scale is None else float(scale)

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale_factor  # (..., L, S)

        if is_causal:
            l, s = attn.size(-2), attn.size(-1)
            causal = torch.ones((l, s), device=attn.device, dtype=torch.bool).triu(1)
            attn = attn.masked_fill(causal, float("-inf"))

        if attn_mask is not None:
            # Support bool masks (True=mask) and additive masks (float with -inf)
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(attn_mask, float("-inf"))
            else:
                attn = attn + attn_mask.float()

        # Stable softmax: subtract max over last dim, and guard fully-masked rows.
        finite = torch.isfinite(attn)
        row_has_finite = finite.any(dim=-1, keepdim=True)
        attn_safe = torch.where(finite, attn, torch.tensor(-1e9, device=attn.device, dtype=attn.dtype))
        attn_safe = attn_safe - attn_safe.max(dim=-1, keepdim=True).values
        weights = torch.softmax(attn_safe, dim=-1)
        weights = weights * row_has_finite.to(weights.dtype)  # fully-masked rows -> all zeros

        if dropout_p and dropout_p > 0 and torch.is_grad_enabled():
            weights = torch.dropout(weights, p=dropout_p, train=True)

        out = torch.matmul(weights, v)
        return out.to(orig_dtype)

    F.scaled_dot_product_attention = _sdpa_fallback

# from aim.v2.utils import load_pretrained

dino_backbones = {
    "small": {"name": "dinov2_vits14_reg", "embedding_size": 384, "patch_size": 14},
    "base": {"name": "dinov2_vitb14_reg", "embedding_size": 768, "patch_size": 14},
    "large": {"name": "dinov2_vitl14_reg", "embedding_size": 1024, "patch_size": 14},
    "giant": {"name": "dinov2_vitg14_reg", "embedding_size": 1536, "patch_size": 14},
    "dinov3_convnext_large": {"name": "dinov3_convnext_large", "embedding_size": 1536, "patch_size": 14},
    "dinov3_vits16": {"name": "dinov3_vits16", "embedding_size": 384, "patch_size": 16},
    "dinov3_vit7b16": {"name": "dinov3_vit7b16", "embedding_size": 4096, "patch_size": 16},
    "dinov3_vith16plus": {"name": "dinov3_vith16plus", "embedding_size": 1280, "patch_size": 16},
    
}


class DinoV2Patches(nn.Module):
    def __init__(self, in_chanels=3, out_channels=None, size="base", freeze=True):
        from torchvision import transforms

        super(DinoV2Patches, self).__init__()
        self.size = size
        self.freeze = bool(freeze)

        if self.size not in dino_backbones:
            raise KeyError(f"Unknown size '{self.size}'. Options: {list(dino_backbones.keys())}")

        model_name = dino_backbones[self.size]["name"]
        self.patch_size = int(dino_backbones[self.size]["patch_size"])
        embed_dim = int(dino_backbones[self.size]["embedding_size"])
        self.out_channels = int(out_channels) if out_channels is not None else embed_dim

        if self.out_channels != embed_dim:
            raise ValueError(
                f"out_channels({self.out_channels}) must match backbone embedding_size({embed_dim}) for size='{self.size}'."
            )

        # Choose repo source:
        # - DINOv2: load from GitHub torch.hub (auto-download)
        # - DINOv3: load from local repo folder containing hubconf.py (because repo naming/availability may vary)
        if self.size.startswith("dinov3_"):
            hub_dir = self._resolve_local_repo("dinov3")
            try:
                self.backbone = load(hub_dir, model_name, pretrained=True, source="local", weights=f"{model_name}.pth")
            except ImportError as e:
                # Common failure: older torchvision without transforms.v2
                raise ImportError(
                    "Failed to import dependencies when loading DINOv3 from local torch.hub repo.\n"
                    "If the error mentions 'torchvision.transforms.v2', please upgrade torchvision in your training env.\n"
                    f"Original error: {e}"
                )
        else:
            # DINOv2 GitHub torch.hub fallback (requires internet on first run)
            self.backbone = load("facebookresearch/dinov2", model_name, pretrained=True)

        # Freeze control
        self._apply_freeze()
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self.inet_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def _apply_freeze(self):
        """Apply freeze setting to backbone params and mode."""
        if self.freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            # Let backbone be trainable; actual train/eval mode will follow this module's mode.
            for p in self.backbone.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True):
        """Override to keep backbone frozen in eval when freeze=True."""
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        else:
            self.backbone.train(mode)
        return self

    @staticmethod
    def _resolve_local_repo(repo_name: str) -> str:
        """Find local torch.hub repo folder containing hubconf.py."""
        candidates = []

        env_key = "DINOV3_HUB_DIR" if repo_name == "dinov3" else "DINOV2_HUB_DIR"
        env = os.getenv(env_key)
        if env:
            candidates.append(Path(env))

        # Common locations: <project_root>/<repo_name> and CWD/<repo_name>
        try:
            project_root = Path(__file__).resolve().parents[3]
            candidates.append(project_root / repo_name)
        except Exception:
            pass
        candidates.append(Path.cwd() / repo_name)

        for p in candidates:
            if (p / "hubconf.py").is_file():
                return str(p)

        raise FileNotFoundError(
            f"Local torch.hub repo '{repo_name}' not found (missing hubconf.py).\n"
            f"Put it at <this_project>/{repo_name}/hubconf.py or set env var {env_key}=D:/path/to/{repo_name}."
        )

    def transform(self, x):
        # x should have shape (B, C, H, W)
        b, c, h, w = x.shape

        # Crop to make H/W multiples of patch size
        p = self.patch_size
        h_new = h - (h % p)
        w_new = w - (w % p)

        dh = h - h_new  # total pixels to drop in height
        dw = w - w_new  # total pixels to drop in width

        # Decide how many pixels to drop from each side.
        # For simplicity, drop half from each edge if dh/dw are even.
        # If dh or dw are odd, we'll drop one extra pixel from the "bottom/right".
        dh_top = dh // 2
        dh_bottom = dh - dh_top
        dw_left = dw // 2
        dw_right = dw - dw_left

        # Crop the tensor from edges
        # Make sure slicing indices do not go negative
        x_cropped = x[:, :, dh_top : h - dh_bottom, dw_left : w - dw_right]

        # normalize with imagenet mean and std
        x_cropped = self.inet_norm(x_cropped)

        return x_cropped

    def forward(self, x):
        x = self.transform(x)
        batch_size = x.shape[0]
        h_tokens = x.shape[2] // self.patch_size
        w_tokens = x.shape[3] // self.patch_size

        if self.freeze:
            with torch.no_grad():
                feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone.forward_features(x)

        x = feats["x_norm_patchtokens"]  # (B, N, C)
        x = x.permute(0, 2, 1)  # (B, C, N)
        x = x.reshape(batch_size, self.out_channels, int(h_tokens), int(w_tokens))
        return x
