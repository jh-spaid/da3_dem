#!/usr/bin/env python3
# train_da3_dem_ndsm_lora_feats.py
"""
Depth Anything V3 (DA3) fine-tune for aerial RGB -> DEM + nDSM
- Uses DA3 intermediate features (export_feat_layers) instead of only final 1ch depth
- Stats-based init for DEM shift/scale and nDSM scale (p95)
- LoRA applied only to "proper" attention projection Linear layers (qkv / proj), using regex on full module names
- Safer head init to avoid blob/fog:
    * dem_head: zeros
    * ndsm_head: zeros + negative bias (softplus near 0)
- Correct DEM clamp range uses dem_clamp_min~max (NOT 0~max)
- Optional DSM auxiliary loss (pred_dsm = dem + ndsm) for stability

NOTE:
- This script is written to be robust against small variations in DA3 output dict keys.
- You may need to adjust --feat_layers depending on the DA3 backbone depth. Start with "5,10,15,20" or "10,15,20".
"""

import os
import glob
import json
import random
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from depth_anything_3.api import DepthAnything3


# =========================================================
#                    UNet Adapter
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetAdapter(nn.Module):
    """
    Simple UNet-style adapter operating on feature maps (B,C,h,w).
    """
    def __init__(self, in_ch, base=64, out_ch=64):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.bottleneck = DoubleConv(base * 2, base * 4)
        self.dec2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.dec1 = DoubleConv(base * 2 + base, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottleneck(F.max_pool2d(e2, 2))

        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1)


# =========================================================
#                        LoRA
# =========================================================
class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear: base(x) + scaling * B(A(dropout(x)))
    """
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base module")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Linear(in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, out_features, bias=False)

        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _get_module_by_name(root: nn.Module, full_name: str):
    """
    Return (parent_module, attr_name, module) for a dotted module name.
    """
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    attr = parts[-1]
    return parent, attr, getattr(parent, attr)


def apply_lora_by_fullname_regex(
    root: nn.Module,
    pattern: str,
    r=8,
    alpha=16,
    dropout=0.0,
    verbose=False,
):
    """
    Apply LoRA to nn.Linear modules whose FULL module name matches regex `pattern`.

    Example pattern:
      r"(?:^|\\.)attn\\.(?:qkv|proj)$"
    """
    rx = re.compile(pattern)
    replaced = 0
    matched_names = []

    # Important: iterate named_modules but replace by parent reference
    for name, mod in list(root.named_modules()):
        if isinstance(mod, nn.Linear) and rx.search(name):
            matched_names.append(name)

    for name in matched_names:
        parent, attr, mod = _get_module_by_name(root, name)
        if isinstance(mod, nn.Linear):
            setattr(parent, attr, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
            if verbose:
                print(f"[LoRA] wrapped: {name}")

    return replaced


def mark_only_lora_trainable(module: nn.Module):
    # freeze all
    for p in module.parameters():
        p.requires_grad = False
    # unfreeze lora
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.weight.requires_grad = True
            m.lora_B.weight.requires_grad = True


def lora_stats(module: nn.Module):
    count = 0
    a_norm = 0.0
    b_norm = 0.0
    for m in module.modules():
        if isinstance(m, LoRALinear):
            count += 1
            a_norm += float(m.lora_A.weight.detach().norm().cpu())
            b_norm += float(m.lora_B.weight.detach().norm().cpu())
    return count, a_norm, b_norm


def list_linear_module_names(module: nn.Module, max_items=200):
    names = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            names.append(name)
            if len(names) >= max_items:
                break
    return names


def _remap_plain_to_lora_keys(ckpt_sd, model_sd):
    """
    Remap plain keys -> LoRA-wrapped base keys for common attention linears.

    We try broad patterns because DA3 variants may name attention differently.
    For any key that ends with ".weight"/".bias" and corresponds to a LoRA-wrapped module,
    we map:
      "...<linear>.weight" -> "...<linear>.base.weight"
    if that exists in model_sd.

    This helps loading older checkpoints (without LoRA) into LoRA model.
    """
    remapped = dict(ckpt_sd)
    mapped = 0
    to_delete = []

    for k, v in ckpt_sd.items():
        if not (k.endswith(".weight") or k.endswith(".bias")):
            continue

        nk = k.replace(".weight", ".base.weight").replace(".bias", ".base.bias")
        if nk != k and nk in model_sd:
            if getattr(model_sd[nk], "shape", None) == getattr(v, "shape", None):
                remapped[nk] = v
                to_delete.append(k)
                mapped += 1

    for k in to_delete:
        remapped.pop(k, None)
    return remapped, mapped


# =========================================================
#                    Stats init
# =========================================================
def estimate_dem_ndsm_stats(
    train_roots,
    split="train",
    dem_clamp_min=-200.0,
    dem_clamp_max=2000.0,
    ndsm_clamp_max=300.0,
    max_files=200,
    max_vals=2_000_000,
    seed=42,
):
    """
    Robust stats for DEM + nDSM init.
    Returns: dem_mean, dem_std, ndsm_mean, ndsm_std, ndsm_p95
    """
    rng = np.random.default_rng(seed)
    files = []
    for r in train_roots:
        r = Path(r)
        files += sorted(glob.glob(str(r / split / "*.npz")))
    if len(files) == 0:
        raise RuntimeError(f"No NPZ files found under train_roots/*/{split}/*.npz")

    if len(files) > max_files:
        files = list(rng.choice(files, size=max_files, replace=False))

    dem_vals = []
    ndsm_vals = []

    def _finite_mask(a):
        return np.isfinite(a) & (a != -9999.0)

    total = 0
    for f in files:
        try:
            d = np.load(f, allow_pickle=False)
            if ("dem" not in d.files) or ("dsm" not in d.files):
                continue

            dem = d["dem"].astype(np.float32, copy=False)
            dsm = d["dsm"].astype(np.float32, copy=False)
            m = d["mask"].astype(bool) if "mask" in d.files else None

            vm = _finite_mask(dem) & _finite_mask(dsm)
            if m is not None:
                vm = vm & m

            if not np.any(vm):
                continue

            dem_v = np.clip(dem[vm], dem_clamp_min, dem_clamp_max)
            ndsm_v = np.clip(dsm[vm] - dem[vm], 0.0, ndsm_clamp_max)

            dem_vals.append(dem_v)
            ndsm_vals.append(ndsm_v)

            total += dem_v.size
            if total >= max_vals:
                break
        except Exception:
            continue

    if len(dem_vals) == 0:
        raise RuntimeError("No valid DEM/DSM values found for initialization.")

    dem_all = np.concatenate(dem_vals, axis=0)
    ndsm_all = np.concatenate(ndsm_vals, axis=0)

    dem_mean = float(np.mean(dem_all))
    dem_std = float(np.std(dem_all))

    ndsm_mean = float(np.mean(ndsm_all))
    ndsm_std = float(np.std(ndsm_all))
    ndsm_p95 = float(np.percentile(ndsm_all, 95))

    print(f"[INIT] Collected {dem_all.size} pixels from up to {len(files)} files")
    print(f"[INIT] DEM  range: [{dem_all.min():.2f}, {dem_all.max():.2f}] m | mean={dem_mean:.2f} std={dem_std:.2f}")
    print(f"[INIT] nDSM range: [{ndsm_all.min():.2f}, {ndsm_all.max():.2f}] m | mean={ndsm_mean:.2f} std={ndsm_std:.2f} p95={ndsm_p95:.2f}")

    return dem_mean, dem_std, ndsm_mean, ndsm_std, ndsm_p95


# =========================================================
#                  Dataset
# =========================================================
class NPZDemNdsmDataset(Dataset):
    """
    NPZ keys:
      - image or rgb
      - dem
      - dsm
      - mask (optional)

    ndsm = clip(dsm - dem, 0, ndsm_clamp_max)
    """
    def __init__(
        self,
        root_dir,
        split="train",
        augment=True,
        target_size=518,
        dem_clamp_min=-200.0,
        dem_clamp_max=2000.0,
        ndsm_clamp_max=300.0,
    ):
        self.root = Path(root_dir)
        self.split = split
        self.augment = augment
        self.target_size = int(target_size)
        self.dem_clamp_min = float(dem_clamp_min)
        self.dem_clamp_max = float(dem_clamp_max)
        self.ndsm_clamp_max = float(ndsm_clamp_max)

        self.files = sorted(glob.glob(str(self.root / split / "*.npz")))
        if len(self.files) == 0:
            raise RuntimeError(f"No NPZ files under: {self.root / split}")

        print(f"  [{split}] {len(self.files)} samples (resize to {self.target_size}×{self.target_size})")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _ensure_hwc3(img):
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img

    @staticmethod
    def _to_chw_float01(img_hwc):
        if img_hwc.dtype == np.uint8:
            img = img_hwc.astype(np.float32) / 255.0
        else:
            img = img_hwc.astype(np.float32)
            if img.max() > 2.0:
                img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        return np.transpose(img, (2, 0, 1))

    @staticmethod
    def _auto_mask(dem, dsm, mask=None):
        vm = np.isfinite(dem) & np.isfinite(dsm) & (dem != -9999.0) & (dsm != -9999.0)
        if mask is not None:
            vm = vm & mask.astype(bool)
        return vm

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=False)

        img = d.get("image", d.get("rgb"))
        if img is None:
            raise KeyError(f"Missing 'image'/'rgb' in {self.files[idx]}")

        dem = d["dem"].astype(np.float32)
        dsm = d["dsm"].astype(np.float32)
        mask = d["mask"].astype(bool) if "mask" in d.files else None

        vm = self._auto_mask(dem, dsm, mask)
        # Keep mask as bool map
        mask = vm.astype(bool)

        dem = np.clip(dem, self.dem_clamp_min, self.dem_clamp_max)
        ndsm = np.clip(dsm - dem, 0.0, self.ndsm_clamp_max)

        img = self._ensure_hwc3(img)
        img = self._to_chw_float01(img)

        x = torch.from_numpy(img)                    # (3,H,W) float [0,1]
        y_dem = torch.from_numpy(dem).unsqueeze(0)   # (1,H,W)
        y_ndsm = torch.from_numpy(ndsm).unsqueeze(0) # (1,H,W)
        m = torch.from_numpy(mask).unsqueeze(0)      # (1,H,W) bool

        # Resize
        if x.shape[1] != self.target_size or x.shape[2] != self.target_size:
            x = F.interpolate(x.unsqueeze(0), size=(self.target_size, self.target_size),
                              mode="bilinear", align_corners=False).squeeze(0)
            y_dem = F.interpolate(y_dem.unsqueeze(0), size=(self.target_size, self.target_size),
                                  mode="bilinear", align_corners=False).squeeze(0)
            y_ndsm = F.interpolate(y_ndsm.unsqueeze(0), size=(self.target_size, self.target_size),
                                   mode="bilinear", align_corners=False).squeeze(0)
            m = F.interpolate(m.unsqueeze(0).float(), size=(self.target_size, self.target_size),
                              mode="nearest").squeeze(0).bool()

        # Augment
        if self.augment and self.split == "train":
            if random.random() < 0.5:
                x = torch.flip(x, [2]); y_dem = torch.flip(y_dem, [2]); y_ndsm = torch.flip(y_ndsm, [2]); m = torch.flip(m, [2])
            if random.random() < 0.5:
                x = torch.flip(x, [1]); y_dem = torch.flip(y_dem, [1]); y_ndsm = torch.flip(y_ndsm, [1]); m = torch.flip(m, [1])
            if random.random() < 0.5:
                k = random.randint(1, 3)
                x = torch.rot90(x, k=k, dims=[1, 2])
                y_dem = torch.rot90(y_dem, k=k, dims=[1, 2])
                y_ndsm = torch.rot90(y_ndsm, k=k, dims=[1, 2])
                m = torch.rot90(m, k=k, dims=[1, 2])

        return {"image": x, "dem": y_dem, "ndsm": y_ndsm, "mask": m, "path": str(self.files[idx])}


# =========================================================
#                     Losses
# =========================================================
def masked_huber(pred, gt, mask, delta=2.0):
    mask = mask > 0
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    diff = pred - gt
    absd = diff.abs()
    hub = torch.where(absd < delta, 0.5 * (diff ** 2) / delta, absd - 0.5 * delta)
    return hub[mask].mean()


def masked_grad_l1(pred, gt, mask):
    mask = (mask > 0).to(dtype=torch.float32)

    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    gt   = torch.nan_to_num(gt,   nan=0.0, posinf=0.0, neginf=0.0)

    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_g = gt  [..., :, 1:] - gt  [..., :, :-1]
    dy_g = gt  [..., 1:, :] - gt  [..., :-1, :]

    mx = mask[..., :, 1:] * mask[..., :, :-1]
    my = mask[..., 1:, :] * mask[..., :-1, :]

    dx = torch.nan_to_num(dx_p - dx_g, nan=0.0, posinf=0.0, neginf=0.0)
    dy = torch.nan_to_num(dy_p - dy_g, nan=0.0, posinf=0.0, neginf=0.0)

    lx = dx.abs() * mx
    ly = dy.abs() * my

    num = lx.sum(dtype=torch.float32) + ly.sum(dtype=torch.float32)
    denom = (mx.sum(dtype=torch.float32) + my.sum(dtype=torch.float32)).clamp(min=1.0)
    return num / denom


# =========================================================
#                 DA3 Feature helpers
# =========================================================
def _get_aux_dict(out):
    # out might be dict-like
    if isinstance(out, dict):
        if "aux" in out and isinstance(out["aux"], dict):
            return out["aux"]
        # some wrappers could use nested objects; fallback
    # attribute-based
    if hasattr(out, "aux") and isinstance(out.aux, dict):
        return out.aux
    return None


def _feat_to_bchw(feat):
    """
    Accepts DA3 aux feature forms and returns BCHW.
    Expected common shape from DA3: (B, N, h, w, C)
    Sometimes could be (B, h, w, C) or already (B, C, h, w)
    """
    if feat is None:
        return None
    if feat.ndim == 5:
        # (B,N,h,w,C) -> pick/avg views -> (B,h,w,C) -> (B,C,h,w)
        B, N, h, w, C = feat.shape
        if N == 1:
            f = feat[:, 0]
        else:
            # average across views
            f = feat.mean(dim=1)
        return f.permute(0, 3, 1, 2).contiguous()
    if feat.ndim == 4:
        # could be (B,h,w,C) or (B,C,h,w)
        if feat.shape[1] in (64, 128, 256, 512, 768, 1024) and feat.shape[2] > 4 and feat.shape[3] > 4:
            # likely BCHW already
            return feat
        # assume BHWC
        return feat.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Unsupported feat shape: {tuple(feat.shape)}")


# =========================================================
#                      Model
# =========================================================
class DA3DemNdsmModel(nn.Module):
    """
    DA3 adapted for DEM + nDSM regression using intermediate feature maps.
    """
    def __init__(
        self,
        model_name="depth-anything/DA3-BASE",
        train_backbone=False,

        # feature export
        feat_layers=(10, 15, 20),
        feat_reduce_ch=128,   # reduce concatenated channels to this
        adapter_base=64,
        adapter_out_ch=64,

        # LoRA
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_regex=r"(?:^|\.)(?:attn)\.(?:qkv|proj)$",

        # clamps
        dem_min=-200.0,
        dem_max=2000.0,
        ndsm_max=300.0,

        # stats init
        init_dem_scale=None,
        init_dem_shift=None,
        init_ndsm_scale=None,

        # head init
        ndsm_bias_init=-4.0,
    ):
        super().__init__()

        print(f"[DA3DemNdsm] Loading model: {model_name}")
        da3_wrapper = DepthAnything3.from_pretrained(model_name)
        self.net = da3_wrapper.model
        print(f"[DA3DemNdsm] Model type: {type(self.net).__name__}")

        self.feat_layers = list(feat_layers)
        self.use_lora = bool(use_lora)
        self.train_backbone = bool(train_backbone)

        # Choose module to apply LoRA / freeze
        backbone_mod = self.net.backbone if hasattr(self.net, "backbone") else (self.net.net if hasattr(self.net, "net") else self.net)

        # Apply LoRA selectively
        if self.use_lora:
            n = apply_lora_by_fullname_regex(
                backbone_mod,
                pattern=lora_regex,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                verbose=False,
            )
            print(f"[LoRA] Applied: {n} Linear layers | regex={lora_regex}")

            if n == 0:
                print("[LoRA] WARNING: No layers matched. Here are sample Linear names:")
                for nm in list_linear_module_names(backbone_mod, max_items=80):
                    print("  -", nm)
                raise RuntimeError("LoRA enabled but 0 Linear modules were wrapped. Adjust --lora_regex.")

        # Freeze backbone if not training backbone
        if not self.train_backbone:
            print("[DA3DemNdsm] Freezing backbone (train heads/adapter + LoRA if enabled)")
            if self.use_lora:
                mark_only_lora_trainable(backbone_mod)
            else:
                for p in backbone_mod.parameters():
                    p.requires_grad = False
        else:
            print("[DA3DemNdsm] Training full backbone")

        # We will build feature reducer lazily after we see one forward (channel counts).
        self._feat_reduce_ch = int(feat_reduce_ch)
        self.feat_reduce = None  # nn.Conv2d(in_ch_total, feat_reduce_ch, 1)

        # Adapter over reduced features
        self.shared_adapter = UNetAdapter(in_ch=self._feat_reduce_ch, base=adapter_base, out_ch=adapter_out_ch)

        # Heads
        self.dem_head = nn.Conv2d(adapter_out_ch, 1, 1)
        self.ndsm_head = nn.Conv2d(adapter_out_ch, 1, 1)

        # Head init to avoid blob/fog
        nn.init.zeros_(self.dem_head.weight)
        nn.init.zeros_(self.dem_head.bias)
        nn.init.zeros_(self.ndsm_head.weight)
        nn.init.constant_(self.ndsm_head.bias, float(ndsm_bias_init))

        # Stats-based affine for DEM
        dem_scale_init = 100.0 if init_dem_scale is None else float(init_dem_scale)
        dem_shift_init = 0.0   if init_dem_shift is None else float(init_dem_shift)
        self.dem_scale = nn.Parameter(torch.tensor(dem_scale_init))
        self.dem_shift = nn.Parameter(torch.tensor(dem_shift_init))

        # Stats-based scale for nDSM (p95)
        ndsm_scale_init = 30.0 if init_ndsm_scale is None else float(init_ndsm_scale)
        self.ndsm_scale = nn.Parameter(torch.tensor(ndsm_scale_init))

        # clamps
        self.dem_min = float(dem_min)
        self.dem_max = float(dem_max)
        self.ndsm_max = float(ndsm_max)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DA3DemNdsm] Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")
        print(f"[DA3DemNdsm] feat_layers={self.feat_layers} feat_reduce_ch={self._feat_reduce_ch}")
        print(f"[DA3DemNdsm] Init DEM scale={self.dem_scale.item():.3f} shift={self.dem_shift.item():.3f}")
        print(f"[DA3DemNdsm] Init nDSM scale={self.ndsm_scale.item():.3f} ndsm_bias_init={ndsm_bias_init:.2f}")

    def _build_feat_reduce_if_needed(self, feat_cat):
        """
        Create 1x1 conv reducer once we know in_ch_total.
        """
        if self.feat_reduce is not None:
            return
        in_ch = int(feat_cat.shape[1])
        self.feat_reduce = nn.Conv2d(in_ch, self._feat_reduce_ch, kernel_size=1, bias=True).to(feat_cat.device)
        nn.init.kaiming_normal_(self.feat_reduce.weight, nonlinearity="relu")
        nn.init.zeros_(self.feat_reduce.bias)
        print(f"[FeatReduce] built: {in_ch} -> {self._feat_reduce_ch}")

    def forward(self, x, return_debug=False):
        """
        x: (B,3,H,W) ImageNet-normalized
        Returns:
          dem  (B,1,H,W)
          ndsm (B,1,H,W)
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)  # (B,1,3,H,W) as your original code

        out = self.net(
            x,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=self.feat_layers,
            infer_gs=False,
        )

        aux = _get_aux_dict(out)
        if aux is None:
            raise RuntimeError("DA3 output has no aux features. export_feat_layers may not be supported in this model/version.")

        feats = []
        missing = []
        for k in self.feat_layers:
            key = f"feat_layer_{k}"
            if key not in aux:
                missing.append(key)
                continue
            f = _feat_to_bchw(aux[key])
            feats.append(f)

        if len(feats) == 0:
            raise RuntimeError(f"No requested features found. Missing keys: {missing}")

        # concat features along channel (B, sumC, h, w)
        feat_cat = torch.cat(feats, dim=1)

        # reduce channels to fixed size
        self._build_feat_reduce_if_needed(feat_cat)
        feat_red = F.relu(self.feat_reduce(feat_cat), inplace=True)

        # adapter
        feat_ad = self.shared_adapter(feat_red)

        # heads at low-res; upsample to original H,W
        dem_rel = self.dem_head(feat_ad)
        ndsm_rel = self.ndsm_head(feat_ad)

        # upsample to match input spatial size (x is B,1,3,H,W)
        H, W = x.shape[-2], x.shape[-1]
        dem_rel = F.interpolate(dem_rel, size=(H, W), mode="bilinear", align_corners=False)
        ndsm_rel = F.interpolate(ndsm_rel, size=(H, W), mode="bilinear", align_corners=False)

        # metric mapping
        dem = self.dem_scale * dem_rel + self.dem_shift
        dem = torch.clamp(dem, self.dem_min, self.dem_max)

        ndsm = self.ndsm_scale * F.softplus(ndsm_rel)  # >=0
        ndsm = torch.clamp(ndsm, 0.0, self.ndsm_max)

        if return_debug:
            # optionally return also feat stats
            return dem, ndsm, {"feat_cat": feat_cat, "feat_red": feat_red, "feat_ad": feat_ad, "dem_rel": dem_rel, "ndsm_rel": ndsm_rel}
        return dem, ndsm

    def unfreeze_backbone(self):
        print("[DA3DemNdsm] Unfreezing backbone (full fine-tuning)")
        for p in self.parameters():
            p.requires_grad = True
        self.train_backbone = True


# =========================================================
#                    Visualization
# =========================================================
@torch.no_grad()
def visualize_fixed_samples(model, dataset, indices, device, epoch, vis_dir):
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for idx in indices:
        s = dataset[idx]
        name = Path(s["path"]).stem

        x = s["image"].unsqueeze(0).to(device)
        y_dem = s["dem"].unsqueeze(0).to(device)
        y_ndsm = s["ndsm"].unsqueeze(0).to(device)
        m = s["mask"].unsqueeze(0).to(device)

        x_norm = (x - mean) / std
        p_dem, p_ndsm = model(x_norm)

        rgb = x.squeeze().cpu().numpy()
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = np.clip(rgb, 0, 1)

        dem_p = p_dem.squeeze().cpu().numpy()
        dem_g = y_dem.squeeze().cpu().numpy()
        nd_p = p_ndsm.squeeze().cpu().numpy()
        nd_g = y_ndsm.squeeze().cpu().numpy()
        valid = m.squeeze().cpu().numpy().astype(bool)

        if valid.sum() == 0:
            continue

        dem_comb = np.concatenate([dem_p[valid], dem_g[valid]])
        dem_vmin, dem_vmax = np.percentile(dem_comb, [2, 98])

        nd_comb = np.concatenate([nd_p[valid], nd_g[valid]])
        nd_vmin, nd_vmax = 0.0, float(np.percentile(nd_comb, 98))

        dem_err = np.ma.masked_where(~valid, np.abs(dem_p - dem_g))
        nd_err = np.ma.masked_where(~valid, np.abs(nd_p - nd_g))
        dem_err_vmax = np.percentile(dem_err.compressed(), 95) if dem_err.count() > 0 else 10
        nd_err_vmax = np.percentile(nd_err.compressed(), 95) if nd_err.count() > 0 else 10

        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.25)

        ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(rgb); ax0.set_title("RGB"); ax0.axis("off")

        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(dem_p, cmap="viridis", vmin=dem_vmin, vmax=dem_vmax)
        ax1.set_title("DEM Pred"); ax1.axis("off"); plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 2])
        im2 = ax2.imshow(dem_g, cmap="viridis", vmin=dem_vmin, vmax=dem_vmax)
        ax2.set_title("DEM GT"); ax2.axis("off"); plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 3])
        im3 = ax3.imshow(dem_err, cmap="hot", vmin=0, vmax=dem_err_vmax)
        ax3.set_title(f"|DEM Err| (95%={dem_err_vmax:.1f}m)"); ax3.axis("off"); plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(nd_p, cmap="magma", vmin=nd_vmin, vmax=nd_vmax)
        ax4.set_title("nDSM Pred"); ax4.axis("off"); plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(nd_g, cmap="magma", vmin=nd_vmin, vmax=nd_vmax)
        ax5.set_title("nDSM GT"); ax5.axis("off"); plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(nd_err, cmap="hot", vmin=0, vmax=nd_err_vmax)
        ax6.set_title(f"|nDSM Err| (95%={nd_err_vmax:.1f}m)"); ax6.axis("off"); plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

        ax7 = fig.add_subplot(gs[1, 3])
        dsm_p = dem_p + nd_p
        dsm_g = dem_g + nd_g
        dsm_comb = np.concatenate([dsm_p[valid], dsm_g[valid]])
        dsm_vmin, dsm_vmax = np.percentile(dsm_comb, [2, 98])
        im7 = ax7.imshow(dsm_p, cmap="viridis", vmin=dsm_vmin, vmax=dsm_vmax)
        ax7.set_title("DSM Pred (=DEM+nDSM)"); ax7.axis("off"); plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

        fig.suptitle(f"Epoch {epoch} - {name}", fontsize=14, weight="bold")
        plt.tight_layout()
        out = vis_dir / f"epoch{epoch:03d}_{name}.png"
        plt.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close()


# =========================================================
#                    Train / Val
# =========================================================
def _imagenet_norm(x, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def train_epoch(
    model, loader, optimizer, scaler, device,
    amp_enabled,
    # losses
    lambda_ndsm,
    lambda_dsm,
    grad_w_dem, grad_w_ndsm,
    huber_w_dem, huber_w_ndsm, huber_w_dsm,
    huber_delta_dem, huber_delta_ndsm, huber_delta_dsm,
    lora_log_state=None,
):
    model.train()

    totals = {
        "loss": 0.0,
        "dem_huber": 0.0, "dem_grad": 0.0,
        "ndsm_huber": 0.0, "ndsm_grad": 0.0,
        "dsm_huber": 0.0,
    }

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        dem = batch["dem"].to(device, non_blocking=True)
        ndsm = batch["ndsm"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)

        x = _imagenet_norm(x, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            p_dem, p_ndsm = model(x)

        # loss in fp32
        p_dem32 = p_dem.float()
        p_ndsm32 = p_ndsm.float()
        dem32 = dem.float()
        ndsm32 = ndsm.float()
        m32 = m.float()

        dem_h = masked_huber(p_dem32, dem32, m32, delta=huber_delta_dem)
        dem_g = masked_grad_l1(p_dem32, dem32, m32) if grad_w_dem > 0 else p_dem32.new_tensor(0.0)

        nd_h = masked_huber(p_ndsm32, ndsm32, m32, delta=huber_delta_ndsm)
        nd_g = masked_grad_l1(p_ndsm32, ndsm32, m32) if grad_w_ndsm > 0 else p_ndsm32.new_tensor(0.0)

        # optional DSM auxiliary (stability)
        dsm_h = p_dem32.new_tensor(0.0)
        if lambda_dsm > 0:
            p_dsm32 = p_dem32 + p_ndsm32
            gt_dsm32 = dem32 + ndsm32
            dsm_h = masked_huber(p_dsm32, gt_dsm32, m32, delta=huber_delta_dsm)

        loss_dem = huber_w_dem * dem_h + grad_w_dem * dem_g
        loss_nd  = huber_w_ndsm * nd_h + grad_w_ndsm * nd_g
        loss_dsm = huber_w_dsm * dsm_h

        loss = loss_dem + lambda_ndsm * loss_nd + lambda_dsm * loss_dsm

        if not torch.isfinite(loss):
            print("[BAD] loss NaN/Inf:", batch["path"][0])
            print("dem_h, dem_g, nd_h, nd_g, dsm_h:", float(dem_h), float(dem_g), float(nd_h), float(nd_g), float(dsm_h))
            break

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # one-time LoRA sanity
        if lora_log_state is not None and not lora_log_state.get("done", False):
            # locate backbone module
            backbone_mod = model.net.backbone if hasattr(model.net, "backbone") else (model.net.net if hasattr(model.net, "net") else model.net)
            count, a_norm, b_norm = lora_stats(backbone_mod)
            print(f"[LoRA] After 1 step: modules={count} | A_norm={a_norm:.4f} B_norm={b_norm:.4f}")
            lora_log_state["done"] = True

        totals["loss"] += loss.item()
        totals["dem_huber"] += dem_h.item()
        totals["dem_grad"] += dem_g.item()
        totals["ndsm_huber"] += nd_h.item()
        totals["ndsm_grad"] += nd_g.item()
        totals["dsm_huber"] += dsm_h.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dem_h": f"{dem_h.item():.3f}",
            "nd_h": f"{nd_h.item():.3f}",
            "λ_nd": f"{lambda_ndsm:.2f}",
            "λ_dsm": f"{lambda_dsm:.2f}",
        })

    n = max(1, len(loader))
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(
    model, loader, device,
    amp_enabled,
    lambda_ndsm,
    lambda_dsm,
    grad_w_dem, grad_w_ndsm,
    huber_w_dem, huber_w_ndsm, huber_w_dsm,
    huber_delta_dem, huber_delta_ndsm, huber_delta_dsm,
):
    model.eval()

    totals = {
        "loss": 0.0,
        "dem_huber": 0.0, "dem_grad": 0.0,
        "ndsm_huber": 0.0, "ndsm_grad": 0.0,
        "dsm_huber": 0.0,
    }

    for batch in tqdm(loader, desc="Validation"):
        x = batch["image"].to(device, non_blocking=True)
        dem = batch["dem"].to(device, non_blocking=True)
        ndsm = batch["ndsm"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)

        x = _imagenet_norm(x, device)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            p_dem, p_ndsm = model(x)

        p_dem32 = p_dem.float()
        p_ndsm32 = p_ndsm.float()
        dem32 = dem.float()
        ndsm32 = ndsm.float()
        m32 = m.float()

        dem_h = masked_huber(p_dem32, dem32, m32, delta=huber_delta_dem)
        dem_g = masked_grad_l1(p_dem32, dem32, m32) if grad_w_dem > 0 else p_dem32.new_tensor(0.0)

        nd_h = masked_huber(p_ndsm32, ndsm32, m32, delta=huber_delta_ndsm)
        nd_g = masked_grad_l1(p_ndsm32, ndsm32, m32) if grad_w_ndsm > 0 else p_ndsm32.new_tensor(0.0)

        dsm_h = p_dem32.new_tensor(0.0)
        if lambda_dsm > 0:
            p_dsm32 = p_dem32 + p_ndsm32
            gt_dsm32 = dem32 + ndsm32
            dsm_h = masked_huber(p_dsm32, gt_dsm32, m32, delta=huber_delta_dsm)

        loss_dem = huber_w_dem * dem_h + grad_w_dem * dem_g
        loss_nd  = huber_w_ndsm * nd_h + grad_w_ndsm * nd_g
        loss_dsm = huber_w_dsm * dsm_h

        loss = loss_dem + lambda_ndsm * loss_nd + lambda_dsm * loss_dsm

        totals["loss"] += loss.item()
        totals["dem_huber"] += dem_h.item()
        totals["dem_grad"] += dem_g.item()
        totals["ndsm_huber"] += nd_h.item()
        totals["ndsm_grad"] += nd_g.item()
        totals["dsm_huber"] += dsm_h.item()

    n = max(1, len(loader))
    return {k: v / n for k, v in totals.items()}


# =========================================================
#                        Main
# =========================================================
def parse_layers(s: str):
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    import argparse

    ap = argparse.ArgumentParser("DA3 RGB -> DEM + nDSM (features + LoRA + stats init)")

    # Data
    ap.add_argument("--train_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # Model
    ap.add_argument("--model_name", type=str, default="depth-anything/DA3-BASE")
    ap.add_argument("--train_backbone", action="store_true")
    ap.add_argument("--unfreeze_epoch", type=int, default=None)

    # Feature layers
    ap.add_argument("--feat_layers", type=str, default="10,15,20",
                    help="DA3 export_feat_layers, comma separated (e.g. 5,10,15,20)")
    ap.add_argument("--feat_reduce_ch", type=int, default=128)
    ap.add_argument("--adapter_base", type=int, default=64)
    ap.add_argument("--adapter_out_ch", type=int, default=64)

    # LoRA
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument(
        "--lora_regex",
        type=str,
        default=r"(?:^|\.)attn\.(?:qkv|proj)$",
        help=r"Regex on FULL module name for nn.Linear to wrap. "
             r"Default targets '...attn.qkv' and '...attn.proj' only."
    )

    # Training
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=4)

    # Sizes / clamps
    ap.add_argument("--target_size", type=int, default=518)
    ap.add_argument("--dem_clamp_min", type=float, default=-200.0)
    ap.add_argument("--dem_clamp_max", type=float, default=2000.0)
    ap.add_argument("--ndsm_clamp_max", type=float, default=300.0)

    # Loss
    ap.add_argument("--lambda_ndsm", type=float, default=0.2)
    ap.add_argument("--lambda_dsm", type=float, default=0.05, help="aux DSM loss weight (stability)")
    ap.add_argument("--grad_w_dem", type=float, default=0.05)
    ap.add_argument("--grad_w_ndsm", type=float, default=0.2)
    ap.add_argument("--huber_w_dem", type=float, default=5.0)
    ap.add_argument("--huber_w_ndsm", type=float, default=1.0)
    ap.add_argument("--huber_w_dsm", type=float, default=1.0)
    ap.add_argument("--huber_delta_dem", type=float, default=10.0)
    ap.add_argument("--huber_delta_ndsm", type=float, default=1.0)
    ap.add_argument("--huber_delta_dsm", type=float, default=10.0)

    # Viz
    ap.add_argument("--viz_samples", type=int, default=5)

    # Stats init
    ap.add_argument("--init_from_stats", action="store_true")
    ap.add_argument("--stats_max_files", type=int, default=200)
    ap.add_argument("--stats_max_vals", type=int, default=2_000_000)

    args = ap.parse_args()

    assert args.target_size % 14 == 0, f"target_size must be 14×N, got {args.target_size}"
    feat_layers = parse_layers(args.feat_layers)
    if len(feat_layers) == 0:
        raise ValueError("--feat_layers is empty. Provide like '10,15,20'")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations_fixed"
    vis_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("DA3 (features) -> DEM + nDSM | LoRA selective | stats init | blob-safe heads")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Input size: {args.target_size}×{args.target_size} (must be multiple of 14)")
    print(f"feat_layers: {feat_layers} | reduce_ch={args.feat_reduce_ch}")
    print(f"LoRA: use={args.use_lora} r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    print(f"LoRA regex: {args.lora_regex}")
    print(f"Loss: L = L_dem + {args.lambda_ndsm}*L_ndsm + {args.lambda_dsm}*L_dsm")
    print(f"DEM huber_w={args.huber_w_dem}, grad_w={args.grad_w_dem}, delta={args.huber_delta_dem}")
    print(f"nDSM huber_w={args.huber_w_ndsm}, grad_w={args.grad_w_ndsm}, delta={args.huber_delta_ndsm}")
    print(f"DSM aux huber_w={args.huber_w_dsm}, delta={args.huber_delta_dsm}")

    # Stats init
    init_dem_shift = None
    init_dem_scale = None
    init_ndsm_scale = None
    if args.init_from_stats:
        dem_mean, dem_std, ndsm_mean, ndsm_std, ndsm_p95 = estimate_dem_ndsm_stats(
            args.train_roots,
            split="train",
            dem_clamp_min=args.dem_clamp_min,
            dem_clamp_max=args.dem_clamp_max,
            ndsm_clamp_max=args.ndsm_clamp_max,
            max_files=args.stats_max_files,
            max_vals=args.stats_max_vals,
            seed=args.seed,
        )
        init_dem_shift = dem_mean
        init_dem_scale = max(dem_std, 1.0)
        init_ndsm_scale = max(ndsm_p95, 1.0)
        print(f"[INIT] Using stats init:")
        print(f"       DEM shift={init_dem_shift:.3f} scale={init_dem_scale:.3f}")
        print(f"       nDSM scale={init_ndsm_scale:.3f} (p95)")

    # Data
    print("\n[DATA] Loading datasets...")
    train_sets = [
        NPZDemNdsmDataset(
            r, "train", augment=True, target_size=args.target_size,
            dem_clamp_min=args.dem_clamp_min, dem_clamp_max=args.dem_clamp_max,
            ndsm_clamp_max=args.ndsm_clamp_max
        )
        for r in args.train_roots
    ]
    val_sets = [
        NPZDemNdsmDataset(
            r, "val", augment=False, target_size=args.target_size,
            dem_clamp_min=args.dem_clamp_min, dem_clamp_max=args.dem_clamp_max,
            ndsm_clamp_max=args.ndsm_clamp_max
        )
        for r in args.train_roots
    ]

    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    val_ds = ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]

    print(f"[DATA] Train: {len(train_ds)} samples")
    print(f"[DATA] Val:   {len(val_ds)} samples")

    fixed_indices = []
    if args.viz_samples > 0:
        rng = random.Random(args.seed)
        fixed_indices = rng.sample(range(len(val_ds)), min(args.viz_samples, len(val_ds)))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True
    )

    # Model
    print("\n[MODEL] Building model...")
    model = DA3DemNdsmModel(
        model_name=args.model_name,
        train_backbone=args.train_backbone,

        feat_layers=feat_layers,
        feat_reduce_ch=args.feat_reduce_ch,
        adapter_base=args.adapter_base,
        adapter_out_ch=args.adapter_out_ch,

        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_regex=args.lora_regex,

        dem_min=args.dem_clamp_min,
        dem_max=args.dem_clamp_max,
        ndsm_max=args.ndsm_clamp_max,

        init_dem_scale=init_dem_scale,
        init_dem_shift=init_dem_shift,
        init_ndsm_scale=init_ndsm_scale,

        ndsm_bias_init=-4.0,
    ).to(device)

    # Optional: show LoRA stats
    if args.use_lora:
        backbone_mod = model.net.backbone if hasattr(model.net, "backbone") else (model.net.net if hasattr(model.net, "net") else model.net)
        count, a_norm, b_norm = lora_stats(backbone_mod)
        print(f"[LoRA] Init: modules={count} | A_norm={a_norm:.4f} B_norm={b_norm:.4f}")

    # Optim
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))

    # Resume
    start_epoch = 1
    best_val = float("inf")
    history = []

    if args.resume and Path(args.resume).exists():
        print(f"\n[RESUME] Loading: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        try:
            model.load_state_dict(ckpt_model, strict=True)
            print("[RESUME] Model state loaded (strict=True)")
        except RuntimeError as e:
            print(f"[RESUME] strict load failed: {e}")
            model_sd = model.state_dict()
            remapped_sd, mapped = _remap_plain_to_lora_keys(ckpt_model, model_sd)
            missing, unexpected = model.load_state_dict(remapped_sd, strict=False)
            print(f"[RESUME] Fallback load(strict=False): mapped={mapped} missing={len(missing)} unexpected={len(unexpected)}")

        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[RESUME] optimizer state skipped: {e}")

        if isinstance(ckpt, dict):
            scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val = ckpt.get("best_val", float("inf"))
            history = ckpt.get("history", [])

        # reset LR to args.lr (important)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
            pg.pop("initial_lr", None)

        remaining = args.epochs - (start_epoch - 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, remaining),
            eta_min=args.lr * 0.05
        )

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Train loop
    print("\n" + "=" * 80)
    print("Training Start")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n[EPOCH {epoch}/{args.epochs}] LR={optimizer.param_groups[0]['lr']:.6f}")

        # optional unfreeze
        if args.unfreeze_epoch and epoch == args.unfreeze_epoch:
            model.unfreeze_backbone()
            new_lr = args.lr * 0.1
            optimizer = torch.optim.AdamW(model.parameters(), lr=new_lr, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1
            )
            print(f"  → Backbone unfrozen, LR={new_lr:.6f}")

        lora_log_state = {"done": False}

        tr = train_epoch(
            model, train_loader, optimizer, scaler, device,
            amp_enabled=(args.amp and device == "cuda"),
            lambda_ndsm=args.lambda_ndsm,
            lambda_dsm=args.lambda_dsm,
            grad_w_dem=args.grad_w_dem,
            grad_w_ndsm=args.grad_w_ndsm,
            huber_w_dem=args.huber_w_dem,
            huber_w_ndsm=args.huber_w_ndsm,
            huber_w_dsm=args.huber_w_dsm,
            huber_delta_dem=args.huber_delta_dem,
            huber_delta_ndsm=args.huber_delta_ndsm,
            huber_delta_dsm=args.huber_delta_dsm,
            lora_log_state=lora_log_state,
        )

        va = validate(
            model, val_loader, device,
            amp_enabled=(args.amp and device == "cuda"),
            lambda_ndsm=args.lambda_ndsm,
            lambda_dsm=args.lambda_dsm,
            grad_w_dem=args.grad_w_dem,
            grad_w_ndsm=args.grad_w_ndsm,
            huber_w_dem=args.huber_w_dem,
            huber_w_ndsm=args.huber_w_ndsm,
            huber_w_dsm=args.huber_w_dsm,
            huber_delta_dem=args.huber_delta_dem,
            huber_delta_ndsm=args.huber_delta_ndsm,
            huber_delta_dsm=args.huber_delta_dsm,
        )

        scheduler.step()

        print(f"[EPOCH {epoch}]")
        print(f"  Train: loss={tr['loss']:.4f} | dem_h={tr['dem_huber']:.4f} dem_g={tr['dem_grad']:.4f} "
              f"| nd_h={tr['ndsm_huber']:.4f} nd_g={tr['ndsm_grad']:.4f} | dsm_h={tr['dsm_huber']:.4f}")
        print(f"  Val:   loss={va['loss']:.4f} | dem_h={va['dem_huber']:.4f} dem_g={va['dem_grad']:.4f} "
              f"| nd_h={va['ndsm_huber']:.4f} nd_g={va['ndsm_grad']:.4f} | dsm_h={va['dsm_huber']:.4f}")

        if args.use_lora:
            backbone_mod = model.net.backbone if hasattr(model.net, "backbone") else (model.net.net if hasattr(model.net, "net") else model.net)
            count, a_norm, b_norm = lora_stats(backbone_mod)
            print(f"  LoRA: modules={count} | A_norm={a_norm:.4f} B_norm={b_norm:.4f}")

        if fixed_indices:
            visualize_fixed_samples(model, val_ds, fixed_indices, device, epoch, vis_dir)

        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": tr, "val": va})

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val,
            "history": history,
            "config": vars(args),
        }

        torch.save(ckpt, out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ✓ Best model saved! (val_loss={best_val:.4f})")

        if epoch % 10 == 0:
            torch.save(ckpt, out_dir / f"epoch{epoch:03d}.pt")

    print("\n" + "=" * 80)
    print(f"Training Complete! Best val loss: {best_val:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


"""
권장 실행 예시 (LoRA + feature + stats init + blob-safe head)

python train_da3_dem_ndsm_lora_feats.py \
  --train_roots /home/wogmlcho86/Depth-Anything-3/korea_dataset_npz_balanced \
  --out_dir ./runs/da3_feats_lora \
  --model_name depth-anything/DA3-BASE \
  --target_size 518 \
  --batch_size 4 \
  --epochs 70 \
  --lr 1e-5 \
  --amp \
  --init_from_stats \
  --use_lora \
  --lora_dropout 0.0 \
  --feat_layers 10,15,20 \
  --lambda_ndsm 0.1 \
  --lambda_dsm 0.05 \
  --huber_w_dem 5.0 \
  --grad_w_dem 0.02

LoRA 타겟이 안 걸리면 --lora_regex 를 조정하세요.
예:  qkv/proj 이름이 'attn.to_qkv' 같이 다르면
  --lora_regex "(?:^|\\.)attn\\.(?:qkv|proj|to_qkv|to_out|out_proj)$"
"""
