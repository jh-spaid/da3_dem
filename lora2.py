#!/usr/bin/env python3
# train_da3_dem_ndsm_stats.py - Depth Anything V3 for DEM + nDSM regression (with stats-based init)
"""
Fine-tune Depth Anything V3 for aerial RGB → DEM + nDSM (DSM-DEM) multitask regression

Inputs:
  - RGB (aerial)

Targets (from NPZ):
  - dem: DEM elevation (meters)
  - dsm: DSM elevation (meters)
  - mask: valid mask (bool) [optional; if missing => all valid]

Derived target:
  - ndsm = clip(dsm - dem, 0, ndsm_clamp_max)

Outputs:
  - pred_dem  (meters)
  - pred_ndsm (meters, >=0 via softplus)
  - optional pred_dsm = pred_dem + pred_ndsm (for analysis)

Key points:
1) Use DepthAnything3.from_pretrained() API and train da3_wrapper.model
2) DA3 outputs relative depth → we learn metric mapping via adapter + (scale,shift)
3) For DEM: affine (scale*rel + shift)
4) For nDSM: nonnegative via softplus + scale
5) Stats-based init for DEM shift/scale and nDSM scale (p95) with --init_from_stats
6) Input size must be multiple of 14 (recommended: 518)
"""

import os
import glob
import json
import random
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


# ==================== UNet adapter ====================
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
    def __init__(self, in_ch=1, base=32, out_ch=32):
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


# ==================== LoRA ====================
class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear: W x + (alpha/r) * B(A(x))
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

        # LoRA params
        self.lora_A = nn.Linear(in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, out_features, bias=False)

        # Init: A ~ N(0, 0.01), B = 0
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


def _match_lora_target(name: str, target: str) -> bool:
    if target == "all":
        return True
    tokens = [t.strip().lower() for t in target.split(",") if t.strip()]
    name_l = name.lower()
    return any(tok in name_l for tok in tokens)


def apply_lora_to_linear(module: nn.Module, target="qkv,proj", r=8, alpha=16, dropout=0.0):
    """
    Recursively wrap matching nn.Linear modules with LoRA.
    Returns number of modules replaced.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and _match_lora_target(name, target):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += apply_lora_to_linear(child, target=target, r=r, alpha=alpha, dropout=dropout)
    return replaced


def mark_only_lora_trainable(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
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
            a_norm += float(m.lora_A.weight.norm().detach().cpu())
            b_norm += float(m.lora_B.weight.norm().detach().cpu())
    return count, a_norm, b_norm


def list_linear_module_names(module: nn.Module, max_items=120):
    names = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            names.append(name)
            if len(names) >= max_items:
                break
    return names


def _remap_plain_to_lora_keys(ckpt_sd, model_sd):
    """
    Map plain attention linear keys to LoRA-wrapped base keys.
    Example:
      ...attn.qkv.weight -> ...attn.qkv.base.weight
      ...attn.proj.bias  -> ...attn.proj.base.bias
    """
    remapped = dict(ckpt_sd)
    mapped = 0
    to_delete = []
    for k, v in ckpt_sd.items():
        nk = k
        if ".attn.qkv." in nk:
            nk = nk.replace(".attn.qkv.", ".attn.qkv.base.")
        elif ".attn.proj." in nk:
            nk = nk.replace(".attn.proj.", ".attn.proj.base.")
        if nk != k and nk in model_sd and getattr(model_sd[nk], "shape", None) == getattr(v, "shape", None):
            remapped[nk] = v
            mapped += 1
            to_delete.append(k)
    for k in to_delete:
        remapped.pop(k, None)
    return remapped, mapped


# ==================== Stats init ====================
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

    Returns:
      dem_mean, dem_std, ndsm_mean, ndsm_std, ndsm_p95
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

    def _valid_mask(arr, mask):
        finite = np.isfinite(arr)
        not_9999 = (arr != -9999.0)
        if mask is None:
            return finite & not_9999
        return mask.astype(bool) & finite & not_9999

    total = 0
    for f in files:
        try:
            d = np.load(f, allow_pickle=False)
            if ("dem" not in d.files) or ("dsm" not in d.files):
                continue

            dem = d["dem"].astype(np.float32, copy=False)
            dsm = d["dsm"].astype(np.float32, copy=False)
            mask = d["mask"] if "mask" in d.files else None

            vm = _valid_mask(dem, mask) & _valid_mask(dsm, mask)
            if not np.any(vm):
                continue

            dem_v = np.clip(dem[vm], dem_clamp_min, dem_clamp_max)
            ndsm_v = np.clip((dsm[vm] - dem[vm]), 0.0, ndsm_clamp_max)

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


# ==================== Model ====================
class DA3DemNdsmModel(nn.Module):
    """
    Depth Anything V3 adapted for DEM + nDSM regression

    - DA3 backbone => relative depth
    - shared adapter => features
    - DEM head => affine scale/shift
    - nDSM head => softplus + scale (>=0)
    """
    def __init__(
        self,
        model_name="depth-anything/DA3-LARGE-1.1",
        train_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target="qkv,proj",
        max_dem=2000.0,
        max_ndsm=300.0,
        init_dem_scale=None,
        init_dem_shift=None,
        init_ndsm_scale=None,
    ):
        super().__init__()

        print(f"[DA3DemNdsm] Loading model: {model_name}")
        da3_wrapper = DepthAnything3.from_pretrained(model_name)
        self.net = da3_wrapper.model
        print(f"[DA3DemNdsm] Model type: {type(self.net).__name__}")

        self.use_lora = bool(use_lora)
        # choose LoRA target: backbone if available, else whole model
        lora_target_mod = self.net.backbone if hasattr(self.net, "backbone") else self.net
        if self.use_lora:
            n = apply_lora_to_linear(
                lora_target_mod,
                target=lora_target,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            print(f"[LoRA] Applied to backbone: {n} Linear layers (target={lora_target})")

        # Freeze backbone if requested
        self.train_backbone = train_backbone
        if not train_backbone:
            print("[DA3DemNdsm] Freezing backbone (training heads only)")
            if hasattr(self.net, "backbone"):
                if self.use_lora:
                    mark_only_lora_trainable(self.net.backbone)
                else:
                    for p in self.net.backbone.parameters():
                        p.requires_grad = False
            elif hasattr(self.net, "net"):
                if self.use_lora:
                    mark_only_lora_trainable(self.net.net)
                else:
                    for p in self.net.net.parameters():
                        p.requires_grad = False
            if hasattr(self.net, "head"):
                for p in self.net.head.parameters():
                    p.requires_grad = True
        else:
            print("[DA3DemNdsm] Training full network")

        # Shared adapter (UNet-style)
        self.shared_adapter = UNetAdapter(in_ch=1, base=32, out_ch=32)

        self.dem_head = nn.Conv2d(32, 1, 1)
        self.ndsm_head = nn.Conv2d(32, 1, 1)

        # DEM affine init (stats-based like your DSM code)
        dem_scale_init = 100.0 if init_dem_scale is None else float(init_dem_scale)
        dem_shift_init = 0.0   if init_dem_shift is None else float(init_dem_shift)
        self.dem_scale = nn.Parameter(torch.tensor(dem_scale_init))
        self.dem_shift = nn.Parameter(torch.tensor(dem_shift_init))

        # nDSM scale init (use p95 recommended)
        ndsm_scale_init = 30.0 if init_ndsm_scale is None else float(init_ndsm_scale)
        self.ndsm_scale = nn.Parameter(torch.tensor(ndsm_scale_init))

        self.max_dem = float(max_dem)
        self.max_ndsm = float(max_ndsm)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DA3DemNdsm] Total params: {total/1e6:.2f}M")
        print(f"[DA3DemNdsm] Trainable params: {trainable/1e6:.2f}M")
        print(f"[DA3DemNdsm] Init DEM scale={self.dem_scale.item():.3f} shift={self.dem_shift.item():.3f}")
        print(f"[DA3DemNdsm] Init nDSM scale={self.ndsm_scale.item():.3f}")

    def forward(self, x, return_debug=False):
        """
        Args:
            x: (B,3,H,W) ImageNet-normalized RGB
        Returns:
            dem:  (B,1,H,W) meters
            ndsm: (B,1,H,W) meters >=0
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)  # (B,1,3,H,W)

        out = self.net(
            x,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],
            infer_gs=False,
        )

        depth = out["depth"]
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        feat = self.shared_adapter(depth)

        dem_rel = self.dem_head(feat)
        dem = self.dem_scale * dem_rel + self.dem_shift
        dem = torch.clamp(dem, 0.0, self.max_dem)

        ndsm_rel = self.ndsm_head(feat)
        ndsm = self.ndsm_scale * F.softplus(ndsm_rel)   # >=0
        ndsm = torch.clamp(ndsm, 0.0, self.max_ndsm)

        if return_debug:
            return dem, ndsm, depth, feat
        return dem, ndsm

    def unfreeze_backbone(self):
        print("[DA3DemNdsm] Unfreezing backbone (full fine-tuning)")
        for p in self.parameters():
            p.requires_grad = True
        self.train_backbone = True


# ==================== Dataset ====================
class NPZDemNdsmDataset(Dataset):
    """
    NPZ dataset:
      - image or rgb
      - dem
      - dsm
      - mask (optional)

    Creates ndsm = clip(dsm-dem, 0, ndsm_clamp_max)
    Resizes to target_size x target_size (multiple of 14).
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

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=False)

        img = d.get("image", d.get("rgb"))
        if img is None:
            raise KeyError(f"Missing 'image'/'rgb' in {self.files[idx]}")

        dem = d["dem"].astype(np.float32)
        dsm = d["dsm"].astype(np.float32)

        mask = d.get("mask", np.ones_like(dem, dtype=bool)).astype(bool)

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

        # Augmentation
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


# ==================== Losses ====================
def masked_huber(pred, gt, mask, delta=2.0):
    mask = mask > 0
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    diff = pred - gt
    absd = diff.abs()
    hub = torch.where(absd < delta, 0.5 * (diff ** 2) / delta, absd - 0.5 * delta)
    return hub[mask].mean()


def masked_grad_l1(pred, gt, mask):
    """Gradient L1 loss (AMP-safe)"""
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


# ==================== Visualization ====================
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
        # show reconstructed DSM for sanity
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


# ==================== Train/Val ====================
def _imagenet_norm(x, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def train_epoch(
    model, loader, optimizer, scaler, device,
    amp_enabled,
    lambda_ndsm,
    grad_w_dem, grad_w_ndsm,
    huber_w_dem, huber_w_ndsm,
    huber_delta_dem, huber_delta_ndsm,
    lora_log_state=None,
):
    model.train()

    totals = {
        "loss": 0.0,
        "dem_huber": 0.0, "dem_grad": 0.0,
        "ndsm_huber": 0.0, "ndsm_grad": 0.0,
    }

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        dem = batch["dem"].to(device, non_blocking=True)
        ndsm = batch["ndsm"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)

        x = _imagenet_norm(x, device)

        optimizer.zero_grad(set_to_none=True)

        # forward only AMP
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            p_dem, p_ndsm = model(x)

        # loss fp32
        p_dem32 = p_dem.float()
        p_ndsm32 = p_ndsm.float()
        dem32 = dem.float()
        ndsm32 = ndsm.float()
        m32 = m.float()

        dem_h = masked_huber(p_dem32, dem32, m32, delta=huber_delta_dem)
        dem_g = masked_grad_l1(p_dem32, dem32, m32) if grad_w_dem > 0 else p_dem32.new_tensor(0.0)
        nd_h = masked_huber(p_ndsm32, ndsm32, m32, delta=huber_delta_ndsm)
        nd_g = masked_grad_l1(p_ndsm32, ndsm32, m32) if grad_w_ndsm > 0 else p_ndsm32.new_tensor(0.0)

        loss_dem = huber_w_dem * dem_h + grad_w_dem * dem_g
        loss_nd = huber_w_ndsm * nd_h + grad_w_ndsm * nd_g
        loss = loss_dem + lambda_ndsm * loss_nd

        if not torch.isfinite(loss):
            print("[BAD] loss NaN/Inf:", batch["path"][0])
            print("dem_h, dem_g, nd_h, nd_g:", float(dem_h), float(dem_g), float(nd_h), float(nd_g))
            break

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # One-time LoRA update sanity check after first optimizer step.
        if lora_log_state is not None and not lora_log_state.get("done", False):
            if getattr(model, "use_lora", False) and hasattr(model, "net"):
                lora_target_mod = model.net.backbone if hasattr(model.net, "backbone") else model.net
                count, a_norm, b_norm = lora_stats(lora_target_mod)
                print(f"[LoRA] After 1 step: modules={count} | A_norm={a_norm:.4f} B_norm={b_norm:.4f}")
            lora_log_state["done"] = True

        totals["loss"] += loss.item()
        totals["dem_huber"] += dem_h.item()
        totals["dem_grad"] += dem_g.item()
        totals["ndsm_huber"] += nd_h.item()
        totals["ndsm_grad"] += nd_g.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dem_h": f"{dem_h.item():.3f}",
            "nd_h": f"{nd_h.item():.3f}",
            "λ": f"{lambda_ndsm:.2f}",
        })

    n = max(1, len(loader))
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(
    model, loader, device,
    amp_enabled,
    lambda_ndsm,
    grad_w_dem, grad_w_ndsm,
    huber_w_dem, huber_w_ndsm,
    huber_delta_dem, huber_delta_ndsm,
):
    model.eval()

    totals = {
        "loss": 0.0,
        "dem_huber": 0.0, "dem_grad": 0.0,
        "ndsm_huber": 0.0, "ndsm_grad": 0.0,
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

        loss_dem = huber_w_dem * dem_h + grad_w_dem * dem_g
        loss_nd = huber_w_ndsm * nd_h + grad_w_ndsm * nd_g
        loss = loss_dem + lambda_ndsm * loss_nd

        totals["loss"] += loss.item()
        totals["dem_huber"] += dem_h.item()
        totals["dem_grad"] += dem_g.item()
        totals["ndsm_huber"] += nd_h.item()
        totals["ndsm_grad"] += nd_g.item()

    n = max(1, len(loader))
    return {k: v / n for k, v in totals.items()}


# ==================== Main ====================
def main():
    import argparse

    ap = argparse.ArgumentParser("DA3 RGB → DEM + nDSM training (stats init)")

    # Data
    ap.add_argument("--train_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # Model
    ap.add_argument("--model_name", type=str, default="depth-anything/DA3-BASE")
    ap.add_argument("--train_backbone", action="store_true")
    ap.add_argument("--unfreeze_epoch", type=int, default=None)
    ap.add_argument("--use_lora", action="store_true", help="apply LoRA to DA3 backbone")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_target", type=str, default="qkv,proj", help="comma tokens or 'all'")

    # Training
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5)
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
    ap.add_argument("--lambda_ndsm", type=float, default=0.2, help="weight for nDSM task")
    ap.add_argument("--grad_w_dem", type=float, default=0.1, help="grad weight for DEM (small)")
    ap.add_argument("--grad_w_ndsm", type=float, default=0.5, help="grad weight for nDSM (bigger ok)")
    ap.add_argument("--huber_w_dem", type=float, default=1.0, help="huber weight for DEM")
    ap.add_argument("--huber_w_ndsm", type=float, default=1.0, help="huber weight for nDSM")
    ap.add_argument("--huber_delta_dem", type=float, default=2.0)
    ap.add_argument("--huber_delta_ndsm", type=float, default=1.0)

    # Viz
    ap.add_argument("--viz_samples", type=int, default=5)

    # Stats init (like your DSM script)
    ap.add_argument("--init_from_stats", action="store_true")
    ap.add_argument("--stats_max_files", type=int, default=200)
    ap.add_argument("--stats_max_vals", type=int, default=2_000_000)

    args = ap.parse_args()

    assert args.target_size % 14 == 0, f"target_size must be 14×N, got {args.target_size}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations_fixed"
    vis_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("Depth Anything V3 → DEM + nDSM Training (stats init)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Input size: {args.target_size}×{args.target_size}")
    print(f"Loss: L = L_dem + {args.lambda_ndsm} * L_ndsm")
    print(f"DEM huber_w={args.huber_w_dem}, grad_w={args.grad_w_dem}")
    print(f"nDSM huber_w={args.huber_w_ndsm}, grad_w={args.grad_w_ndsm}")
    print(f"LoRA: use={args.use_lora} r={args.lora_r} alpha={args.lora_alpha} "
          f"dropout={args.lora_dropout} target={args.lora_target}")

    # Init from stats
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
        init_ndsm_scale = max(ndsm_p95, 1.0)  # robust
        print(f"[INIT] Using stats init:")
        print(f"       DEM  shift={init_dem_shift:.3f} scale={init_dem_scale:.3f}")
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
    print("\n[MODEL] Building DA3 DEM+nDSM model...")
    model = DA3DemNdsmModel(
        model_name=args.model_name,
        train_backbone=args.train_backbone,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
        max_dem=args.dem_clamp_max,
        max_ndsm=args.ndsm_clamp_max,
        init_dem_scale=init_dem_scale,
        init_dem_shift=init_dem_shift,
        init_ndsm_scale=init_ndsm_scale,
    ).to(device)

    if args.use_lora and hasattr(model, "net"):
        lora_target_mod = model.net.backbone if hasattr(model.net, "backbone") else model.net
        count, a_norm, b_norm = lora_stats(lora_target_mod)
        print(f"[LoRA] Init: modules={count} | A_norm={a_norm:.4f} B_norm={b_norm:.4f}")
        if count == 0:
            names = list_linear_module_names(lora_target_mod, max_items=80)
            print("[LoRA] No modules matched target. Sample Linear names:")
            for n in names:
                print(f"  - {n}")
            raise RuntimeError(
                "use_lora=True but no modules were wrapped. "
                "Check --lora_target (e.g., qkv,proj) and backbone module names."
            )

    # Optim
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))

    # Resume
    start_epoch = 1
    best_val = float("inf")
    history = []
    """
    if args.resume and Path(args.resume).exists():
        print(f"\n[RESUME] Loading: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        history = ckpt.get("history", [])"""
    if args.resume and Path(args.resume).exists():
        print(f"\n[RESUME] Loading: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        try:
            model.load_state_dict(ckpt_model)
            print("[RESUME] Model state loaded with strict=True")
        except RuntimeError as e:
            print(f"[RESUME] strict load failed: {e}")
            model_sd = model.state_dict()
            remapped_sd, mapped = _remap_plain_to_lora_keys(ckpt_model, model_sd)
            missing, unexpected = model.load_state_dict(remapped_sd, strict=False)
            print(
                "[RESUME] Fallback load(strict=False): "
                f"mapped={mapped} missing={len(missing)} unexpected={len(unexpected)}"
            )

        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[RESUME] optimizer state skipped: {e}")
        if isinstance(ckpt, dict):
            scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))

        if isinstance(ckpt, dict):
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val = ckpt.get("best_val", float("inf"))
            history = ckpt.get("history", [])

        # ✅ (중요) optimizer lr을 커맨드라인 lr로 복구
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
            pg.pop("initial_lr", None)  # 있으면 지워서 혼동 방지(옵션)

        # ✅ scheduler 리셋: 남은 epoch 기준
        remaining = args.epochs - (start_epoch - 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, remaining),
            eta_min=args.lr * 0.05
        )

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Train loop
    print("\n" + "=" * 70)
    print("Training Start")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n[EPOCH {epoch}/{args.epochs}] LR={optimizer.param_groups[0]['lr']:.6f}")

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
            grad_w_dem=args.grad_w_dem,
            grad_w_ndsm=args.grad_w_ndsm,
            huber_w_dem=args.huber_w_dem,
            huber_w_ndsm=args.huber_w_ndsm,
            huber_delta_dem=args.huber_delta_dem,
            huber_delta_ndsm=args.huber_delta_ndsm,
            lora_log_state=lora_log_state,
        )

        va = validate(
            model, val_loader, device,
            amp_enabled=(args.amp and device == "cuda"),
            lambda_ndsm=args.lambda_ndsm,
            grad_w_dem=args.grad_w_dem,
            grad_w_ndsm=args.grad_w_ndsm,
            huber_w_dem=args.huber_w_dem,
            huber_w_ndsm=args.huber_w_ndsm,
            huber_delta_dem=args.huber_delta_dem,
            huber_delta_ndsm=args.huber_delta_ndsm,
        )

        scheduler.step()

        print(f"[EPOCH {epoch}]")
        print(f"  Train: loss={tr['loss']:.4f} | dem_h={tr['dem_huber']:.4f} dem_g={tr['dem_grad']:.4f} "
              f"| nd_h={tr['ndsm_huber']:.4f} nd_g={tr['ndsm_grad']:.4f}")
        print(f"  Val:   loss={va['loss']:.4f} | dem_h={va['dem_huber']:.4f} dem_g={va['dem_grad']:.4f} "
              f"| nd_h={va['ndsm_huber']:.4f} nd_g={va['ndsm_grad']:.4f}")
        if args.use_lora and hasattr(model, "net"):
            lora_target_mod = model.net.backbone if hasattr(model.net, "backbone") else model.net
            count, a_norm, b_norm = lora_stats(lora_target_mod)
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

    print("\n" + "=" * 70)
    print(f"Training Complete! Best val loss: {best_val:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

"""
python train_dem.py \
  --train_roots /home/wogmlcho86/Depth-Anything-3/korea_dataset \
  --out_dir ./runs/da3_dem_ndsm \
  --model_name depth-anything/DA3-BASE \
  --target_size 518 \
  --batch_size 4 \
  --epochs 30 \
  --lr 5e-5 \
  --lambda_ndsm 0.2 \
  --grad_w_dem 0.1 \
  --grad_w_ndsm 0.5 \
  --init_from_stats \
  --amp


"""

"""

python train_dem.py \
  --train_roots /home/wogmlcho86/Depth-Anything-3/korea_dataset \
  --out_dir ./runs/da3_dem_ndsm \
  --model_name depth-anything/DA3-BASE \
  --target_size 518 \
  --batch_size 4 \
  --epochs 30 \
  --lr 2e-5 \
  --lambda_ndsm 0.05 \
  --huber_w_dem 10.0 \
  --grad_w_dem 0.0 \
  --huber_w_ndsm 1.0 \
  --grad_w_ndsm 0.5 \
  --huber_delta_dem 10.0 \
  --huber_delta_ndsm 1.0 \
  --init_from_stats \
  --amp


"""
