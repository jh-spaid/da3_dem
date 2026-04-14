#!/usr/bin/env python3
"""
Depth Anything V3 - DEM + nDSM regression
Base: train_dem.py (v1)  +  중간 피처(eager init)  +  LoRA

변경점 요약 vs 1번 코드:
1. DA3 중간 피처(export_feat_layers) → feat_reduce(1x1 Conv, eager init) → UNetAdapter
2. LoRA: backbone의 attn.qkv Linear에만 선택적으로 적용
3. depth residual 없음 (1번과 동일한 DEM head 구조 유지)
4. feat_reduce를 __init__에서 eager init (optimizer에 정상 등록)
5. 나머지 loss / dataset / schedule 등 1번과 동일
"""

import os
import glob
import json
import random
import re
import contextlib
from pathlib import Path
import wandb

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from depth_anything_3.api import DepthAnything3


# ==================== UNet adapter ====================
"""
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
        )"""
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReplicationPad2d(1),          # zero → replication
            nn.Conv2d(in_ch, out_ch, 3, padding=0),   # padding=1 → 0
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),          # zero → replication
            nn.Conv2d(out_ch, out_ch, 3, padding=0),
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
    """
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b  = self.bottleneck(F.max_pool2d(e2, 2))
        u2 = F.interpolate(b,  size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1) """
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b  = self.bottleneck(F.max_pool2d(e2, 2))
        u2 = F.interpolate(b,  size=e2.shape[-2:], mode="bilinear", align_corners=True)  # False→True
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=True)  # False→True
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1)



# ==================== LoRA ====================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base     = base
        self.r        = r
        self.scaling  = alpha / max(1, r)
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A   = nn.Linear(base.in_features,  r, bias=False)
        self.lora_B   = nn.Linear(r, base.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def apply_lora(root: nn.Module, pattern: str, r=8, alpha=16, dropout=0.0):
    """fullname regex로 매칭되는 nn.Linear에 LoRA 적용. 적용 수 반환."""
    rx = re.compile(pattern)
    targets = [n for n, m in root.named_modules()
               if isinstance(m, nn.Linear) and rx.search(n)]

    for name in targets:
        parts  = name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        mod  = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))

    if targets:
        print(f"[LoRA] Applied to {len(targets)} layers | regex='{pattern}'")
    else:
        print("[LoRA] WARNING: 0 layers matched. Printing Linear names for debugging:")
        for n, m in root.named_modules():
            if isinstance(m, nn.Linear):
                print(f"  {n}")
    return len(targets)


# ==================== Feature helpers ====================
def _feat_to_bchw(feat: torch.Tensor) -> torch.Tensor:
    """DA3 aux feature를 (B,C,H,W)로 변환."""
    if feat.ndim == 5:                           # (B,N,h,w,C)
        B, N, h, w, C = feat.shape
        feat = feat.mean(1) if N > 1 else feat[:, 0]
        return feat.permute(0, 3, 1, 2).contiguous()
    if feat.ndim == 4:
        # BCHW vs BHWC 판단: C가 작으면 BCHW일 가능성이 높음
        if feat.shape[1] <= feat.shape[3]:
            return feat.permute(0, 3, 1, 2).contiguous()  # BHWC
        return feat                                         # BCHW
    raise ValueError(f"Unsupported feat shape: {tuple(feat.shape)}")


def probe_feat_channels(model_net, feat_layers, target_size, device):
    """
    Dummy forward로 각 feat_layer의 채널 수를 합산해 feat_reduce in_ch를 구함.
    (eager init을 위해 __init__ 에서 한 번만 호출)
    모델을 device로 먼저 옮긴 뒤 dummy를 같은 device에서 실행 → RoPE 버퍼 device 불일치 방지
    """
    model_net = model_net.to(device)
    dummy = torch.zeros(1, 1, 3, target_size, target_size, device=device)
    with torch.no_grad():
        out = model_net(
            dummy,
            extrinsics=None, intrinsics=None,
            export_feat_layers=feat_layers,
            infer_gs=False,
        )
    aux = out.get("aux", {}) if isinstance(out, dict) else {}
    total_ch = 0
    found = []
    for k in feat_layers:
        key = f"feat_layer_{k}"
        if key in aux:
            f = _feat_to_bchw(aux[key])
            total_ch += f.shape[1]
            found.append((key, f.shape[1]))
        else:
            print(f"[probe] WARNING: {key} not found in aux. Available: {list(aux.keys())}")
    print(f"[probe] feat_layers={feat_layers} → {found} → total_ch={total_ch}")
    return total_ch


# ==================== Stats init ====================
def estimate_dem_ndsm_stats(
    train_roots, split="train",
    dem_clamp_min=-200., dem_clamp_max=2000., ndsm_clamp_max=300.,
    max_files=200, max_vals=2_000_000, seed=42,
):
    rng   = np.random.default_rng(seed)
    files = []
    for r in train_roots:
        files += sorted(glob.glob(str(Path(r) / split / "*.npz")))
    if not files:
        raise RuntimeError(f"No NPZ files under {train_roots}/*/{split}/*.npz")
    if len(files) > max_files:
        files = list(rng.choice(files, max_files, replace=False))

    dem_vals, ndsm_vals, total = [], [], 0
    for f in files:
        try:
            d    = np.load(f, allow_pickle=False)
            if "dem" not in d.files or "dsm" not in d.files:
                continue
            dem  = d["dem"].astype(np.float32)
            dsm  = d["dsm"].astype(np.float32)
            mask = d["mask"].astype(bool) if "mask" in d.files else None
            vm   = np.isfinite(dem) & np.isfinite(dsm) & (dem != -9999.) & (dsm != -9999.)
            if mask is not None: vm &= mask
            if not np.any(vm): continue
            dv   = np.clip(dem[vm], dem_clamp_min, dem_clamp_max)
            nv   = np.clip(dsm[vm] - dem[vm], 0., ndsm_clamp_max)
            dem_vals.append(dv); ndsm_vals.append(nv); total += dv.size
            if total >= max_vals: break
        except Exception:
            continue
    if not dem_vals:
        raise RuntimeError("No valid DEM/DSM values found.")

    da, na = np.concatenate(dem_vals), np.concatenate(ndsm_vals)
    dm, ds = float(da.mean()), float(da.std())
    nm, ns = float(na.mean()), float(na.std())
    np95   = float(np.percentile(na, 95))
    print(f"[INIT] {da.size} pixels | DEM mean={dm:.2f} std={ds:.2f} | nDSM mean={nm:.2f} p95={np95:.2f}")
    return dm, ds, nm, ns, np95


# ==================== Model ====================
class DA3DemNdsmModel(nn.Module):
    """
    DA3 backbone → (optional) 중간 피처 concat → UNetAdapter → DEM/nDSM head

    use_feats=False 이면 1번 코드와 완전히 동일한 구조 (DA3 final depth만 사용).
    use_feats=True  이면 중간 피처를 feat_reduce로 줄인 뒤 UNetAdapter에 입력.
    """
    def __init__(
        self,
        model_name="depth-anything/DA3-BASE",
        train_backbone=False,
        # 중간 피처
        use_feats=True,
        feat_layers=(10, 15, 20),
        feat_reduce_ch=64,
        adapter_base=32,
        # LoRA
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_regex=r"(?:^|\.)attn\.qkv$",
        # 범위
        max_dem=2000., dem_min=-200., max_ndsm=300.,
        # 스칼라 init
        init_dem_scale=None, init_dem_shift=None, init_ndsm_scale=None,
        # probe용 (eager init)
        target_size=518,
    ):
        super().__init__()
        self.use_feats  = use_feats
        self.feat_layers = list(feat_layers)

        print(f"[Model] Loading: {model_name}")
        wrapper   = DepthAnything3.from_pretrained(model_name)
        self.net  = wrapper.model

        # backbone 참조 (LoRA / freeze 공통)
        self.backbone = (self.net.backbone if hasattr(self.net, "backbone")
                         else (self.net.net if hasattr(self.net, "net") else self.net))

        # LoRA 먼저 적용 (freeze 전에)
        if use_lora:
            n = apply_lora(self.backbone, lora_regex, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            if n == 0:
                raise RuntimeError("LoRA: 0 layers matched. Adjust --lora_regex.")

        # backbone freeze/unfreeze
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if use_lora:
                # LoRA 파라미터만 다시 켜기
                for m in self.backbone.modules():
                    if isinstance(m, LoRALinear):
                        m.lora_A.weight.requires_grad = True
                        m.lora_B.weight.requires_grad = True
            print("[Model] Backbone frozen" + (" (LoRA trainable)" if use_lora else ""))
        else:
            print("[Model] Full backbone training")

        # DA3 head는 항상 trainable
        if hasattr(self.net, "head"):
            for p in self.net.head.parameters():
                p.requires_grad = True

        # probe용 device: CUDA 있으면 CUDA에서 실행 (RoPE 버퍼 device 불일치 방지)
        probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_feats:
            total_ch = probe_feat_channels(self.net, self.feat_layers, target_size, probe_device)
            if total_ch == 0:
                raise RuntimeError("probe_feat_channels returned 0. Check feat_layers.")
            self.feat_reduce = nn.Sequential(
                nn.Conv2d(total_ch, feat_reduce_ch, 1, bias=True),
                nn.BatchNorm2d(feat_reduce_ch),
                nn.ReLU(inplace=True),
            )
            adapter_in = feat_reduce_ch
            print(f"[Model] feat_reduce: {total_ch} → {feat_reduce_ch}")
        else:
            self.feat_reduce = None
            adapter_in = 1          # DA3 final depth 1채널 (1번과 동일)

        self.shared_adapter = UNetAdapter(in_ch=adapter_in, base=adapter_base, out_ch=32)
        self.dem_head        = nn.Conv2d(32, 1, 1)
        self.ndsm_head       = nn.Conv2d(32, 1, 1)

        # 스칼라 파라미터 init
        self.dem_scale  = nn.Parameter(torch.tensor(float(init_dem_scale  or 100.)))
        self.dem_shift  = nn.Parameter(torch.tensor(float(init_dem_shift  or 0.)))
        self.ndsm_scale = nn.Parameter(torch.tensor(float(init_ndsm_scale or 30.)))

        self.dem_min  = float(dem_min)
        self.max_dem  = float(max_dem)
        self.max_ndsm = float(max_ndsm)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total={total/1e6:.2f}M  Trainable={trainable/1e6:.2f}M")
        print(f"[Model] dem_scale={self.dem_scale.item():.2f}  dem_shift={self.dem_shift.item():.2f}  ndsm_scale={self.ndsm_scale.item():.2f}")

    # ── forward ──────────────────────────────────────────────────
    def forward(self, x):
        """x: (B,3,H,W) ImageNet-normalized"""
        H, W = x.shape[-2], x.shape[-1]
        if x.ndim == 4:
            x = x.unsqueeze(1)   # (B,1,3,H,W)

        out = self.net(
            x,
            extrinsics=None, intrinsics=None,
            export_feat_layers=self.feat_layers if self.use_feats else [],
            infer_gs=False,
        )

        # ── 입력 특징 선택 ──
        if self.use_feats:
            aux = out.get("aux", {}) if isinstance(out, dict) else {}
            feats = []
            for k in self.feat_layers:
                key = f"feat_layer_{k}"
                if key in aux:
                    f = _feat_to_bchw(aux[key])
                    f = F.interpolate(f, size=(H // 14, W // 14),
                                      mode="bilinear", align_corners=False)
                    feats.append(f)
            if not feats:
                raise RuntimeError("No aux features extracted. Check feat_layers.")
            feat_cat = torch.cat(feats, dim=1)
            feat_cat = torch.nan_to_num(feat_cat)
            inp = self.feat_reduce(feat_cat)          # (B, feat_reduce_ch, h, w)
        else:
            # 1번과 동일: DA3 final depth
            depth = out["depth"]
            if depth.ndim == 3:
                depth = depth.unsqueeze(1)
            inp = depth                               # (B, 1, H, W)

        inp = F.interpolate(inp, size=(H, W), mode="bilinear", align_corners=False)

        feat = self.shared_adapter(inp)

        # DEM
        dem_rel = self.dem_head(feat)
        dem = self.dem_scale * dem_rel + self.dem_shift
        dem = torch.clamp(dem, self.dem_min, self.max_dem)

        # nDSM
        ndsm_rel = self.ndsm_head(feat)
        ndsm = self.ndsm_scale * F.softplus(ndsm_rel)
        ndsm = torch.clamp(ndsm, 0., self.max_ndsm)

        return dem, ndsm

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True
        print("[Model] Backbone unfrozen (full fine-tuning)")


# ==================== Dataset ====================
class NPZDemNdsmDataset(Dataset):
    def __init__(self, root_dir, split="train", augment=True, target_size=518,
                 dem_clamp_min=-200., dem_clamp_max=2000., ndsm_clamp_max=300.):
        self.root         = Path(root_dir)
        self.split        = split
        self.augment      = augment
        self.target_size  = int(target_size)
        self.dem_clamp_min = float(dem_clamp_min)
        self.dem_clamp_max = float(dem_clamp_max)
        self.ndsm_clamp_max = float(ndsm_clamp_max)
        self.files = sorted(glob.glob(str(self.root / split / "*.npz")))
        if not self.files:
            raise RuntimeError(f"No NPZ files under: {self.root / split}")
        print(f"  [{split}] {len(self.files)} samples (→ {self.target_size}×{self.target_size})")

    def __len__(self): return len(self.files)

    @staticmethod
    def _ensure_hwc3(img):
        if img.ndim == 2:           return np.stack([img]*3, -1)
        if img.shape[0] == 3:       return np.transpose(img, (1,2,0))
        return img

    @staticmethod
    def _to_chw01(img):
        img = img.astype(np.float32)
        if img.max() > 2.: img /= 255.
        return np.transpose(np.clip(img, 0, 1), (2,0,1))

    def __getitem__(self, idx):
        d    = np.load(self.files[idx], allow_pickle=False)
        img  = d.get("image", d.get("rgb"))
        if img is None: raise KeyError(f"No image/rgb in {self.files[idx]}")
        dem  = d["dem"].astype(np.float32)
        dsm  = d["dsm"].astype(np.float32)
        mask = d["mask"].astype(bool) if "mask" in d.files else np.ones_like(dem, dtype=bool)

        dem  = np.clip(dem, self.dem_clamp_min, self.dem_clamp_max)
        ndsm = np.clip(dsm - dem, 0., self.ndsm_clamp_max)

        x      = torch.from_numpy(self._to_chw01(self._ensure_hwc3(img)))
        y_dem  = torch.from_numpy(dem).unsqueeze(0)
        y_ndsm = torch.from_numpy(ndsm).unsqueeze(0)
        m      = torch.from_numpy(mask).unsqueeze(0)

        S = self.target_size
        if x.shape[1] != S or x.shape[2] != S:
            x      = F.interpolate(x.unsqueeze(0),      (S,S), mode="bilinear",  align_corners=False).squeeze(0)
            y_dem  = F.interpolate(y_dem.unsqueeze(0),  (S,S), mode="bilinear",  align_corners=False).squeeze(0)
            y_ndsm = F.interpolate(y_ndsm.unsqueeze(0), (S,S), mode="bilinear",  align_corners=False).squeeze(0)
            m      = F.interpolate(m.unsqueeze(0).float(),(S,S),mode="nearest").squeeze(0).bool()

        if self.augment and self.split == "train":
            if random.random() < .5:
                x=torch.flip(x,[2]); y_dem=torch.flip(y_dem,[2]); y_ndsm=torch.flip(y_ndsm,[2]); m=torch.flip(m,[2])
            if random.random() < .5:
                x=torch.flip(x,[1]); y_dem=torch.flip(y_dem,[1]); y_ndsm=torch.flip(y_ndsm,[1]); m=torch.flip(m,[1])
            if random.random() < .5:
                k=random.randint(1,3)
                x=torch.rot90(x,k,[1,2]); y_dem=torch.rot90(y_dem,k,[1,2])
                y_ndsm=torch.rot90(y_ndsm,k,[1,2]); m=torch.rot90(m,k,[1,2])

        return {"image":x, "dem":y_dem, "ndsm":y_ndsm, "mask":m, "path":str(self.files[idx])}


# ==================== Loss ====================
def masked_huber(pred, gt, mask, delta=2.):
    mask = mask > 0
    if mask.sum() == 0: return pred.new_tensor(0.)
    d = pred - gt
    a = d.abs()
    h = torch.where(a < delta, .5*d*d/delta, a - .5*delta)
    return h[mask].mean()

def masked_mae_rmse(pred, gt, mask):
    m = (mask > 0) & torch.isfinite(pred) & torch.isfinite(gt)
    if m.sum() == 0:
        z = pred.new_tensor(0.)
        return z, z
    e = (pred - gt).abs()[m]
    return e.mean(), torch.sqrt((e**2).mean())

def masked_grad_l1(pred, gt, mask):
    m = (mask > 0).float()
    pred = torch.nan_to_num(pred); gt = torch.nan_to_num(gt)
    dx = (pred[...,:,1:]-pred[...,:,:-1]) - (gt[...,:,1:]-gt[...,:,:-1])
    dy = (pred[...,1:,:]-pred[...,:-1,:]) - (gt[...,1:,:]-gt[...,:-1,:])
    mx = m[...,:,1:]*m[...,:,:-1]; my = m[...,1:,:]*m[...,:-1,:]
    num   = (dx.abs()*mx).sum() + (dy.abs()*my).sum()
    denom = (mx.sum() + my.sum()).clamp(1.)
    return num / denom


# ==================== Viz ====================
MEAN = torch.tensor([.485,.456,.406]).view(1,3,1,1)
STD  = torch.tensor([.229,.224,.225]).view(1,3,1,1)

@torch.no_grad()
def visualize_fixed_samples(model, dataset, indices, device, epoch, vis_dir):
    model.eval()
    for idx in indices:
        s    = dataset[idx]
        name = Path(s["path"]).stem
        x    = s["image"].unsqueeze(0).to(device)
        m    = s["mask"].unsqueeze(0).to(device)
        xn   = (x - MEAN.to(device)) / STD.to(device)
        pd, pn = model(xn)

        rgb   = np.transpose(np.clip(x.squeeze().cpu().numpy(), 0, 1), (1,2,0))
        dp  = pd.squeeze().cpu().numpy()
        dg  = s["dem"].squeeze().numpy()
        np_ = pn.squeeze().cpu().numpy()
        ng  = s["ndsm"].squeeze().numpy()
        valid = m.squeeze().cpu().numpy().astype(bool)
        if not valid.any(): continue

        # ── 공통 범위 계산 ──
        dv1, dv2 = np.percentile(np.concatenate([dp[valid], dg[valid]]), [2, 98])
        nv2      = float(np.percentile(np.concatenate([np_[valid], ng[valid]]), 98))

        de  = np.ma.masked_where(~valid, np.abs(dp  - dg))
        ne  = np.ma.masked_where(~valid, np.abs(np_ - ng))
        dev = np.percentile(de.compressed(), 95) if de.count() > 0 else 10
        nev = np.percentile(ne.compressed(), 95) if ne.count() > 0 else 10

        dsm_p = np.ma.masked_where(~valid, dp  + np_)
        dsm_g = np.ma.masked_where(~valid, dg  + ng)
        sv1, sv2 = np.percentile(
            np.concatenate([dsm_p.compressed(), dsm_g.compressed()]), [2, 98])

        # ── 2행 4열 레이아웃 ──
        # 행0: RGB | DEM Pred | DEM GT | |DEM Err|
        # 행1: DSM Pred | nDSM Pred | nDSM GT | |nDSM Err|
        fig = plt.figure(figsize=(20, 10))
        gs  = GridSpec(2, 4, figure=fig, hspace=.3, wspace=.25)

        def _show(ax, data, title, vmin, vmax, cmap):
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=.046, pad=.04)

        # 행 0
        ax = fig.add_subplot(gs[0, 0]); ax.imshow(rgb); ax.set_title("RGB"); ax.axis("off")
        _show(fig.add_subplot(gs[0, 1]), np.ma.masked_where(~valid, dp),
              "DEM Pred (m)",  dv1, dv2, "viridis")
        _show(fig.add_subplot(gs[0, 2]), np.ma.masked_where(~valid, dg),
              "DEM GT (m)",    dv1, dv2, "viridis")
        _show(fig.add_subplot(gs[0, 3]), de,
              f"|DEM Err|  p95={dev:.1f}m", 0, dev, "hot")

        # 행 1
        _show(fig.add_subplot(gs[1, 0]), dsm_p,
              "DSM Pred (=DEM+nDSM)", sv1, sv2, "viridis")
        _show(fig.add_subplot(gs[1, 1]), np.ma.masked_where(~valid, np_),
              "nDSM Pred (m)", 0, nv2, "magma")
        _show(fig.add_subplot(gs[1, 2]), np.ma.masked_where(~valid, ng),
              "nDSM GT (m)",   0, nv2, "magma")
        _show(fig.add_subplot(gs[1, 3]), ne,
              f"|nDSM Err|  p95={nev:.1f}m", 0, nev, "hot")

        fig.suptitle(f"Epoch {epoch} — {name}", fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig(vis_dir / f"epoch{epoch:03d}_{name}.png",
                    dpi=100, bbox_inches="tight", facecolor="white")
        plt.close()


# ==================== Train / Val ====================
def _norm(x, device):
    return (x - MEAN.to(device)) / STD.to(device)


def _run_epoch(model, loader, optimizer, scaler, device, cfg, train=True, is_master=True):
    model.train() if train else model.eval()
    totals = {k: 0. for k in [
        "loss","dem_huber","dem_grad","ndsm_huber","ndsm_grad",
        "dem_mae","dem_rmse","ndsm_mae","ndsm_rmse","skipped"]}
    valid  = 0

    ctx = contextlib.nullcontext if not train else contextlib.nullcontext
    pbar = tqdm(loader, desc="Train" if train else "Val", disable=(not is_master))

    for batch in pbar:
        x    = _norm(batch["image"].to(device), device)
        dem  = batch["dem"].to(device)
        ndsm = batch["ndsm"].to(device)
        m    = batch["mask"].to(device)

        amp_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                   if cfg["amp"] else contextlib.nullcontext())

        with amp_ctx:
            pd, pn = model(x)

        if not (torch.isfinite(pd).all() and torch.isfinite(pn).all()):
            totals["skipped"] += 1
            if train: optimizer.zero_grad(set_to_none=True)
            continue

        pd32, pn32 = pd.float(), pn.float()
        d32, n32, m32 = dem.float(), ndsm.float(), m.float()

        dh = masked_huber(pd32, d32, m32, cfg["huber_delta_dem"])
        dg = masked_grad_l1(pd32, d32, m32) if cfg["grad_w_dem"] > 0 else pd32.new_tensor(0.)
        nh = masked_huber(pn32, n32, m32, cfg["huber_delta_ndsm"])
        ng = masked_grad_l1(pn32, n32, m32) if cfg["grad_w_ndsm"] > 0 else pn32.new_tensor(0.)
        d_mae, d_rmse = masked_mae_rmse(pd32, d32, m32)
        n_mae, n_rmse = masked_mae_rmse(pn32, n32, m32)

        loss = (cfg["huber_w_dem"]*dh + cfg["grad_w_dem"]*dg
              + cfg["lambda_ndsm"]*(cfg["huber_w_ndsm"]*nh + cfg["grad_w_ndsm"]*ng))

        if not torch.isfinite(loss):
            totals["skipped"] += 1
            if train: optimizer.zero_grad(set_to_none=True)
            continue

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        totals["loss"]       += loss.item()
        totals["dem_huber"]  += dh.item()
        totals["dem_grad"]   += dg.item()
        totals["ndsm_huber"] += nh.item()
        totals["ndsm_grad"]  += ng.item()
        totals["dem_mae"]    += d_mae.item()
        totals["dem_rmse"]   += d_rmse.item()
        totals["ndsm_mae"]   += n_mae.item()
        totals["ndsm_rmse"]  += n_rmse.item()
        valid += 1

        if train:
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             dem_h=f"{dh.item():.3f}", nd_h=f"{nh.item():.3f}",
                             skip=int(totals["skipped"]))

    # DDP: aggregate metric sums and valid counts across all ranks.
    if dist.is_available() and dist.is_initialized():
        keys = [
            "loss","dem_huber","dem_grad","ndsm_huber","ndsm_grad",
            "dem_mae","dem_rmse","ndsm_mae","ndsm_rmse","skipped"
        ]
        t = torch.tensor([totals[k] for k in keys] + [float(valid)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for i, k in enumerate(keys):
            totals[k] = float(t[i].item())
        valid = int(t[-1].item())

    n = max(1, valid)
    return {k: (v/n if k!="skipped" else v) for k,v in totals.items()}


def _setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = bool(args.ddp or world_size > 1)
    if not distributed:
        return False, 0, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA in this script. Please launch on GPUs.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    ndev = torch.cuda.device_count()
    if local_rank >= ndev:
        raise RuntimeError(
            f"Invalid local_rank={local_rank} for visible CUDA device count={ndev}. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method="env://")
    return True, local_rank, rank, world_size


# ==================== Main ====================
def main():
    import argparse
    ap = argparse.ArgumentParser("DA3 DEM+nDSM v3: 1번 베이스 + 중간 피처 + LoRA")

    # Data
    ap.add_argument("--train_roots", nargs="+", required=True)
    ap.add_argument("--out_dir",     type=str,  required=True)

    # Model
    ap.add_argument("--model_name",     type=str,  default="depth-anything/DA3-BASE")
    ap.add_argument("--train_backbone", action="store_true")
    ap.add_argument("--unfreeze_epoch", type=int,  default=None)

    # 중간 피처
    ap.add_argument("--use_feats",      action="store_true",
                    help="DA3 중간 피처 사용 (기본: DA3 final depth만 사용, 1번과 동일)")
    ap.add_argument("--feat_layers",    type=str,  default="8,10,11")
    ap.add_argument("--feat_reduce_ch", type=int,  default=64)
    ap.add_argument("--adapter_base",   type=int,  default=32)

    # LoRA
    ap.add_argument("--use_lora",       action="store_true")
    ap.add_argument("--lora_r",         type=int,  default=8)
    ap.add_argument("--lora_alpha",     type=float,default=16.)
    ap.add_argument("--lora_dropout",   type=float,default=0.)
    ap.add_argument("--lora_regex",     type=str,  default=r"attn\.(qkv|proj)",
                    help="LoRA를 적용할 Linear layer의 fullname regex. "
                         "--list_linears 로 레이어 이름 먼저 확인 가능")

    # Training
    ap.add_argument("--batch_size",  type=int,   default=4)
    ap.add_argument("--epochs",      type=int,   default=50)
    ap.add_argument("--lr",          type=float, default=1e-5)
    ap.add_argument("--lr_lora",     type=float, default=None,
                    help="LoRA 파라미터 전용 LR (None이면 --lr 사용)")
    ap.add_argument("--wd",          type=float, default=0.01)
    ap.add_argument("--amp",         action="store_true")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--resume",      type=str,   default=None)
    ap.add_argument("--num_workers", type=int,   default=4)

    # Sizes / clamps
    ap.add_argument("--target_size",    type=int,   default=518)
    ap.add_argument("--dem_clamp_min",  type=float, default=-200.)
    ap.add_argument("--dem_clamp_max",  type=float, default=2000.)
    ap.add_argument("--ndsm_clamp_max", type=float, default=300.)

    # Loss  (1번과 동일한 기본값)
    ap.add_argument("--lambda_ndsm",      type=float, default=0.05)
    ap.add_argument("--grad_w_dem",       type=float, default=0.04)
    ap.add_argument("--grad_w_ndsm",      type=float, default=0.5)
    ap.add_argument("--huber_w_dem",      type=float, default=10.)
    ap.add_argument("--huber_w_ndsm",     type=float, default=1.)
    ap.add_argument("--huber_delta_dem",  type=float, default=10.)
    ap.add_argument("--huber_delta_ndsm", type=float, default=1.)

    # Viz / Stats
    ap.add_argument("--viz_samples",    type=int,  default=5)
    ap.add_argument("--init_from_stats",action="store_true")
    ap.add_argument("--stats_max_files",type=int,  default=200)
    ap.add_argument("--stats_max_vals", type=int,  default=2_000_000)

    ap.add_argument("--list_linears", action="store_true",
                    help="모델의 Linear 레이어 이름을 출력하고 종료 (LoRA regex 확인용)")
    ap.add_argument("--ddp", action="store_true",
                    help="Enable DDP (also auto-enabled when launched with torchrun WORLD_SIZE>1).")
    ap.add_argument("--dist_backend", type=str, default="nccl", choices=["nccl", "gloo"])

        # W&B
    ap.add_argument("--use_wandb", action="store_true",
                    help="Enable Weights & Biases logging (master rank only)")
    ap.add_argument("--wandb_project", type=str,
                    default=os.environ.get("WANDB_PROJECT", "da3-dem-ndsm"))
    ap.add_argument("--wandb_entity", type=str,
                    default=os.environ.get("WANDB_ENTITY", None))
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_tags", type=str, default="vertex-ai,dem,ndsm")
    
    args = ap.parse_args()
    assert args.target_size % 14 == 0, f"target_size must be multiple of 14, got {args.target_size}"

    # LoRA regex 디버깅 모드
    if args.list_linears:
        print("[list_linears] Loading model to inspect Linear layer names...")
        wrapper = DepthAnything3.from_pretrained(args.model_name)
        net = wrapper.model
        bb  = net.backbone if hasattr(net, "backbone") else (net.net if hasattr(net, "net") else net)
        print("\n=== nn.Linear layer names in backbone ===")
        for n, m in bb.named_modules():
            if isinstance(m, nn.Linear):
                print(f"  {n}  [{m.in_features} -> {m.out_features}]")
        print("==========================================")
        print("위 이름 중 LoRA를 적용할 패턴을 --lora_regex 에 지정하세요.")
        print("예시: --lora_regex 'attn\\.qkv'")
        return

    feat_layers = [int(x) for x in args.feat_layers.split(",") if x.strip()]

    distributed, local_rank, rank, world_size = _setup_distributed(args)
    is_master = (rank == 0)

    random.seed(args.seed + rank); np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank); torch.cuda.manual_seed_all(args.seed + rank)

    if distributed:
        device = f"cuda:{local_rank}"
    else:
        device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations_fixed"; vis_dir.mkdir(exist_ok=True)

    if is_master:
        print("\n" + "="*70)
        print("DA3 DEM+nDSM v3  |  use_feats=%s  use_lora=%s" % (args.use_feats, args.use_lora))
        if distributed:
            print(f"DDP enabled | world_size={world_size} rank={rank} local_rank={local_rank}")
        print("="*70)

    # Stats init
    init_dem_scale = init_dem_shift = init_ndsm_scale = None
    """if args.init_from_stats:
        dm, ds, _, _, np95 = estimate_dem_ndsm_stats(
            args.train_roots, split="train",
            dem_clamp_min=args.dem_clamp_min, dem_clamp_max=args.dem_clamp_max,
            ndsm_clamp_max=args.ndsm_clamp_max,
            max_files=args.stats_max_files, max_vals=args.stats_max_vals, seed=args.seed,
        )
        init_dem_shift  = dm
        init_dem_scale  = max(ds, 1.)
        init_ndsm_scale = max(np95, 1.)

        init_dem_scale = init_dem_shift = init_ndsm_scale = None"""
    if args.init_from_stats:
        if is_master:
            dm, ds, _, _, np95 = estimate_dem_ndsm_stats(
                args.train_roots, split="train",
                dem_clamp_min=args.dem_clamp_min,
                dem_clamp_max=args.dem_clamp_max,
                ndsm_clamp_max=args.ndsm_clamp_max,
                max_files=args.stats_max_files,
                max_vals=args.stats_max_vals,
                seed=args.seed,
            )
            vals = torch.tensor([dm, ds, np95], device=device, dtype=torch.float32)
        else:
            vals = torch.zeros(3, device=device, dtype=torch.float32)

        if distributed:
            dist.broadcast(vals, src=0)

        dm, ds, np95 = [float(x) for x in vals.tolist()]
        init_dem_shift = dm
        init_dem_scale = max(ds, 1.0)
        init_ndsm_scale = max(np95, 1.0)

    # Data
    if is_master:
        print("\n[DATA]")
    mkds = lambda r, sp, aug: NPZDemNdsmDataset(
        r, sp, augment=aug, target_size=args.target_size,
        dem_clamp_min=args.dem_clamp_min, dem_clamp_max=args.dem_clamp_max,
        ndsm_clamp_max=args.ndsm_clamp_max)

    from torch.utils.data import ConcatDataset
    tr_sets = [mkds(r,"train",True)  for r in args.train_roots]
    va_sets = [mkds(r,"val",  False) for r in args.train_roots]
    tr_ds   = ConcatDataset(tr_sets) if len(tr_sets)>1 else tr_sets[0]
    va_ds   = ConcatDataset(va_sets) if len(va_sets)>1 else va_sets[0]
    if is_master:
        print(f"  Train={len(tr_ds)}  Val={len(va_ds)}")

    fixed_idx = random.Random(args.seed).sample(range(len(va_ds)), min(args.viz_samples, len(va_ds)))
    tr_sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    va_sampler = DistributedSampler(va_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    tr_loader = DataLoader(
        tr_ds, args.batch_size,
        shuffle=(tr_sampler is None),
        sampler=tr_sampler,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )
    va_loader = DataLoader(
        va_ds, args.batch_size,
        shuffle=False,
        sampler=va_sampler,
        num_workers=max(1,args.num_workers//2), pin_memory=True
    )

    # Model
    if is_master:
        print("\n[MODEL]")
    model = DA3DemNdsmModel(
        model_name=args.model_name,
        train_backbone=args.train_backbone,
        use_feats=args.use_feats,
        feat_layers=feat_layers,
        feat_reduce_ch=args.feat_reduce_ch,
        adapter_base=args.adapter_base,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_regex=args.lora_regex,
        max_dem=args.dem_clamp_max,
        dem_min=args.dem_clamp_min,
        max_ndsm=args.ndsm_clamp_max,
        init_dem_scale=init_dem_scale,
        init_dem_shift=init_dem_shift,
        init_ndsm_scale=init_ndsm_scale,
        target_size=args.target_size,
    ).to(device)
    if distributed:
        #model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=True
)
    model_ref = model.module if distributed else model

    # Optimizer: LoRA / 그 외 분리 가능
    def _build_optim(lr_default):
        if args.use_lora and args.lr_lora is not None:
            lora_p  = [p for n,p in model.named_parameters() if p.requires_grad and "lora_" in n]
            other_p = [p for n,p in model.named_parameters() if p.requires_grad and "lora_" not in n]
            groups  = [{"params": lora_p,  "lr": args.lr_lora, "weight_decay": 0.},
                       {"params": other_p, "lr": lr_default,   "weight_decay": args.wd}]
            if is_master:
                print(f"[OPT] LoRA params={sum(p.numel() for p in lora_p)} @ lr={args.lr_lora}")
                print(f"[OPT] Other params={sum(p.numel() for p in other_p)} @ lr={lr_default}")
            return torch.optim.AdamW(groups)
        else:
            trainable = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.AdamW(trainable, lr=lr_default, weight_decay=args.wd)

    optimizer = _build_optim(args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    #scaler    = torch.amp.GradScaler("cuda", enabled=(args.amp and device=="cuda"))
    scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(args.amp and torch.cuda.is_available())
)

    cfg = {k: getattr(args, k) for k in [
        "lambda_ndsm","grad_w_dem","grad_w_ndsm",
        "huber_w_dem","huber_w_ndsm","huber_delta_dem","huber_delta_ndsm","amp"]}

    # Resume
    start_epoch, best_val, history = 1, float("inf"), []
    if args.resume and Path(args.resume).exists():
        if is_master:
            print(f"\n[RESUME] {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model_sd = model_ref.state_dict()

        # Shape-safe load: keep compatible tensors only (useful when adapter size changed).
        compatible_sd = {}
        skipped_shape = 0
        skipped_missing = 0
        for k, v in ckpt_model.items():
            if k not in model_sd:
                skipped_missing += 1
                continue
            if getattr(model_sd[k], "shape", None) == getattr(v, "shape", None):
                compatible_sd[k] = v
            else:
                skipped_shape += 1

        missing, unexpected = model_ref.load_state_dict(compatible_sd, strict=False)
        arch_changed = (skipped_shape > 0)
        if is_master:
            print(
                f"[RESUME] model loaded (shape-safe): loaded={len(compatible_sd)} "
                f"skipped_shape={skipped_shape} skipped_missing={skipped_missing} "
                f"missing_now={len(missing)} unexpected_now={len(unexpected)}"
            )
        if arch_changed:
            if is_master:
                print("[RESUME] optimizer/scaler state skipped due to architecture change (shape mismatch).")
        else:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                if is_master:
                    print("[RESUME] optimizer state skipped")
            scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("best_val", float("inf"))
        history     = ckpt.get("history", [])
        # LR 리셋
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr_lora if (args.use_lora and args.lr_lora and "lora_" in str(pg.get("params",""))) else args.lr
            pg.pop("initial_lr", None)
        remaining = args.epochs - (start_epoch - 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1,remaining), eta_min=args.lr*0.05)

    if is_master:
        (out_dir/"config.json").write_text(json.dumps(vars(args), indent=2))

    # W&B (master rank only)
    if is_master and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
            config=vars(args),
            dir=str(out_dir),
        )

    # ─── Training loop ────────────────────────────────────────────
    if is_master:
        print("\n" + "="*70 + "\nTraining Start\n" + "="*70)

    for epoch in range(start_epoch, args.epochs+1):
        if distributed:
            tr_sampler.set_epoch(epoch)
            va_sampler.set_epoch(epoch)
        lr_now = optimizer.param_groups[0]["lr"]
        if is_master:
            print(f"\n[EPOCH {epoch}/{args.epochs}]  LR={lr_now:.2e}")

        if args.unfreeze_epoch and epoch == args.unfreeze_epoch:
            model_ref.unfreeze_backbone()
            optimizer = _build_optim(args.lr * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs-epoch+1)

        tr = _run_epoch(model, tr_loader, optimizer, scaler, device, cfg, train=True, is_master=is_master)
        va = _run_epoch(model, va_loader, None,      None,   device, cfg, train=False, is_master=is_master)
        scheduler.step()

        if is_master:
            print(f"  Train: loss={tr['loss']:.4f} | dem_h={tr['dem_huber']:.4f} "
                  f"dem_g={tr['dem_grad']:.4f} | nd_h={tr['ndsm_huber']:.4f} "
                  f"nd_g={tr['ndsm_grad']:.4f} | skip={int(tr['skipped'])}")
            print(f"         MAE/RMSE: DEM={tr['dem_mae']:.3f}/{tr['dem_rmse']:.3f}m  "
                  f"nDSM={tr['ndsm_mae']:.3f}/{tr['ndsm_rmse']:.3f}m")
            print(f"  Val:   loss={va['loss']:.4f} | dem_h={va['dem_huber']:.4f} "
                  f"dem_g={va['dem_grad']:.4f} | nd_h={va['ndsm_huber']:.4f} "
                  f"nd_g={va['ndsm_grad']:.4f} | skip={int(va['skipped'])}")
            print(f"         MAE/RMSE: DEM={va['dem_mae']:.3f}/{va['dem_rmse']:.3f}m  "
                  f"nDSM={va['ndsm_mae']:.3f}/{va['ndsm_rmse']:.3f}m")

        # LoRA norm 출력
        if args.use_lora and is_master:
            a_norms, b_norms = [], []
            for m in model_ref.modules():
                if isinstance(m, LoRALinear):
                    a_norms.append(m.lora_A.weight.detach().norm().item())
                    b_norms.append(m.lora_B.weight.detach().norm().item())
            print(f"  LoRA: {len(a_norms)} layers | "
                  f"A_norm avg={np.mean(a_norms):.4f} max={np.max(a_norms):.4f} | "
                  f"B_norm avg={np.mean(b_norms):.4f} max={np.max(b_norms):.4f}")

        if is_master and fixed_idx:
            visualize_fixed_samples(model_ref, va_ds, fixed_idx, device, epoch, vis_dir)

        if is_master:
            history.append({"epoch":epoch,"lr":lr_now,"train":tr,"val":va})
            ckpt = {"epoch":epoch,"model":model_ref.state_dict(),
                    "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                    "scaler":scaler.state_dict(),"best_val":best_val,
                    "history":history,"config":vars(args)}
            torch.save(ckpt, out_dir/"last.pt")
            (out_dir/"history.json").write_text(json.dumps(history, indent=2))

            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "lr": lr_now,
                    "train/loss": tr["loss"],
                    "train/dem_mae": tr["dem_mae"],
                    "train/dem_rmse": tr["dem_rmse"],
                    "train/dem_mse": tr["dem_rmse"] ** 2,
                    "train/ndsm_mae": tr["ndsm_mae"],
                    "train/ndsm_rmse": tr["ndsm_rmse"],
                    "train/ndsm_mse": tr["ndsm_rmse"] ** 2,
                    "val/loss": va["loss"],
                    "val/dem_mae": va["dem_mae"],
                    "val/dem_rmse": va["dem_rmse"],
                    "val/dem_mse": va["dem_rmse"] ** 2,
                    "val/ndsm_mae": va["ndsm_mae"],
                    "val/ndsm_rmse": va["ndsm_rmse"],
                    "val/ndsm_mse": va["ndsm_rmse"] ** 2,
                }, step=epoch)

            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(ckpt, out_dir/"best.pt")
                print(f"  ✓ Best saved! val_loss={best_val:.4f}")

            if epoch % 10 == 0:
                torch.save(ckpt, out_dir/f"epoch{epoch:03d}.pt")

    if is_master:
        print(f"\n{'='*70}\nDone. Best val loss={best_val:.4f}\n{'='*70}")
        if args.use_wandb:
            wandb.summary["best_val_loss"] = best_val
            wandb.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

