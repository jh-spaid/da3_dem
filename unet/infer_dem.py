#!/usr/bin/env python3
# infer_da3_dem.py - DA3 DEM+nDSM checkpoint inference

import math
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import from_origin, Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
except Exception:
    rasterio = None
    Window = None
    from_origin = None
    Affine = None
    CRS = None
    Resampling = None

if Window is None:
    class Window:
        def __init__(self, col_off, row_off, width, height):
            self.col_off = col_off
            self.row_off = row_off
            self.width = width
            self.height = height
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from depth_anything_3.api import DepthAnything3


# -------------------- UNet adapter --------------------
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


# -------------------- Model --------------------
class DA3DemNdsmModel(nn.Module):
    """
    Depth Anything V3 for DEM + nDSM regression
    Must match training checkpoint structure
    """
    def __init__(
        self,
        model_name="depth-anything/DA3-BASE",
        train_backbone=False,
        max_dem=2000.0,
        max_ndsm=300.0,
    ):
        super().__init__()

        print(f"[DA3DemNdsm] Loading model: {model_name}")
        da3_wrapper = DepthAnything3.from_pretrained(model_name)
        self.net = da3_wrapper.model

        if not train_backbone:
            if hasattr(self.net, "net"):
                for p in self.net.net.parameters():
                    p.requires_grad = False
            if hasattr(self.net, "head"):
                for p in self.net.head.parameters():
                    p.requires_grad = False

        # Shared adapter (MUST match training code exactly)
        self.shared_adapter = UNetAdapter(in_ch=1, base=32, out_ch=32)

        self.dem_head = nn.Conv2d(32, 1, 1)
        self.ndsm_head = nn.Conv2d(32, 1, 1)

        self.dem_scale = nn.Parameter(torch.tensor(100.0))
        self.dem_shift = nn.Parameter(torch.tensor(0.0))
        self.ndsm_scale = nn.Parameter(torch.tensor(30.0))

        self.max_dem = float(max_dem)
        self.max_ndsm = float(max_ndsm)

    def forward(self, x, return_debug=False):
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalized RGB
        Returns:
            dem:  (B, 1, H, W) meters
            ndsm: (B, 1, H, W) meters >=0
            dsm:  (B, 1, H, W) meters (optional, = dem + ndsm)
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)

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
        ndsm = self.ndsm_scale * F.softplus(ndsm_rel)
        ndsm = torch.clamp(ndsm, 0.0, self.max_ndsm)

        if return_debug:
            dsm = dem + ndsm
            return dem, ndsm, dsm, depth, feat
        
        return dem, ndsm


def load_model_da3(ckpt_path, device):
    """Load checkpoint and reconstruct model"""
    print(f"[CKPT] Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = ckpt.get("config", {})
    model_name = cfg.get("model_name", "depth-anything/DA3-BASE")
    max_dem = cfg.get("dem_clamp_max", 2000.0)
    max_ndsm = cfg.get("ndsm_clamp_max", 300.0)

    model = DA3DemNdsmModel(
        model_name=model_name,
        max_dem=max_dem,
        max_ndsm=max_ndsm
    )
    
    model.load_state_dict(ckpt["model"], strict=True)
    
    print(f"[CKPT] ✓ Loaded DA3DemNdsmModel ({model_name})")
    print(f"       max_dem={max_dem}, max_ndsm={max_ndsm}")
    print(f"       dem_scale={model.dem_scale.item():.3f}, dem_shift={model.dem_shift.item():.3f}")
    print(f"       ndsm_scale={model.ndsm_scale.item():.3f}")

    return model


# -------------------- Utils --------------------
def gaussian_2d(size, sigma=None):
    H, W = size
    if sigma is None:
        sigma = 0.25 * min(H, W)
    cy, cx = (H - 1) / 2, (W - 1) / 2
    y = np.arange(H).reshape(-1, 1)
    x = np.arange(W).reshape(1, -1)
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    g /= g.max()
    return g.astype(np.float32)


def compute_tile_size_for_grid(img_size, n_grid, overlap_ratio=0.67):
    H, W = img_size
    stride_h = H / n_grid
    stride_w = W / n_grid

    tile_h = int(stride_h / (1 - overlap_ratio))
    tile_w = int(stride_w / (1 - overlap_ratio))
    tile_size = max(tile_h, tile_w)

    max_tile = min(H, W)
    tile_size = min(tile_size, max_tile)
    tile_size = max(tile_size, 256)

    stride = int(tile_size * (1 - overlap_ratio))
    stride = max(stride, 1)
    ny = (H + stride - 1) // stride
    nx = (W + stride - 1) // stride
    return tile_size, (ny, nx)


def read_rgb_tile(src, window):
    arr = src.read(
        indexes=list(range(1, min(4, src.count + 1))),
        window=window, boundless=True, fill_value=0
    )
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)

    dt = src.dtypes[0]

    if dt == "uint8":
        arr /= 255.0
    elif dt == "uint16":
        arr /= 65535.0
    else:
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        if mx > 1.5:
            arr /= (255.0 if mx <= 300.0 else 65535.0)

    return np.clip(arr, 0.0, 1.0)


def read_rgb_tile_array(img, window):
    y0, x0 = int(window.row_off), int(window.col_off)
    h, w = int(window.height), int(window.width)
    H, W = img.shape[:2]

    out = np.zeros((h, w, 3), dtype=img.dtype)

    y1 = min(y0 + h, H)
    x1 = min(x0 + w, W)
    if y1 > y0 and x1 > x0:
        out[0 : y1 - y0, 0 : x1 - x0] = img[y0:y1, x0:x1]
    return out


def _round_to_14(x: int) -> int:
    return int(math.ceil(x / 14) * 14)


def predict_tile(model, rgb, device, mean, std, infer_upsample=1, amp=False, output_type="dem"):
    """
    Predict DEM or nDSM for a single tile
    
    Args:
        output_type: "dem", "ndsm", or "dsm"
    """
    H, W = rgb.shape[:2]

    if infer_upsample > 1:
        s = infer_upsample
        rgb = cv2.resize(rgb, (W * s, H * s), interpolation=cv2.INTER_CUBIC)
        H2, W2 = rgb.shape[:2]
    else:
        H2, W2 = H, W

    H14, W14 = _round_to_14(H2), _round_to_14(W2)
    if (H14, W14) != (H2, W2):
        rgb_in = cv2.resize(rgb, (W14, H14), interpolation=cv2.INTER_LINEAR)
    else:
        rgb_in = rgb

    x = torch.from_numpy(rgb_in).permute(2, 0, 1).unsqueeze(0).to(device)
    x = (x - mean) / std

    with torch.no_grad():
        if amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred_dem, pred_ndsm = model(x)
        else:
            pred_dem, pred_ndsm = model(x)

    # Select output
    if output_type == "dem":
        pred = pred_dem
    elif output_type == "ndsm":
        pred = pred_ndsm
    elif output_type == "dsm":
        pred = pred_dem + pred_ndsm
    else:
        raise ValueError(f"Invalid output_type: {output_type}")

    pred = pred.squeeze(0).squeeze(0).float().detach().cpu().numpy()

    if pred.shape[:2] != (H2, W2):
        pred = cv2.resize(pred, (W2, H2), interpolation=cv2.INTER_LINEAR)

    if infer_upsample > 1:
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)

    return pred


def predict_single_scale_array(
    img, model, device, mean, std,
    tile_size, overlap_ratio, infer_upsample, amp, sigma_factor=8,
    output_type="dem"
):
    """Predict on numpy array (for PNG/JPG inputs)"""
    H, W = img.shape[:2]
    tile_size = min(tile_size, H, W)

    overlap = int(tile_size * overlap_ratio)
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"Invalid stride: tile_size={tile_size}, overlap={overlap}")

    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    sigma = tile_size / sigma_factor
    gauss_weight = gaussian_2d((tile_size, tile_size), sigma=sigma)

    ny = (H + stride - 1) // stride
    nx = (W + stride - 1) // stride
    total = nx * ny

    pbar = tqdm(total=total, desc=f"  {ny}×{nx} grid (tile={tile_size})")
    for iy in range(ny):
        y0 = iy * stride
        y0 = max(0, min(y0, H - tile_size))

        for ix in range(nx):
            x0 = ix * stride
            x0 = max(0, min(x0, W - tile_size))

            window = Window(x0, y0, tile_size, tile_size)
            rgb = read_rgb_tile_array(img, window)
            actual_h, actual_w = rgb.shape[:2]

            pred = predict_tile(model, rgb, device, mean, std, infer_upsample, amp, output_type)

            if actual_h != tile_size or actual_w != tile_size:
                weight = cv2.resize(gauss_weight, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)
            else:
                weight = gauss_weight

            y_end = min(y0 + actual_h, H)
            x_end = min(x0 + actual_w, W)

            acc[y0:y_end, x0:x_end] += pred[:y_end - y0, :x_end - x0] * weight[:y_end - y0, :x_end - x0]
            wsum[y0:y_end, x0:x_end] += weight[:y_end - y0, :x_end - x0]
            pbar.update(1)

    pbar.close()
    result = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 1e-9)
    return result


def predict_single_scale(
    src, model, device, mean, std,
    tile_size, overlap_ratio, infer_upsample, amp, sigma_factor=8,
    output_type="dem"
):
    """Predict on rasterio source (for TIF inputs)"""
    H, W = src.height, src.width
    tile_size = min(tile_size, H, W)

    overlap = int(tile_size * overlap_ratio)
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"Invalid stride: tile_size={tile_size}, overlap={overlap}")

    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    sigma = tile_size / sigma_factor
    gauss_weight = gaussian_2d((tile_size, tile_size), sigma=sigma)

    ny = (H + stride - 1) // stride
    nx = (W + stride - 1) // stride
    total = nx * ny

    pbar = tqdm(total=total, desc=f"  {ny}×{nx} grid (tile={tile_size})")
    for iy in range(ny):
        y0 = iy * stride
        y0 = max(0, min(y0, H - tile_size))

        for ix in range(nx):
            x0 = ix * stride
            x0 = max(0, min(x0, W - tile_size))

            window = Window(x0, y0, tile_size, tile_size)
            rgb = read_rgb_tile(src, window)
            actual_h, actual_w = rgb.shape[:2]

            pred = predict_tile(model, rgb, device, mean, std, infer_upsample, amp, output_type)

            if actual_h != tile_size or actual_w != tile_size:
                weight = cv2.resize(gauss_weight, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)
            else:
                weight = gauss_weight

            y_end = min(y0 + actual_h, H)
            x_end = min(x0 + actual_w, W)

            acc[y0:y_end, x0:x_end] += pred[:y_end - y0, :x_end - x0] * weight[:y_end - y0, :x_end - x0]
            wsum[y0:y_end, x0:x_end] += weight[:y_end - y0, :x_end - x0]
            pbar.update(1)

    pbar.close()
    result = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 1e-9)
    return result


def predict_multiscale_adaptive(
    src, model, device,
    max_grid=10, grid_levels=None, overlap_ratio=0.67,
    infer_upsample=2, amp=False, global_weight=0.2,
    output_type="dem"
):
    """Multi-scale prediction for DEM, nDSM, or DSM"""
    if isinstance(src, np.ndarray):
        H, W = src.shape[:2]
        is_array = True
    else:
        H, W = src.height, src.width
        is_array = False

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print("\n" + "=" * 60)
    print(f"DA3 DEM+nDSM MULTI-SCALE PREDICTION ({output_type.upper()})")
    print("=" * 60)
    print(f"Input size: {W} × {H}")
    print(f"Max grid: {max_grid}×{max_grid}")
    print(f"Output: {output_type}")

    if grid_levels is None:
        max_possible_grid = max(3, int(min(H, W) / 500))
        grid_levels = []
        for g in [3, 5, 7, max_grid]:
            if g <= max_possible_grid:
                grid_levels.append(g)
        if len(grid_levels) < 2:
            grid_levels = [max(2, max_possible_grid - 1), max_possible_grid]

    print(f"Grid levels: {grid_levels}")

    # Global context
    global_pred = None
    if global_weight > 0:
        print(f"\n[GLOBAL] Downsampled prediction ({output_type})...")
        scale = min(1.0, 1024.0 / max(H, W))
        global_h = int(H * scale)
        global_w = int(W * scale)

        if is_array:
            global_rgb = cv2.resize(src, (global_w, global_h), interpolation=cv2.INTER_LINEAR)
        else:
            if Resampling is None:
                raise RuntimeError("rasterio.enums.Resampling not available")
            global_rgb = src.read(
                indexes=list(range(1, min(4, src.count + 1))),
                out_shape=(min(3, src.count), global_h, global_w),
                resampling=Resampling.bilinear
            )
            if global_rgb.shape[0] == 1:
                global_rgb = np.repeat(global_rgb, 3, axis=0)
            global_rgb = np.transpose(global_rgb, (1, 2, 0))

            if global_rgb.dtype == np.uint8:
                global_rgb = global_rgb.astype(np.float32) / 255.0
            else:
                global_rgb = global_rgb.astype(np.float32)
                mx = global_rgb.max() if global_rgb.size else 0.0
                if mx > 1.5:
                    global_rgb = global_rgb / (65535.0 if mx > 300 else mx)
            global_rgb = np.clip(global_rgb, 0.0, 1.0)

        global_pred_small = predict_tile(model, global_rgb, device, mean, std, 
                                         infer_upsample=1, amp=amp, output_type=output_type)
        global_pred = cv2.resize(global_pred_small, (W, H), interpolation=cv2.INTER_LINEAR)
        print(f"  ✓ Global: {global_w}×{global_h} → {W}×{H}")

    # Multi-scale
    all_preds = []
    for n_grid in grid_levels:
        print(f"\n[SCALE] Target grid: ~{n_grid}×{n_grid}")
        tile_size, (actual_ny, actual_nx) = compute_tile_size_for_grid((H, W), n_grid, overlap_ratio)
        print(f"  Tile size: {tile_size}, Grid: {actual_ny}×{actual_nx}")

        if tile_size < 256:
            print("  ⚠️  Skip (tile too small)")
            continue

        if is_array:
            pred = predict_single_scale_array(
                src, model, device, mean, std,
                tile_size=tile_size,
                overlap_ratio=overlap_ratio,
                infer_upsample=infer_upsample,
                amp=amp,
                sigma_factor=8,
                output_type=output_type
            )
        else:
            pred = predict_single_scale(
                src, model, device, mean, std,
                tile_size=tile_size,
                overlap_ratio=overlap_ratio,
                infer_upsample=infer_upsample,
                amp=amp,
                sigma_factor=8,
                output_type=output_type
            )
        all_preds.append(pred)
        print("  ✓ Completed")

    print(f"\n[FUSION] Combining {len(all_preds)} scales...")
    if len(all_preds) == 0:
        raise RuntimeError("No valid predictions generated")

    local_pred = np.mean(all_preds, axis=0)

    if global_pred is not None and global_weight > 0:
        local_weight = 1.0 - global_weight
        final_pred = global_pred * global_weight + local_pred * local_weight
        print(f"  Global: {global_weight:.1%}, Local: {local_weight:.1%}")
    else:
        final_pred = local_pred

    print("  ✓ Fusion complete")
    return final_pred


# -------------------- Main --------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="DA3 DEM+nDSM inference")

    ap.add_argument("--in_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    
    ap.add_argument("--output_type", type=str, default="all", 
                    choices=["dem", "ndsm", "dsm", "all"],
                    help="Which output to generate")

    ap.add_argument("--max_grid", type=int, default=10)
    ap.add_argument("--grid_levels", nargs="+", type=int, default=None)
    ap.add_argument("--overlap_ratio", type=float, default=0.67)
    ap.add_argument("--infer_upsample", type=int, default=2)
    ap.add_argument("--global_weight", type=float, default=0.2)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--save_png", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    model = load_model_da3(args.ckpt, device)
    model = model.to(device).eval()

    in_path = Path(args.in_path)
    is_tif = in_path.suffix.lower() in [".tif", ".tiff"]

    # Determine which outputs to generate
    outputs_to_gen = ["dem", "ndsm", "dsm"] if args.output_type == "all" else [args.output_type]

    print(f"\n[INPUT] {in_path}")
    print(f"  Outputs to generate: {', '.join([o.upper() for o in outputs_to_gen])}")

    try:
        if is_tif:
            if rasterio is None:
                raise RuntimeError("rasterio required for TIF")
            
            with rasterio.open(in_path) as src:
                print(f"  Size: {src.width} × {src.height}")
                print(f"  CRS: {src.crs}")
                
                results = {}
                for out_type in outputs_to_gen:
                    print(f"\n{'='*60}")
                    print(f"Generating {out_type.upper()}...")
                    print('='*60)
                    
                    pred = predict_multiscale_adaptive(
                        src, model, device,
                        max_grid=args.max_grid,
                        grid_levels=args.grid_levels,
                        overlap_ratio=args.overlap_ratio,
                        infer_upsample=args.infer_upsample,
                        amp=args.amp,
                        global_weight=args.global_weight,
                        output_type=out_type
                    )
                    results[out_type] = pred

                    # Save TIF
                    out_tif = out_dir / f"pred_{out_type}.tif"
                    meta = src.meta.copy()
                    meta.update(
                        count=1, dtype="float32", nodata=-9999.0,
                        compress="deflate", predictor=3, tiled=True,
                        blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER"
                    )
                    with rasterio.open(out_tif, "w", **meta) as dst:
                        dst.write(pred.astype(np.float32), 1)
                    
                    print(f"\n[SAVED] {out_tif}")
                    
                    # Stats
                    valid = np.isfinite(pred) & (pred != 0)
                    if valid.any():
                        v = pred[valid]
                        print(f"  Min: {v.min():.2f}m")
                        print(f"  Max: {v.max():.2f}m")
                        print(f"  Mean: {v.mean():.2f}m")
                        print(f"  Std: {v.std():.2f}m")

                        # PNG
                        if args.save_png:
                            vmin, vmax = np.percentile(v, [1, 99])
                            plt.figure(figsize=(15, 12))
                            im = plt.imshow(pred, cmap="viridis" if out_type != "ndsm" else "magma",
                                           vmin=vmin, vmax=vmax)
                            plt.title(f"{out_type.upper()} Prediction", fontsize=16, weight="bold")
                            plt.colorbar(im, fraction=0.046, pad=0.04, label="Height (m)")
                            plt.axis("off")
                            plt.tight_layout()
                            
                            out_png = out_dir / f"pred_{out_type}.png"
                            plt.savefig(out_png, dpi=300, bbox_inches="tight")
                            plt.close()
                            print(f"  PNG: {out_png}")

        else:
            # PNG/JPG handling
            print(f"  Loading image file...")
            bgr = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise FileNotFoundError(f"Failed to read image: {in_path}")
            
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            # Normalize to [0,1]
            if rgb.dtype == np.uint8 or rgb.max() <= 255:
                rgb = rgb / 255.0
            elif rgb.max() > 1.5:
                rgb = rgb / 65535.0
            rgb = np.clip(rgb, 0.0, 1.0)
            
            print(f"  Size: {rgb.shape[1]} × {rgb.shape[0]}")
            
            results = {}
            for out_type in outputs_to_gen:
                print(f"\n{'='*60}")
                print(f"Generating {out_type.upper()}...")
                print('='*60)
                
                pred = predict_multiscale_adaptive(
                    rgb, model, device,
                    max_grid=args.max_grid,
                    grid_levels=args.grid_levels,
                    overlap_ratio=args.overlap_ratio,
                    infer_upsample=args.infer_upsample,
                    amp=args.amp,
                    global_weight=args.global_weight,
                    output_type=out_type
                )
                results[out_type] = pred

                # Save TIF (if rasterio available)
                if rasterio is not None and from_origin is not None:
                    out_tif = out_dir / f"pred_{out_type}.tif"
                    meta = {
                        "driver": "GTiff",
                        "height": rgb.shape[0],
                        "width": rgb.shape[1],
                        "count": 1,
                        "dtype": "float32",
                        "nodata": -9999.0,
                        "transform": from_origin(0, 0, 1, 1),
                        "compress": "deflate",
                        "predictor": 3,
                    }
                    with rasterio.open(out_tif, "w", **meta) as dst:
                        dst.write(pred.astype(np.float32), 1)
                    print(f"\n[SAVED] {out_tif}")
                
                # Stats
                valid = np.isfinite(pred) & (pred != 0)
                if valid.any():
                    v = pred[valid]
                    print(f"  Min: {v.min():.2f}m")
                    print(f"  Max: {v.max():.2f}m")
                    print(f"  Mean: {v.mean():.2f}m")
                    print(f"  Std: {v.std():.2f}m")

                    # PNG
                    if args.save_png:
                        vmin, vmax = np.percentile(v, [1, 99])
                        plt.figure(figsize=(15, 12))
                        im = plt.imshow(pred, cmap="viridis" if out_type != "ndsm" else "magma",
                                       vmin=vmin, vmax=vmax)
                        plt.title(f"{out_type.upper()} Prediction", fontsize=16, weight="bold")
                        plt.colorbar(im, fraction=0.046, pad=0.04, label="Height (m)")
                        plt.axis("off")
                        plt.tight_layout()
                        
                        out_png = out_dir / f"pred_{out_type}.png"
                        plt.savefig(out_png, dpi=300, bbox_inches="tight")
                        plt.close()
                        print(f"  PNG: {out_png}")

        print("\n" + "=" * 60)
        print("✓ Inference Complete!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

    """ python infer_dem.py \
  --in_path /home/wogmlcho86/korea/sample/123_w.png \
  --overlap_ratio 0.67 \
  --infer_upsample 2 \
  --out_dir ./output \
  --ckpt ./runs/da3_dem_refine/best.pt \
  --output_type dem \
  --max_grid 20 \
  --grid_levels 10 15 20 \
  --amp \
  --save_png"""
