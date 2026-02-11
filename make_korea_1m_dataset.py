#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_korea_4326_dataset.py

QGIS 출력 데이터셋 구조 (EPSG:4326):
  dataset_root/
    RGB/      35716082.tif
    DSM/      35716082.tif
    DEM/      35716082.tif (선택적)

출력:
  out_dir/
    train/*.npz
    val/*.npz
    visualizations/train/*.png
    visualizations/val/*.png

[방법 1 + 보완]
1) "공간 블록" 단위로 균형 샘플링:
   - block_key = f"{tile_id}_sx{x0}_sy{y0}"  (tile_read 윈도우 단위)
   - 블록당 최대 패치 수(--max_patches_per_block)로 캡(oversampling 방지)
2) Train/Val split도 block 단위로 수행 (--split_by 기본)
   - 같은 블록의 패치가 train/val에 섞이는 누수 최소화
3) split_by=block_tile 옵션 추가
   - 타일 내부에서 block 단위로 split
   - 중복 없음 + 분포 유사성을 동시에 만족
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.crs import CRS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Visualization
# -----------------------------
def visualize_samples(output_dir, dataset_name, split="train", n_samples=5):
    output_dir = Path(output_dir)
    split_dir = output_dir / split
    vis_dir = output_dir / "visualizations" / split
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not split_dir.exists():
        print(f"  ⚠️  Split directory not found: {split_dir}")
        return

    npz_files = sorted(split_dir.glob("*.npz"))[:n_samples]
    if len(npz_files) == 0:
        print(f"  ⚠️  No NPZ files found in {split_dir}")
        return

    print(f"\n[{dataset_name.upper()} - {split.upper()}] Creating visualizations...")

    for idx, npz_path in enumerate(npz_files):
        try:
            data = np.load(npz_path, allow_pickle=True)
            rgb = data["rgb"]
            dsm = data["dsm"]
            mask = data["mask"].astype(bool)

            if rgb.ndim == 3 and rgb.shape[0] == 3:
                rgb_vis = np.transpose(rgb, (1, 2, 0))
            else:
                rgb_vis = rgb

            valid_dsm = dsm[mask]
            if len(valid_dsm) == 0:
                print(f"  ⚠️  No valid DSM data in {npz_path.name}")
                continue

            dsm_min = float(valid_dsm.min())
            dsm_max = float(valid_dsm.max())
            dsm_mean = float(valid_dsm.mean())
            dsm_std = float(valid_dsm.std())

            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            axes[0, 0].imshow(rgb_vis)
            axes[0, 0].set_title("RGB Image", fontsize=14, fontweight="bold")
            axes[0, 0].axis("off")

            dsm_display = dsm.copy()
            dsm_display[~mask] = np.nan
            im1 = axes[0, 1].imshow(dsm_display, cmap="terrain", vmin=dsm_min, vmax=dsm_max)
            axes[0, 1].set_title(f"DSM Height\n[{dsm_min:.1f}, {dsm_max:.1f}]m",
                                 fontsize=14, fontweight="bold")
            axes[0, 1].axis("off")
            cbar1 = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            cbar1.set_label("Height (m)", fontsize=10)

            axes[1, 0].hist(valid_dsm.flatten(), bins=50, alpha=0.7, edgecolor="black")
            axes[1, 0].axvline(dsm_mean, color="red", linestyle="--", linewidth=2,
                               label=f"Mean: {dsm_mean:.1f}m")
            axes[1, 0].set_title("DSM Distribution", fontsize=14, fontweight="bold")
            axes[1, 0].set_xlabel("Height (m)", fontsize=11)
            axes[1, 0].set_ylabel("Frequency", fontsize=11)
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].axis("off")
            meta_str = data.get("meta", "{}")
            if isinstance(meta_str, bytes):
                meta_str = meta_str.decode("utf-8")
            meta = json.loads(meta_str) if isinstance(meta_str, str) else {}

            stats_text = f"""
Dataset: {dataset_name.upper()}
Split:   {split.upper()}
File:    {npz_path.name}

DSM mean/std: {dsm_mean:.2f} / {dsm_std:.2f}
DSM min/max:  {dsm_min:.2f} / {dsm_max:.2f}

Valid ratio: {100 * mask.mean():.1f}%
RGB shape:   {rgb.shape}
DSM shape:   {dsm.shape}
GSD tgt:     {meta.get("tgt_gsd", "N/A")} m/px
TileID:      {meta.get("tile_id", "N/A")}
Block:       {meta.get("block_key", "N/A")}
"""
            axes[1, 1].text(
                0.05, 0.5, stats_text,
                fontsize=11, verticalalignment="center",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3)
            )

            plt.suptitle(f"{dataset_name.upper()} - {split.upper()} - Sample {idx+1}",
                         fontsize=16, fontweight="bold", y=0.98)
            plt.tight_layout()

            output_path = vis_dir / f"{dataset_name}_{split}_sample_{idx+1:02d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"  ✓ Saved: {output_path.name}")

        except Exception as e:
            print(f"  ⚠️  Error visualizing {npz_path.name}: {e}")

    print(f"  → Visualizations saved to: {vis_dir}")


# -----------------------------
# Utils
# -----------------------------
def city_name_from_root(root: Path) -> str:
    return "korea"


def extract_tile_key(name: str) -> Optional[str]:
    """
    파일명에서 타일 ID 추출:
      35716082.tif -> "35716082"
      35716082.tfw -> "35716082"
    """
    name_lower = name.lower()
    name_base = name_lower.replace('.tif', '').replace('.tiff', '').replace('.tfw', '').replace('.prj', '')
    m = re.match(r'^(\d{6,})$', name_base)
    if m:
        return m.group(1)
    return None


def find_image_files(dataset_root: Path) -> List[Path]:
    img_dir = dataset_root / "RGB"
    if not img_dir.exists():
        raise RuntimeError(f"RGB directory not found: {img_dir}")
    candidates = sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.tiff"))
    return candidates


def find_dsm_files(dataset_root: Path) -> List[Path]:
    dsm_dir = dataset_root / "DSM"
    if not dsm_dir.exists():
        raise RuntimeError(f"DSM directory not found: {dsm_dir}")
    tifs = sorted(dsm_dir.glob("*.tif")) + sorted(dsm_dir.glob("*.tiff"))
    return tifs


def find_dem_files(dataset_root: Path) -> List[Path]:
    dem_dir = dataset_root / "DEM"
    if not dem_dir.exists():
        return []
    tifs = sorted(dem_dir.glob("*.tif")) + sorted(dem_dir.glob("*.tiff"))
    return tifs


def build_raster_index(files: List[Path]) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in files:
        k = extract_tile_key(p.name)
        if k is not None:
            idx[k] = p
    return idx


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_wkt(crs) -> Optional[str]:
    try:
        return crs.to_wkt() if crs else None
    except Exception:
        return None


def read_resampled_window(
    src: rasterio.io.DatasetReader,
    window: Window,
    out_h: int,
    out_w: int,
    resampling: Resampling,
    indexes=None,
    boundless: bool = True,
    fill_value=0
) -> np.ndarray:
    data = src.read(
        indexes=indexes,
        window=window,
        out_shape=(src.count if indexes is None else len(indexes), out_h, out_w),
        resampling=resampling,
        boundless=boundless,
        fill_value=fill_value
    )
    return data


def dataset_gsd(ds: rasterio.io.DatasetReader) -> float:
    t = ds.transform
    if ds.crs and ds.crs.to_epsg() == 4326:
        return float(abs(t.a)) * 111000
    else:
        return float(abs(t.a))


def compute_scale(src_gsd: float, tgt_gsd: float) -> float:
    return float(src_gsd) / float(tgt_gsd)


def group_split(
    items: List[Tuple[Path, str, str]],
    train_ratio: float,
    seed: int,
    split_by: str = "block",
) -> Tuple[List[Path], List[Path]]:
    """
    items: [(npz_path, tile_id, block_key), ...]  (현재는 tmp_dir 아래)
    split_by:
      - "patch": 완전 랜덤
      - "block": block_key 단위로 묶어서 split
      - "tile":  tile_id 단위로 묶어서 split
      - "block_tile": 타일 내부에서 block 단위 split (중복 없음 + 분포 유사)
    """
    rng = random.Random(seed)

    if split_by == "patch":
        paths = [p for p, _, _ in items]
        rng.shuffle(paths)
        n_train = int(len(paths) * train_ratio)
        return paths[:n_train], paths[n_train:]

    if split_by in ("block", "tile"):
        groups: Dict[str, List[Path]] = {}
        for p, tile_id, block_key in items:
            key = block_key if split_by == "block" else tile_id
            groups.setdefault(key, []).append(p)

        keys = list(groups.keys())
        rng.shuffle(keys)
        n_train_keys = int(len(keys) * train_ratio)

        train_keys = set(keys[:n_train_keys])
        train_paths: List[Path] = []
        val_paths: List[Path] = []

        for k, plist in groups.items():
            if k in train_keys:
                train_paths.extend(plist)
            else:
                val_paths.extend(plist)

        rng.shuffle(train_paths)
        rng.shuffle(val_paths)
        return train_paths, val_paths

    if split_by == "block_tile":
        # tile -> block -> [paths]
        tiles: Dict[str, Dict[str, List[Path]]] = {}
        for p, tile_id, block_key in items:
            tiles.setdefault(tile_id, {}).setdefault(block_key, []).append(p)

        train_paths: List[Path] = []
        val_paths: List[Path] = []

        for tile_id, blocks in tiles.items():
            block_keys = list(blocks.keys())
            rng.shuffle(block_keys)

            if len(block_keys) == 1:
                if rng.random() < train_ratio:
                    train_keys = set(block_keys)
                else:
                    train_keys = set()
            else:
                n_train = int(round(len(block_keys) * train_ratio))
                n_train = min(max(1, n_train), len(block_keys) - 1)
                train_keys = set(block_keys[:n_train])

            for bk, plist in blocks.items():
                if bk in train_keys:
                    train_paths.extend(plist)
                else:
                    val_paths.extend(plist)

        rng.shuffle(train_paths)
        rng.shuffle(val_paths)
        return train_paths, val_paths

    raise ValueError(f"Unknown split_by: {split_by}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="한국 데이터셋 생성 (EPSG:4326)")

    ap.add_argument("--dataset_root", type=str, required=True,
                    help="데이터셋 루트 (RGB/, DSM/, DEM/ 포함)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="출력 디렉토리")

    ap.add_argument("--tgt_gsd", type=float, default=1.0,
                    help="목표 GSD (미터)")

    ap.add_argument("--tile_read", type=int, default=1024,
                    help="한 번에 읽을 타일 크기")
    ap.add_argument("--stride_read", type=int, default=1024,
                    help="타일 읽기 스트라이드")

    ap.add_argument("--patch", type=int, default=512,
                    help="패치 크기")
    ap.add_argument("--stride_patch", type=int, default=64,
                    help="패치 스트라이드")

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--max_patches", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--min_valid_ratio", type=float, default=0.7)
    ap.add_argument("--dsm_nodata", type=float, default=-9999)
    ap.add_argument("--dem_nodata", type=float, default=-9999)

    ap.add_argument("--use_dem", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument("--visualize", action="store_true", default=False)
    ap.add_argument("--n_viz_samples", type=int, default=5)

    # ===== 방법 1 핵심 옵션 =====
    ap.add_argument("--max_patches_per_block", type=int, default=20,
                    help="(균형 샘플링) 블록(tile_read 윈도우)당 최대 저장 패치 수. 0이면 제한 없음.")
    ap.add_argument("--split_by", type=str, default="block",
                    choices=["patch", "block", "tile", "block_tile"],
                    help="train/val split 단위: patch, block, tile, block_tile(타일 내부 block split)")

    args = ap.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    city = city_name_from_root(root)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("한국 데이터셋 생성 (EPSG:4326) [균형 샘플링: block cap + group split]")
    print("=" * 70)

    images = find_image_files(root)
    dsm_files = find_dsm_files(root)
    dem_files = find_dem_files(root) if args.use_dem else []

    if not images:
        raise RuntimeError(f"No RGB tiles found under {root / 'RGB'}")
    if not dsm_files:
        raise RuntimeError(f"No DSM tiles found under {root / 'DSM'}")

    dsm_index = build_raster_index(dsm_files)
    dem_index = build_raster_index(dem_files) if dem_files else {}

    print(f"[INFO] city={city}")
    print(f"[INFO] RGB tiles={len(images)}")
    print(f"[INFO] DSM tiles={len(dsm_files)} indexed={len(dsm_index)}")
    if args.use_dem:
        print(f"[INFO] DEM tiles={len(dem_files)} indexed={len(dem_index)}")

    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    tmp_dir = out_dir / "_tmp"
    ensure_dir(train_dir)
    ensure_dir(val_dir)
    ensure_dir(tmp_dir)

    # kept: [(tmp_path, tile_id, block_key), ...]
    kept_items: List[Tuple[Path, str, str]] = []
    total_candidates = 0
    total_kept = 0

    # 블록별 저장 캡
    block_counts: Dict[str, int] = {}

    for img_path in images:
        tile_id = extract_tile_key(img_path.name)
        if tile_id is None:
            if args.verbose:
                print(f"[WARN] cannot parse tile id: {img_path.name}")
            continue

        if tile_id not in dsm_index:
            if args.verbose:
                print(f"[WARN] DSM not found for tile_id={tile_id}, skip {img_path.name}")
            continue

        dsm_path = dsm_index[tile_id]
        dem_path = dem_index.get(tile_id, None) if args.use_dem else None

        with rasterio.open(img_path) as img_ds, rasterio.open(dsm_path) as dsm_ds:
            dem_ds = rasterio.open(dem_path) if dem_path is not None else None

            # CRS 확인
            if img_ds.crs is None:
                if args.verbose:
                    print(f"[WARN] RGB has no CRS. Assume EPSG:4326: {img_path.name}")
                img_crs = CRS.from_epsg(4326)
            else:
                img_crs = img_ds.crs

            if dsm_ds.crs is None:
                if args.verbose:
                    print(f"[WARN] DSM has no CRS. Assume EPSG:4326: {dsm_path.name}")
                dsm_crs = CRS.from_epsg(4326)
            else:
                dsm_crs = dsm_ds.crs

            if dem_ds is not None and dem_ds.crs is None:
                if args.verbose:
                    print(f"[WARN] DEM has no CRS. Assume EPSG:4326: {dem_path}")
                dem_crs = CRS.from_epsg(4326)
            elif dem_ds is not None:
                dem_crs = dem_ds.crs
            else:
                dem_crs = None

            img_src_gsd = dataset_gsd(img_ds)
            scale = compute_scale(img_src_gsd, args.tgt_gsd)
            if scale <= 0:
                raise RuntimeError("Invalid scale computed.")

            dsm_nodata = args.dsm_nodata if args.dsm_nodata is not None else dsm_ds.nodata
            dem_nodata = args.dem_nodata if args.dem_nodata is not None else (dem_ds.nodata if dem_ds else None)

            # DSM/DEM을 RGB 그리드에 정렬
            dsm_vrt = WarpedVRT(
                dsm_ds,
                crs=img_crs,
                transform=img_ds.transform,
                width=img_ds.width,
                height=img_ds.height,
                resampling=Resampling.bilinear,
                nodata=dsm_nodata,
            )

            dem_vrt = None
            if dem_ds is not None:
                dem_vrt = WarpedVRT(
                    dem_ds,
                    crs=img_crs,
                    transform=img_ds.transform,
                    width=img_ds.width,
                    height=img_ds.height,
                    resampling=Resampling.bilinear,
                    nodata=dem_nodata,
                )

            H, W = img_ds.height, img_ds.width

            if args.verbose:
                print(f"\n[TILE] {tile_id}")
                print(f"  RGB: {img_path.name} ({W}x{H}) gsd={img_src_gsd:.4f}m")
                print(f"  DSM: {dsm_path.name}")
                if dem_path:
                    print(f"  DEM: {Path(dem_path).name}")

            for y0 in range(0, H, args.stride_read):
                for x0 in range(0, W, args.stride_read):
                    win_w = min(args.tile_read, W - x0)
                    win_h = min(args.tile_read, H - y0)
                    if win_w <= 0 or win_h <= 0:
                        continue

                    out_w = max(1, int(round(win_w * scale)))
                    out_h = max(1, int(round(win_h * scale)))

                    if out_w < args.patch or out_h < args.patch:
                        continue

                    win = Window(x0, y0, win_w, win_h)

                    # ===== block_key (균형 샘플링 단위) =====
                    block_key = f"{tile_id}_sx{x0}_sy{y0}"

                    # RGB 읽기
                    img_bands = img_ds.count
                    use_bands = [1, 2, 3] if img_bands >= 3 else list(range(1, img_bands + 1))

                    rgb_tile = read_resampled_window(
                        img_ds, win, out_h, out_w,
                        resampling=Resampling.bilinear,
                        indexes=use_bands,
                        boundless=True, fill_value=0
                    ).astype(np.float32)

                    # uint8로 변환
                    src_dtype = img_ds.dtypes[0]
                    mx = float(np.nanmax(rgb_tile)) if rgb_tile.size > 0 else 0.0

                    if "16" in str(src_dtype):
                        rgb_tile = (rgb_tile / 65535.0) * 255.0
                    elif "float" in str(src_dtype):
                        if mx <= 1.5:
                            rgb_tile = rgb_tile * 255.0
                    elif mx > 255:
                        rgb_tile = (rgb_tile / mx) * 255.0

                    rgb_tile = np.clip(rgb_tile, 0, 255).astype(np.uint8)

                    # DSM 읽기
                    dsm_tile = read_resampled_window(
                        dsm_vrt, win, out_h, out_w,
                        resampling=Resampling.bilinear,
                        indexes=[1],
                        boundless=False,
                        fill_value=float(dsm_nodata) if dsm_nodata is not None else 0.0
                    )[0].astype(np.float32)

                    # DEM 읽기
                    dem_tile = None
                    if dem_vrt is not None:
                        dem_tile = read_resampled_window(
                            dem_vrt, win, out_h, out_w,
                            resampling=Resampling.bilinear,
                            indexes=[1],
                            boundless=False,
                            fill_value=float(dem_nodata) if dem_nodata is not None else 0.0
                        )[0].astype(np.float32)

                    # 유효 마스크
                    valid = np.isfinite(dsm_tile)
                    if dsm_nodata is not None:
                        valid &= (dsm_tile != float(dsm_nodata))

                    if dem_tile is not None:
                        valid_dem = np.isfinite(dem_tile)
                        if dem_nodata is not None:
                            valid_dem &= (dem_tile != float(dem_nodata))
                        valid &= valid_dem

                    # 패치 추출
                    for py in range(0, out_h - args.patch + 1, args.stride_patch):
                        for px in range(0, out_w - args.patch + 1, args.stride_patch):
                            total_candidates += 1

                            # ===== 블록당 캡 체크 =====
                            if args.max_patches_per_block and args.max_patches_per_block > 0:
                                c = block_counts.get(block_key, 0)
                                if c >= args.max_patches_per_block:
                                    continue

                            dsm_patch = dsm_tile[py:py + args.patch, px:px + args.patch]
                            m_patch = valid[py:py + args.patch, px:px + args.patch]
                            vr = float(m_patch.mean())
                            if vr < args.min_valid_ratio:
                                continue

                            rgb_patch = rgb_tile[:, py:py + args.patch, px:px + args.patch]
                            if int(rgb_patch.mean()) < 1 and int(rgb_patch.std()) < 1:
                                continue

                            meta = {
                                "city": city,
                                "tile_id": tile_id,
                                "block_key": block_key,
                                "img_path": str(img_path),
                                "dsm_path": str(dsm_path),
                                "dem_path": str(dem_path) if dem_path else None,
                                "src_gsd": img_src_gsd,
                                "tgt_gsd": args.tgt_gsd,
                                "tile_read": args.tile_read,
                                "stride_read": args.stride_read,
                                "patch": args.patch,
                                "stride_patch": args.stride_patch,
                                "x0_src": int(x0),
                                "y0_src": int(y0),
                                "win_w_src": int(win_w),
                                "win_h_src": int(win_h),
                                "px_tgt": int(px),
                                "py_tgt": int(py),
                                "valid_ratio": vr,
                                "img_crs_wkt": safe_wkt(img_crs),
                                "dsm_crs_wkt": safe_wkt(dsm_crs),
                                "dem_crs_wkt": safe_wkt(dem_crs) if dem_ds else None,
                            }

                            uid = total_kept
                            base = f"{uid:08d}_{city}_{tile_id}_sx{x0}_sy{y0}_tx{px}_ty{py}.npz"
                            out_path = tmp_dir / base

                            save_kwargs = dict(
                                rgb=rgb_patch,
                                dsm=dsm_patch,
                                mask=m_patch.astype(np.uint8),
                                meta=json.dumps(meta)
                            )

                            if dem_tile is not None:
                                dem_patch = dem_tile[py:py + args.patch, px:px + args.patch]
                                save_kwargs["dem"] = dem_patch

                            np.savez_compressed(out_path, **save_kwargs)

                            kept_items.append((out_path, tile_id, block_key))
                            total_kept += 1

                            # 캡 카운트 증가
                            if args.max_patches_per_block and args.max_patches_per_block > 0:
                                block_counts[block_key] = block_counts.get(block_key, 0) + 1

                            if args.verbose and total_kept % 100 == 0:
                                print(f"  [{total_kept}] patches created...")

                            if total_kept >= args.max_patches:
                                break
                        if total_kept >= args.max_patches:
                            break
                    if total_kept >= args.max_patches:
                        break
                if total_kept >= args.max_patches:
                    break

            dsm_vrt.close()
            if dem_vrt is not None:
                dem_vrt.close()
            if dem_ds is not None:
                dem_ds.close()

        if total_kept >= args.max_patches:
            break

    if not kept_items:
        print(f"Done. kept 0/{total_candidates} patches")
        print("Saved: train=0 val=0")
        return

    # ===== 그룹 단위 split (block 기본) =====
    train_list, val_list = group_split(
        kept_items,
        train_ratio=args.train_ratio,
        seed=args.seed,
        split_by=args.split_by
    )

    # move to train/val
    for p in train_list:
        p.rename(train_dir / p.name)
    for p in val_list:
        p.rename(val_dir / p.name)

    print(f"\nDone. kept {len(kept_items)}/{total_candidates} patches")
    print(f"Saved: train={len(train_list)} val={len(val_list)}")
    print(f"[SPLIT] split_by={args.split_by}  train_ratio={args.train_ratio}")
    if args.max_patches_per_block and args.max_patches_per_block > 0:
        print(f"[BALANCE] max_patches_per_block={args.max_patches_per_block}  blocks={len(block_counts)}")

    if args.visualize and len(kept_items) > 0:
        print("\n" + "=" * 70)
        print("Creating Visualizations")
        print("=" * 70)

        if len(train_list) > 0:
            visualize_samples(out_dir, city, split="train", n_samples=args.n_viz_samples)
        if len(val_list) > 0:
            visualize_samples(out_dir, city, split="val", n_samples=args.n_viz_samples)

        print("\n" + "=" * 70)
        print("✓ Visualization Complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
