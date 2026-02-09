from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry,
    QgsCoordinateReferenceSystem, QgsWkbTypes,
    QgsMapSettings, QgsMapRendererSequentialJob, QgsRectangle,
    QgsCoordinateTransform
)
from PyQt5.QtCore import QSize
import processing
import os

# =========================
# 0) 사용자 설정
# =========================

# ⚠️ 중요: 레이어 패널의 정확한 이름을 입력하세요
GRID_NAME = "TN_MAPINDX_5K"        # 그리드 레이어 (EPSG:5179)
DEM_NAME  = "NASADEM_HGT_n35e127"   # DEM 레이어
DSM_NAME  = "ALPSMLC30_N035E127_DSM"  # DSM 레이어
VW_NAME   = "VWorld Satellite"      # VWorld 위성영상

OUT_DIR = r"C:/Users/user/workspace/output_tiles_4326"
TILE_ID_FIELD = "MAPIDCD_NO"
TARGET_EPSG = "EPSG:4326"  # 출력 좌표계

# VWorld PNG 가로 픽셀(종횡비 유지)
VW_WIDTH_PX = 1024
VW_DPI = 96

# 최대 타일 개수
MAX_TILES = 500  # 테스트용

# =========================
# 1) 레이어 가져오기
# =========================
proj = QgsProject.instance()

def get_layer_by_name(name, required=True):
    layers = proj.mapLayersByName(name)
    if not layers:
        if required:
            print(f"❌ 레이어를 찾을 수 없음: {name}")
            print("📋 사용 가능한 레이어:")
            for layer in proj.mapLayers().values():
                print(f"   - {layer.name()} (CRS: {layer.crs().authid()})")
            raise RuntimeError(f"레이어를 찾을 수 없음: {name}")
        return None
    layer = layers[0]
    print(f"✅ 레이어 로드: {name} (CRS: {layer.crs().authid()})")
    return layer

print("\n" + "="*60)
print("🚀 래스터 타일 자르기 시작")
print("="*60)

print("\n📋 프로젝트의 모든 래스터 레이어:")
for layer in proj.mapLayers().values():
    if hasattr(layer, 'extent'):  # 래스터 레이어 확인
        ext = layer.extent()
        print(f"   - {layer.name()}")
        print(f"     CRS: {layer.crs().authid()}")
        print(f"     범위: ({ext.xMinimum():.2f}, {ext.yMinimum():.2f}) - ({ext.xMaximum():.2f}, {ext.yMaximum():.2f})")

print("\n" + "="*60)

grid = get_layer_by_name(GRID_NAME)
dem  = get_layer_by_name(DEM_NAME, required=False)
dsm  = get_layer_by_name(DSM_NAME, required=False)
vw   = get_layer_by_name(VW_NAME, required=False)

if grid.geometryType() != QgsWkbTypes.PolygonGeometry:
    raise RuntimeError("그리드 레이어는 폴리곤 타입이어야 합니다.")

# 출력 디렉토리 생성
os.makedirs(OUT_DIR, exist_ok=True)
dsm_dir = os.path.join(OUT_DIR, "DSM")
dem_dir = os.path.join(OUT_DIR, "DEM")
rgb_dir = os.path.join(OUT_DIR, "RGB")
os.makedirs(dsm_dir, exist_ok=True)
os.makedirs(dem_dir, exist_ok=True)
os.makedirs(rgb_dir, exist_ok=True)

target_crs = QgsCoordinateReferenceSystem(TARGET_EPSG)
target_wkt = target_crs.toWkt()

print(f"\n📁 출력 디렉토리: {OUT_DIR}")
print(f"🎯 출력 좌표계: {TARGET_EPSG}")

# 레이어 범위 출력
print("\n📏 레이어 범위 정보:")
if dsm:
    dsm_ext = dsm.extent()
    dsm_crs = dsm.crs()
    print(f"   DSM: {dsm_crs.authid()}")
    print(f"        범위: ({dsm_ext.xMinimum():.6f}, {dsm_ext.yMinimum():.6f}) - ({dsm_ext.xMaximum():.6f}, {dsm_ext.yMaximum():.6f})")
    
if dem:
    dem_ext = dem.extent()
    dem_crs = dem.crs()
    print(f"   DEM: {dem_crs.authid()}")
    print(f"        범위: ({dem_ext.xMinimum():.6f}, {dem_ext.yMinimum():.6f}) - ({dem_ext.xMaximum():.6f}, {dem_ext.yMaximum():.6f})")

# =========================
# 2) DEM/DSM: Warp+Clip (TARGET_EPSG로 저장)
# =========================
def warp_clip_to_target(in_raster, extent_target, out_path, resampling=1, nodata=None):
    """래스터를 범위로 자르고 목표 좌표계로 변환"""
    
    # 범위를 문자열로 변환
    extent_str = f"{extent_target.xMinimum()},{extent_target.xMaximum()},{extent_target.yMinimum()},{extent_target.yMaximum()}"
    
    params = {
        "INPUT": in_raster,
        "SOURCE_CRS": None,
        "TARGET_CRS": target_crs,
        "RESAMPLING": resampling,   # 0=Nearest, 1=Bilinear
        "NODATA": nodata,
        "TARGET_RESOLUTION": None,
        "OPTIONS": "",
        "DATA_TYPE": 0,
        "TARGET_EXTENT": extent_str,
        "TARGET_EXTENT_CRS": target_crs,
        "MULTITHREADING": False,
        "EXTRA": "",
        "OUTPUT": out_path
    }
    
    try:
        result = processing.run("gdal:warpreproject", params)
        if result and result["OUTPUT"] and os.path.exists(result["OUTPUT"]):
            file_size = os.path.getsize(result["OUTPUT"]) / 1024
            print(f"      ✓ {os.path.basename(out_path)} ({file_size:.1f} KB)")
            return result["OUTPUT"]
        else:
            print(f"      ✗ {os.path.basename(out_path)} 생성 실패")
            return None
    except Exception as e:
        print(f"      ✗ 오류: {e}")
        return None

# =========================
# 3) Vworld: 렌더링 TIF + TFW + PRJ
# =========================
def write_tfw(tfw_path, pixel_size_x, pixel_size_y, x_center_ul, y_center_ul):
    """World file 생성"""
    A = pixel_size_x
    D = 0.0
    B = 0.0
    E = -abs(pixel_size_y)
    C = x_center_ul
    F = y_center_ul
    with open(tfw_path, "w", encoding="utf-8") as f:
        f.write(f"{A}\n{D}\n{B}\n{E}\n{C}\n{F}\n")

def render_vworld_tile_tif(vw_layer, extent_target: QgsRectangle, out_tif, width_px=1024, dpi=96):
    """VWorld를 GeoTIFF로 렌더링"""
    w = extent_target.width()
    h = extent_target.height()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"유효하지 않은 범위: {extent_target.toString()}")

    height_px = max(1, int(round(width_px * (h / w))))

    ms = QgsMapSettings()
    ms.setLayers([vw_layer])
    ms.setDestinationCrs(target_crs)
    ms.setExtent(extent_target)
    ms.setOutputSize(QSize(width_px, height_px))
    ms.setOutputDpi(dpi)

    job = QgsMapRendererSequentialJob(ms)
    job.start()
    job.waitForFinished()
    img = job.renderedImage()

    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    
    # 임시 TIF로 저장
    temp_tif = out_tif.replace(".tif", "_temp.tif")
    ok = img.save(temp_tif, "TIFF")
    if not ok:
        raise RuntimeError(f"TIF 저장 실패: {temp_tif}")

    # GDAL로 GeoTIFF 변환 (좌표계 내장)
    extent_str = f"{extent_target.xMinimum()},{extent_target.xMaximum()},{extent_target.yMinimum()},{extent_target.yMaximum()}"
    
    params = {
        "INPUT": temp_tif,
        "TARGET_CRS": target_crs,
        "TARGET_EXTENT": extent_str,
        "TARGET_EXTENT_CRS": target_crs,
        "NODATA": None,
        "COPY_SUBDATASETS": False,
        "OPTIONS": "",
        "EXTRA": "",
        "DATA_TYPE": 0,
        "OUTPUT": out_tif
    }
    
    try:
        processing.run("gdal:translate", params)
        # 임시 파일 삭제
        if os.path.exists(temp_tif):
            os.remove(temp_tif)
    except Exception as e:
        # 임시 파일을 최종 파일로 이동
        if os.path.exists(temp_tif):
            os.rename(temp_tif, out_tif)
        print(f"      ⚠ GeoTIFF 변환 실패, 일반 TIF로 저장: {e}")
    
    # TFW (World file) 생성
    px_x = extent_target.width() / width_px
    px_y = extent_target.height() / height_px
    x_center_ul = extent_target.xMinimum() + px_x / 2.0
    y_center_ul = extent_target.yMaximum() - px_y / 2.0

    tfw_path = os.path.splitext(out_tif)[0] + ".tfw"
    write_tfw(tfw_path, px_x, px_y, x_center_ul, y_center_ul)

    # PRJ 파일 생성
    prj_path = os.path.splitext(out_tif)[0] + ".prj"
    with open(prj_path, "w", encoding="utf-8") as f:
        f.write(target_wkt)
    
    file_size = os.path.getsize(out_tif) / 1024
    print(f"      ✓ {os.path.basename(out_tif)} ({file_size:.1f} KB, {width_px}x{height_px})")

# =========================
# 4) 타일별 처리
# =========================
# 선택 타일이 있으면 선택만, 없으면 전체
features = list(grid.selectedFeatures())
if features:
    print(f"\n📌 선택된 {len(features)}개 타일 처리")
else:
    print(f"\n🔍 DSM/DEM 범위 내의 타일 자동 선택 중...")
    
    # DSM/DEM 범위 확인
    dsm_extent_4326 = None
    if dsm:
        dsm_extent_4326 = dsm.extent()
    elif dem:
        dsm_extent_4326 = dem.extent()
    
    if dsm_extent_4326:
        # 5179 → 4326 변환기
        ct_5179_to_4326 = QgsCoordinateTransform(
            grid.crs(),
            QgsCoordinateReferenceSystem("EPSG:4326"),
            QgsProject.instance()
        )
        
        # 범위 내 타일 필터링
        filtered_features = []
        for feat in grid.getFeatures():
            geom = QgsGeometry(feat.geometry())
            geom.transform(ct_5179_to_4326)
            tile_extent = geom.boundingBox()
            
            # DSM 범위 내에 완전히 포함되는지 확인 (경계선 제외)
            if (tile_extent.xMinimum() >= dsm_extent_4326.xMinimum() and
                tile_extent.xMaximum() <= dsm_extent_4326.xMaximum() and
                tile_extent.yMinimum() >= dsm_extent_4326.yMinimum() and
                tile_extent.yMaximum() <= dsm_extent_4326.yMaximum()):
                filtered_features.append(feat)
                if len(filtered_features) >= MAX_TILES:
                    break
        
        features = filtered_features
        print(f"✅ DSM/DEM 범위와 겹치는 {len(features)}개 타일 발견")
    else:
        features = list(grid.getFeatures())[:MAX_TILES]
        print(f"⚠️ DSM/DEM 없음. 처음 {len(features)}개 타일 처리")
    
    if len(features) == 0:
        print("❌ 처리할 타일이 없습니다.")
        print("   DSM/DEM 범위와 그리드 타일이 겹치지 않습니다.")
        raise SystemExit(0)

field_names = [f.name() for f in grid.fields()]
if TILE_ID_FIELD not in field_names:
    raise RuntimeError(f"타일 필드 '{TILE_ID_FIELD}' 없음. 현재 필드: {field_names}")

# 타일 geom -> target CRS 변환기
ct = QgsCoordinateTransform(grid.crs(), target_crs, QgsProject.instance())

print("="*60)

stats = {'dsm': 0, 'dem': 0, 'vw': 0, 'total': len(features)}

for i, feat in enumerate(features, 1):
    tile_id = str(feat[TILE_ID_FIELD]).strip() or str(feat.id())
    tile_id_safe = tile_id.replace(" ", "_").replace("/", "_")

    print(f"\n[{i}/{len(features)}] 🔹 타일: {tile_id_safe}")

    # 타일 geometry를 target CRS로 변환
    geom_target = QgsGeometry(feat.geometry())
    geom_target.transform(ct)
    rect_target = geom_target.boundingBox()
    
    print(f"   범위: ({rect_target.xMinimum():.6f}, {rect_target.yMinimum():.6f}) - ({rect_target.xMaximum():.6f}, {rect_target.yMaximum():.6f})")

    # DSM 처리 (범위 사용) - 완전히 포함되는지 체크
    if dsm:
        dsm_extent = dsm.extent()
        # 타일이 DSM 범위 내에 완전히 포함되는지 확인
        is_inside = (rect_target.xMinimum() >= dsm_extent.xMinimum() and
                     rect_target.xMaximum() <= dsm_extent.xMaximum() and
                     rect_target.yMinimum() >= dsm_extent.yMinimum() and
                     rect_target.yMaximum() <= dsm_extent.yMaximum())
        
        if is_inside:
            out_dsm = os.path.join(dsm_dir, f"{tile_id_safe}.tif")
            if warp_clip_to_target(dsm, rect_target, out_dsm, resampling=1, nodata=-9999):
                stats['dsm'] += 1
        else:
            print(f"      ⊘ DSM: 타일이 데이터 범위 밖")

    # DEM 처리 (범위 사용) - 완전히 포함되는지 체크
    if dem:
        dem_extent = dem.extent()
        # 타일이 DEM 범위 내에 완전히 포함되는지 확인
        is_inside = (rect_target.xMinimum() >= dem_extent.xMinimum() and
                     rect_target.xMaximum() <= dem_extent.xMaximum() and
                     rect_target.yMinimum() >= dem_extent.yMinimum() and
                     rect_target.yMaximum() <= dem_extent.yMaximum())
        
        if is_inside:
            out_dem = os.path.join(dem_dir, f"{tile_id_safe}.tif")
            if warp_clip_to_target(dem, rect_target, out_dem, resampling=1, nodata=-9999):
                stats['dem'] += 1
        else:
            print(f"      ⊘ DEM: 타일이 데이터 범위 밖")

    # VWorld 처리
    if vw:
        out_vw = os.path.join(rgb_dir, f"{tile_id_safe}.tif")
        try:
            render_vworld_tile_tif(vw, rect_target, out_vw, width_px=VW_WIDTH_PX, dpi=VW_DPI)
            stats['vw'] += 1
        except Exception as e:
            print(f"      ✗ VWorld 렌더링 실패: {e}")

print("\n" + "="*60)
print("✅ 처리 완료!")
print("="*60)
print(f"총 타일: {stats['total']}개")
print(f"DSM 성공: {stats['dsm']}개")
print(f"DEM 성공: {stats['dem']}개")
print(f"VWorld 성공: {stats['vw']}개")
print(f"\n📁 출력: {OUT_DIR}")
print(f"   - DSM: {dsm_dir}")
print(f"   - DEM: {dem_dir}")
print(f"   - RGB: {rgb_dir}")
print("="*60)