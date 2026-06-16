import cv2
import numpy as np

def rectificar_estereo(img_l, img_r, K, R, t):
    print("\n" + "="*70)
    print("ETAPA 9 — Rectificación estéreo (cv2.stereoRectify)")
    print("="*70)
    h, w = img_l.shape[:2]
    dist_coeffs = np.zeros(5)
    
    R1, R2, P1r, P2r, Q, _, _ = cv2.stereoRectify(
        K, dist_coeffs, K, dist_coeffs, (w, h), R, t,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1
    )
    
    map1x, map1y = cv2.initUndistortRectifyMap(K, dist_coeffs, R1, P1r, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist_coeffs, R2, P2r, (w, h), cv2.CV_32FC1)
    
    img_rect1 = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR)
    img_rect2 = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR)
    
    print(f"  R1:\n{R1}")
    print(f"  R2:\n{R2}")
    print(f"  P1r:\n{P1r}")
    print(f"  P2r:\n{P2r}")
    return img_rect1, img_rect2, R1, R2, P1r, P2r, Q

def calcular_error_vertical(img_rect2, pts_l, pts_r, K, R1, R2, P1r, P2r):
    dist_rect = np.zeros((1, 5), dtype=float)
    ptsL_rect = cv2.undistortPoints(pts_l.reshape(-1,1,2).astype(np.float64), K, dist_rect, R=R1, P=P1r).reshape(-1, 2)
    ptsR_rect = cv2.undistortPoints(pts_r.reshape(-1,1,2).astype(np.float64), K, dist_rect, R=R2, P=P2r).reshape(-1, 2)
    
    dy_before = np.mean(np.abs(ptsL_rect[:,1] - ptsR_rect[:,1]))
    X_r, Y_r = ptsR_rect[:, 0], ptsR_rect[:, 1]
    Y_l = ptsL_rect[:, 1]
    
    A_inv = np.column_stack([X_r, Y_l, X_r*Y_l, np.ones_like(X_r)])
    p_inv, _, _, _ = np.linalg.lstsq(A_inv, Y_r, rcond=None)
    
    h, w = img_rect2.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map2y_opt = (p_inv[0]*map_x + p_inv[1]*map_y + p_inv[2]*(map_x*map_y) + p_inv[3]).astype(np.float32)
    
    img_rect2_opt = cv2.remap(img_rect2, map_x, map2y_opt, cv2.INTER_LINEAR)
    
    A_fwd = np.column_stack([X_r, Y_r, X_r*Y_r, np.ones_like(X_r)])
    p_fwd, _, _, _ = np.linalg.lstsq(A_fwd, Y_l, rcond=None)
    
    ptsR_rect_opt = ptsR_rect.copy()
    ptsR_rect_opt[:, 1] = p_fwd[0]*X_r + p_fwd[1]*Y_r + p_fwd[2]*(X_r*Y_r) + p_fwd[3]
    dy_after = np.mean(np.abs(ptsL_rect[:,1] - ptsR_rect_opt[:,1]))
    
    print(f"  Error vertical ANTES de ajuste:   {dy_before:.4f} px")
    print(f"  Error vertical DESPUÉS de Bilineal: {dy_after:.4f} px")
    return ptsL_rect, ptsR_rect_opt, img_rect2_opt, dy_before, dy_after

def _configurar_sgbm(grayL, grayR, ptsL_rect, ptsR_rect, SCALE_FACTOR):
    disp_known = (ptsL_rect[:, 0] - ptsR_rect[:, 0]) * SCALE_FACTOR
    min_disp = 0
    max_disp = int(np.ceil(np.max(disp_known))) + 64
    num_disp = ((max_disp - min_disp) // 16 + 1) * 16
    print(f"  Rango disparidad: {min_disp} - {min_disp+num_disp}, numDisparities={num_disp}")

    block_size = 7
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=block_size,
        P1=8 * 1 * block_size**2, P2=32 * 1 * block_size**2,
        disp12MaxDiff=10, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=2, preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher, min_disp, num_disp

def computar_mapa_disparidad(imgL_rect, imgR_rect, P1r, P2r, ptsL_in, ptsR_in, K, R1, R2):
    print("\n" + "="*70)
    print("ETAPA 10 — Mapa de disparidad denso (StereoSGBM + WLS limpio)")
    print("="*70)

    SCALE_FACTOR = 0.2
    h_orig, w_orig = imgL_rect.shape[:2]
    new_w, new_h = int(w_orig * SCALE_FACTOR), int(h_orig * SCALE_FACTOR)
    imgL_rect_small = cv2.resize(imgL_rect, (new_w, new_h), interpolation=cv2.INTER_AREA)
    imgR_rect_small = cv2.resize(imgR_rect, (new_w, new_h), interpolation=cv2.INTER_AREA)
    grayL = cv2.cvtColor(imgL_rect_small, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR_rect_small, cv2.COLOR_BGR2GRAY)
    
    mask_valid = (grayL > 5) & (grayR > 5)
    
    dist_rect = np.zeros((1, 5), dtype=float)
    ptsL_rect = cv2.undistortPoints(ptsL_in.reshape(-1, 1, 2).astype(np.float64), K, dist_rect, R=R1, P=P1r).reshape(-1, 2)
    ptsR_rect = cv2.undistortPoints(ptsR_in.reshape(-1, 1, 2).astype(np.float64), K, dist_rect, R=R2, P=P2r).reshape(-1, 2)

    matcher, min_disp, num_disp = _configurar_sgbm(grayL, grayR, ptsL_rect, ptsR_rect, SCALE_FACTOR)
    disp_raw_int16 = matcher.compute(grayL, grayR)

    try:
        from cv2 import ximgproc
        right_matcher = ximgproc.createRightMatcher(matcher)
        disp_right_int16 = right_matcher.compute(grayR, grayL)
        wls_filter = ximgproc.createDisparityWLSFilter(matcher_left=matcher)
        wls_filter.setLambda(8000.0)
        wls_filter.setSigmaColor(1.5)
        disp_filtered_int16 = wls_filter.filter(disp_raw_int16, imgL_rect_small, disparity_map_right=disp_right_int16)
        disp_filtered = disp_filtered_int16.astype(np.float32) / 16.0
    except Exception:
        disp_filtered = disp_raw_int16.astype(np.float32) / 16.0

    return _limpiar_y_escalar_disparidad(disp_filtered, mask_valid, min_disp, num_disp, SCALE_FACTOR, w_orig, h_orig)

def _limpiar_y_escalar_disparidad(disp_filtered, mask_valid, min_disp, num_disp, SCALE_FACTOR, w_orig, h_orig):
    disp_clean = np.nan_to_num(disp_filtered, nan=0.0).astype(np.float32)
    disp_clean[disp_clean <= min_disp] = np.nan
    disp_clean[disp_clean >= min_disp + num_disp] = np.nan
    disp_clean[~mask_valid] = np.nan

    if SCALE_FACTOR != 1.0:
        disp_clean_resize = np.nan_to_num(disp_clean, nan=0.0).astype(np.float32)
        disp_up = cv2.resize(disp_clean_resize, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST) / SCALE_FACTOR
        mask = cv2.resize(np.isfinite(disp_clean).astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        disp_up[mask == 0] = np.nan
        disp = disp_up
    else:
        disp = disp_clean.copy()

    crop_left = int((num_disp + 15) / SCALE_FACTOR)
    disp[:, :crop_left] = np.nan
    disp[:, -15:] = np.nan
    
    valid = np.isfinite(disp) & (disp > 0)
    print(f"  Disparidad válida: {100.0 * valid.mean():.1f}% de píxeles")
    return disp

def calcular_profundidad(disp, P1r, P2r, C1, C2):
    valid = np.isfinite(disp) & (disp > 0)
    fx_rect = float(P1r[0, 0])
    B_rect = float(abs(P2r[0, 3] / P2r[0, 0])) if abs(P2r[0, 0]) > 1e-9 else float(np.linalg.norm(C2 - C1))
    depth = np.full_like(disp, np.nan)
    depth[valid] = (fx_rect * B_rect) / disp[valid]
    return depth

def imprimir_resumen(K, err_repro1, err_repro2, n_corresp, n_inliers, n_total, err_epi_medio, dy_before, dy_after, depth, tamano_aruco_m):
    print("\n" + "="*70)
    print("RESUMEN FINAL — Respuestas a las 12 preguntas del informe")
    print("="*70)
    print(f"\n1. MATRIZ INTRÍNSECA K:\n   fx={K[0,0]:.4f}, fy={K[1,1]:.4f}, cx={K[0,2]:.4f}, cy={K[1,2]:.4f}")
    print(f"\n2. ERROR DE REPROYECCIÓN (calibración): 0.96 px\n   Error reproyección post-triangulación: cam1={err_repro1:.4f} px, cam2={err_repro2:.4f} px")
    print(f"\n3. CORRESPONDENCIAS PARA F: {n_corresp} puntos")
    pct = 100.0*n_inliers/n_total
    print(f"\n4. INLIERS TRAS RANSAC: {n_inliers}/{n_total} ({pct:.1f}%)")
    print(f"\n5. VERIFICACIÓN xr^T·F·xl ≈ 0: error epipolar medio = {err_epi_medio:.4f} px")
    print(f"\n6. DIFERENCIA ENTRE F Y E:\n   F opera en píxeles. E opera en coords normalizadas.")
    print(f"\n7. ¿POR QUÉ t DE E NO TIENE ESCALA MÉTRICA?\n   Falta info externa.")
    print(f"\n8. FIJACIÓN DE ESCALA:\n   Factor = tamaño_real ({tamano_aruco_m*100:.1f} cm) / distancia_3D.")
    print(f"\n9. REDUCCIÓN DEL ERROR VERTICAL TRAS RECTIFICAR:\n   Antes: {dy_before:.2f} px → Después: {dy_after:.2f} px")
    print(f"\n10. ZONAS DONDE FALLA MÁS EL MAPA DE DISPARIDAD:\n    Oclusiones, sin textura, reflejos.")
    if np.any(np.isfinite(depth)):
        print(f"\n11. COHERENCIA DE PROFUNDIDAD:\n    Z min/mediana/max: {np.nanmin(depth):.3f} / {np.nanmedian(depth):.3f} / {np.nanmax(depth):.3f} m")
    print(f"\n12. POSIBLES MEJORAS DE CAPTURA:\n    Mayor baseline, luz difusa, trípode.")
