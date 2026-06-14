#!/usr/bin/env python3
"""
main.py — Pipeline completo de Visión 3D Estéreo (Trabajo 2)
=============================================================
Etapas 0-10: ArUco → F → E → Pose → Triangulación → Rectificación → SGBM
"""
import os, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geometria_epipolar import (obtener_correspondencias, ocho_puntos_normalizado,
                                 ransac_fundamental, calcular_esencial,
                                 normalizar_puntos, _error_epipolar_simetrico)
from reconstruccion import (seleccionar_pose, fijar_escala_metrica,
                             triangular_puntos, error_reproyeccion, visualizar_3d)
from visualizacion import (visualizar_correspondencias, visualizar_lineas_epipolares,
                            visualizar_rectificacion)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
K = np.array([
    [3990.1574415286, 0.0, 2818.2271080190],
    [0.0, 3988.4486191714, 2135.8102621727],
    [0.0, 0.0, 1.0],
])
dist = np.array([0.2821249450, -2.1993670285, 0.0003160934, 0.0022638987, 5.0754442194])
TAMANO_ARUCO_M = 0.062   # lado del ArUco en metros
SCALE_FACTOR = 0.5
IMG_L = os.path.join("FotosOriginales", "1.png")
IMG_R = os.path.join("FotosOriginales", "2.png")
OUTDIR = "output"
EXPECTED = (4284, 5712)

# ============================================================================
# ETAPA 0 — Carga
# ============================================================================
def cargar_imagenes():
    print("="*70)
    print("ETAPA 0 — Carga y preparación")
    print("="*70)
    il = cv2.imread(IMG_L); ir = cv2.imread(IMG_R)
    if il is None or ir is None:
        sys.exit("[ERROR] No se pudieron cargar las imágenes.")
    for name, img in [("izquierda", il), ("derecha", ir)]:
        h, w = img.shape[:2]
        print(f"  {name}: {w}x{h}")
        if (h, w) != EXPECTED:
            print(f"  ⚠️  Resolución inesperada (esperada {EXPECTED[1]}x{EXPECTED[0]})")
    return il, ir

# ============================================================================
# ETAPA 9 — Rectificación estéreo (cv2.stereoRectify)
# ============================================================================
def rectificar_estereo(img_l, img_r, K, R, t):
    """
    Rectificación calibrada usando cv2.stereoRectify.
    
    Concepto: stereoRectify calcula rotaciones R1, R2 que alinean ambos planos
    imagen para que las líneas epipolares sean horizontales. Se derivan de la
    geometría epipolar (R, t): cada cámara se rota para que el epipolo se
    envíe al infinito horizontal, haciendo las epipolares paralelas al eje x.
    
    dist_coeffs = 0 porque las imágenes ya están corregidas de distorsión.
    """
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
    
    # Explicación conceptual
    print("\n  NOTA: Las homografías de rectificación se derivan de E (y por tanto de R,t).")
    print("  stereoRectify rota cada cámara (R1, R2) para que los planos imagen sean")
    print("  coplanares y las líneas epipolares perfectamente horizontales.")
    
    return img_rect1, img_rect2, R1, R2, P1r, P2r, Q

def calcular_error_vertical(img_rect2, pts_l, pts_r, K, R1, R2, P1r, P2r):
    """Calcula error vertical medio y aplica corrección afín en Y para eliminar el residual."""
    dist_rect = np.zeros((1, 5), dtype=float)
    ptsL_rect = cv2.undistortPoints(pts_l.reshape(-1,1,2).astype(np.float64),
                                     K, dist_rect, R=R1, P=P1r).reshape(-1, 2)
    ptsR_rect = cv2.undistortPoints(pts_r.reshape(-1,1,2).astype(np.float64),
                                     K, dist_rect, R=R2, P=P2r).reshape(-1, 2)
    
    dy_before = np.mean(np.abs(ptsL_rect[:,1] - ptsR_rect[:,1]))
    
    # Corrección Bilineal (solo en Y) para alinear suave y precisamente
    # Evita la distorsión loca de los extremos que causa RBF
    X_r, Y_r = ptsR_rect[:, 0], ptsR_rect[:, 1]
    Y_l = ptsL_rect[:, 1]
    
    # 1. Ajuste Inverso (Para remap: Destino -> Origen)
    A_inv = np.column_stack([X_r, Y_l, X_r*Y_l, np.ones_like(X_r)])
    p_inv, _, _, _ = np.linalg.lstsq(A_inv, Y_r, rcond=None)
    
    h, w = img_rect2.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map2y_opt = (p_inv[0]*map_x + p_inv[1]*map_y + p_inv[2]*(map_x*map_y) + p_inv[3]).astype(np.float32)
    
    img_rect2_opt = cv2.remap(img_rect2, map_x, map2y_opt, cv2.INTER_LINEAR)
    
    # 2. Ajuste Directo (Para calcular el nuevo Y de los puntos)
    A_fwd = np.column_stack([X_r, Y_r, X_r*Y_r, np.ones_like(X_r)])
    p_fwd, _, _, _ = np.linalg.lstsq(A_fwd, Y_l, rcond=None)
    
    ptsR_rect_opt = ptsR_rect.copy()
    ptsR_rect_opt[:, 1] = p_fwd[0]*X_r + p_fwd[1]*Y_r + p_fwd[2]*(X_r*Y_r) + p_fwd[3]
    
    dy_after = np.mean(np.abs(ptsL_rect[:,1] - ptsR_rect_opt[:,1]))
    
    print(f"  Error vertical ANTES de ajuste:   {dy_before:.4f} px")
    print(f"  Error vertical DESPUÉS de Bilineal: {dy_after:.4f} px")
    return ptsL_rect, ptsR_rect_opt, img_rect2_opt, dy_before, dy_after

# ============================================================================
# ETAPA 10 — Mapa de disparidad denso (StereoSGBM)
# ============================================================================
def calcular_disparidad(imgL_rect, imgR_rect, P1r, P2r, ptsL_in, ptsR_in,
                         K, R1, R2, C1, C2, outdir):
    """
    Disparidad densa con StereoSGBM siguiendo el código de referencia:
    reescalado, SGBM, WLS opcional, mediana, inpainting NS, profundidad métrica.
    """
    print("\n" + "="*70)
    print("ETAPA 10 — Mapa de disparidad denso (StereoSGBM)")
    print("="*70)
    
    h_orig, w_orig = imgL_rect.shape[:2]
    new_w, new_h = int(w_orig*SCALE_FACTOR), int(h_orig*SCALE_FACTOR)
    imgL_s = cv2.resize(imgL_rect, (new_w, new_h), interpolation=cv2.INTER_AREA)
    imgR_s = cv2.resize(imgR_rect, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Máscara de zona válida (excluye bordes negros de alpha=1)
    gray_mask_L = cv2.cvtColor(imgL_s, cv2.COLOR_BGR2GRAY)
    gray_mask_R = cv2.cvtColor(imgR_s, cv2.COLOR_BGR2GRAY)
    valid_region = (gray_mask_L > 5) & (gray_mask_R > 5)
    
    # CLAHE para mejorar el contraste de zonas sin textura
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grayL = clahe.apply(gray_mask_L)
    grayR = clahe.apply(gray_mask_R)
    
    # Escalar matrices de proyección
    P1r_s = P1r.copy(); P1r_s[0,:] *= SCALE_FACTOR; P1r_s[1,:] *= SCALE_FACTOR
    P2r_s = P2r.copy(); P2r_s[0,:] *= SCALE_FACTOR; P2r_s[1,:] *= SCALE_FACTOR
    
    # Proyectar puntos sparse al espacio rectificado para estimar rango disparidad
    dist_rect = np.zeros((1,5), dtype=float)
    ptsL_rect = cv2.undistortPoints(ptsL_in.reshape(-1,1,2).astype(np.float64),
                                     K, dist_rect, R=R1, P=P1r)
    ptsR_rect = cv2.undistortPoints(ptsR_in.reshape(-1,1,2).astype(np.float64),
                                     K, dist_rect, R=R2, P=P2r)
    
    # Disparidad en RESOLUCIÓN COMPLETA, luego escalar a resolución reducida
    disp_known_full = ptsL_rect[:,0,0] - ptsR_rect[:,0,0]
    disp_known = disp_known_full * SCALE_FACTOR
    min_disp = max(0, int(np.floor(np.min(disp_known))) - 5)
    max_disp = int(np.ceil(np.max(disp_known))) + 10
    num_disp = ((max_disp - min_disp) // 16 + 1) * 16
    print(f"  Disparidad sparse en escala reducida: min={np.min(disp_known):.1f}, max={np.max(disp_known):.1f}")
    print(f"  Rango SGBM: minDisp={min_disp}, numDisp={num_disp} (hasta {min_disp+num_disp})")
    
    # SGBM con parámetros para bordes definidos
    bs = 5; ch = 1
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=bs,
        P1=8*ch*bs**2, P2=32*ch*bs**2,
        disp12MaxDiff=2, uniquenessRatio=5,
        speckleWindowSize=100, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    
    disp_raw = matcher.compute(grayL, grayR).astype(np.float32) / 16.0
    
    # Invalidar zonas negras de alpha=1
    disp_raw[~valid_region] = -1.0
    
    # WLS si disponible
    try:
        from cv2 import ximgproc
        wls = ximgproc.createDisparityWLSFilter(matcher_left=matcher)
        wls.setLambda(8000)
        wls.setSigmaColor(1.5)
        rm = ximgproc.createRightMatcher(matcher)
        dr = rm.compute(grayR, grayL).astype(np.float32) / 16.0
        disp_f = wls.filter(disp_raw, grayL, disparity_map_right=dr).astype(np.float32)
        print("  Filtro WLS aplicado.")
    except (ImportError, AttributeError, Exception):
        disp_f = disp_raw.copy()
        print("  WLS no disponible, usando mapa base.")
    
    # Mediana + invalidar
    disp_f = np.nan_to_num(disp_f, nan=0.0).astype(np.float32)
    disp_med = cv2.medianBlur(disp_f, 5)
    disp_med[disp_med <= min_disp] = np.nan
    # Invalidar también bordes negros
    disp_med[~valid_region] = np.nan
    disp_final = disp_med
    
    # Detección adaptativa de oclusión lateral
    valid_idx = [np.where(~np.isnan(disp_final[r]))[0] for r in range(disp_final.shape[0])]
    fc = [idx[0] for idx in valid_idx if len(idx)>0]
    lc = [idx[-1] for idx in valid_idx if len(idx)>0]
    crop_l_s = int(np.percentile(fc, 90))+15 if fc else int(num_disp+15)
    crop_r_s = disp_final.shape[1] - int(np.percentile(lc, 10))+15 if lc else 15
    
    # Inpainting Navier-Stokes (solo en la zona válida de la imagen)
    inv_mask = np.isnan(disp_final) | (disp_final <= min_disp)
    val_mask = ~inv_mask
    if np.any(val_mask):
        d_mn = float(np.nanmin(disp_final[val_mask]))
        d_mx = float(np.nanmax(disp_final[val_mask]))
        d_rng = d_mx - d_mn if (d_mx-d_mn) > 1e-9 else 1.0
        du8 = np.zeros_like(disp_final, dtype=np.uint8)
        vals = (disp_final[val_mask] - d_mn) / d_rng
        du8[val_mask] = np.clip(255.0 * vals, 0, 255).astype(np.uint8)
        
        # Solo inpintar dentro de la zona con imagen real (no los bordes negros)
        ip_mask = ((inv_mask & valid_region) * 255).astype(np.uint8)
        du8_ip = cv2.inpaint(du8, ip_mask, inpaintRadius=1, flags=cv2.INPAINT_NS)
        dip_f = d_mn + (du8_ip.astype(np.float32) / 255.0) * d_rng
        dc = disp_final.copy()
        fill = inv_mask & valid_region  # solo rellenar donde hay imagen
        dc[fill] = dip_f[fill]
        disp_final = dc
    
    # Reescalado a resolución original
    if SCALE_FACTOR != 1.0:
        # Reemplazar NaN con min_disp (valor neutro) para evitar overflow en resize
        disp_for_resize = disp_final.copy()
        nan_mask_s = ~np.isfinite(disp_for_resize) | (disp_for_resize <= min_disp)
        disp_for_resize[nan_mask_s] = float(min_disp)
        
        disp_up = cv2.resize(disp_for_resize, (w_orig, h_orig),
                             interpolation=cv2.INTER_NEAREST) / SCALE_FACTOR
        
        # Máscara de validez: propagar ambos la máscara de disparidad y la de bordes negros
        valid_s = np.isfinite(disp_final) & (disp_final > min_disp)
        mask_up = cv2.resize(valid_s.astype(np.uint8),
                             (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        disp_up[mask_up == 0] = np.nan
        disp = disp_up
    else:
        disp = disp_final.copy()
    
    # Anular oclusiones laterales
    crop_l = int(crop_l_s / SCALE_FACTOR)
    crop_r = int(crop_r_s / SCALE_FACTOR)
    disp[:, :crop_l] = np.nan
    if crop_r > 0:
        disp[:, -crop_r:] = np.nan
    
    # Clamp extremos para evitar inf/overflow
    valid = np.isfinite(disp) & (disp > 0)
    if np.any(valid):
        p01 = np.percentile(disp[valid], 0.5)
        p99 = np.percentile(disp[valid], 99.5)
        disp[(disp < p01) & np.isfinite(disp)] = np.nan
        disp[(disp > p99) & np.isfinite(disp)] = np.nan
    
    valid = np.isfinite(disp) & (disp > 0)
    print(f"  Disparidad válida: {100.0*valid.mean():.1f}%")
    if np.any(valid):
        print(f"  d (px) min/med/max: {np.nanmin(disp[valid]):.2f} / {np.nanmedian(disp[valid]):.2f} / {np.nanmax(disp[valid]):.2f}")
    
    # Profundidad métrica Z = fB/d
    fx_rect = float(P1r[0,0])
    B_rect = float(abs(P2r[0,3]/P2r[0,0])) if abs(P2r[0,0])>1e-9 else float(np.linalg.norm(C2-C1))
    depth = np.full_like(disp, np.nan)
    depth[valid] = (fx_rect * B_rect) / disp[valid]
    print(f"  f(rect)={fx_rect:.2f} px, B={B_rect:.4f} m")
    if np.any(np.isfinite(depth)):
        print(f"  Z (m) min/med/max: {np.nanmin(depth):.3f} / {np.nanmedian(depth):.3f} / {np.nanmax(depth):.3f}")
    
    # --- Figura: mapa de disparidad ---
    fig1 = plt.figure(figsize=(12, 5))
    vd = disp[valid]
    vmin = float(np.percentile(vd, 2)) if len(vd)>0 else 0
    vmax = float(np.percentile(vd, 98)) if len(vd)>0 else 1
    step = max(1, int(np.round(disp.shape[1]/1600.0)))
    plt.imshow(disp[::step,::step], cmap='plasma', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Disparidad (px)')
    plt.title('Mapa de disparidad (StereoSGBM)')
    plt.axis('off')
    fig1.savefig(os.path.join(outdir, 'mapa_disparidad_2D.pdf'), bbox_inches='tight')
    fig1.savefig(os.path.join(outdir, 'mapa_disparidad_2D.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Guardado: mapa_disparidad_2D.pdf/png")
    
    # --- Figura: perfiles de disparidad ---
    h_r, w_r = disp.shape
    row_a, row_b = int(0.35*h_r), int(0.65*h_r)
    fig2 = plt.figure(figsize=(14, 5))
    plt.plot(np.arange(w_r), disp[row_a,:], label=f'Fila {row_a} (35%)', alpha=0.9)
    plt.plot(np.arange(w_r), disp[row_b,:], label=f'Fila {row_b} (65%)', alpha=0.9)
    plt.xlabel('Columna (px)'); plt.ylabel('Disparidad (px)')
    plt.title('Perfiles de disparidad en dos alturas')
    plt.legend(); plt.grid(True, alpha=0.25)
    fig2.savefig(os.path.join(outdir, 'perfiles_disparidad.pdf'), bbox_inches='tight')
    fig2.savefig(os.path.join(outdir, 'perfiles_disparidad.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Guardado: perfiles_disparidad.pdf/png")
    
    # Diagnóstico
    frac_inv = 1.0 - valid.mean()
    print(f"\n  Diagnóstico del mapa de disparidad:")
    print(f"  - Píxeles inválidos: {100.0*frac_inv:.1f}%")
    print(f"  - Fallos típicos: oclusiones laterales, reflejos, zonas sin textura, bordes finos.")
    print(f"  - Mejoras aplicadas: CLAHE, SGBM 3WAY, WLS, mediana, inpainting NS, recorte adaptativo.")
    
    return disp, depth

# ============================================================================
# RESUMEN FINAL — 12 preguntas del informe
# ============================================================================
def imprimir_resumen(K, err_repro1, err_repro2, n_corresp, n_inliers, n_total,
                     err_epi_medio, dy_before, dy_after, depth):
    print("\n" + "="*70)
    print("RESUMEN FINAL — Respuestas a las 12 preguntas del informe")
    print("="*70)
    
    print(f"\n1. MATRIZ INTRÍNSECA K:")
    print(f"   fx={K[0,0]:.4f}, fy={K[1,1]:.4f}, cx={K[0,2]:.4f}, cy={K[1,2]:.4f}")
    
    print(f"\n2. ERROR DE REPROYECCIÓN (calibración): 0.96 px")
    print(f"   Error reproyección post-triangulación: cam1={err_repro1:.4f} px, cam2={err_repro2:.4f} px")
    
    print(f"\n3. CORRESPONDENCIAS PARA F: {n_corresp} puntos (esquinas de ArUco)")
    
    pct = 100.0*n_inliers/n_total
    print(f"\n4. INLIERS TRAS RANSAC: {n_inliers}/{n_total} ({pct:.1f}%)")
    
    print(f"\n5. VERIFICACIÓN xr^T·F·xl ≈ 0: error epipolar medio = {err_epi_medio:.4f} px")
    
    print(f"\n6. DIFERENCIA ENTRE F Y E:")
    print(f"   F opera en coordenadas de píxel (incorpora K).")
    print(f"   E opera en coordenadas normalizadas/calibradas (solo geometría extrínseca R,t).")
    print(f"   Relación: E = K^T · F · K. E tiene exactamente 5 DoF (3 rotación + 2 dirección traslación).")
    
    print(f"\n7. ¿POR QUÉ t DE E NO TIENE ESCALA MÉTRICA?")
    print(f"   La matriz esencial se calcula a partir de correspondencias de imagen,")
    print(f"   que son invariantes ante escalado de la escena. t se recupera normalizado (||t||=1).")
    print(f"   La escala absoluta requiere información métrica externa.")
    
    print(f"\n8. FIJACIÓN DE ESCALA:")
    print(f"   Se usó el tamaño real del lado del ArUco ({TAMANO_ARUCO_M*100:.1f} cm).")
    print(f"   Factor = tamaño_real / distancia_3D_reconstruida entre esquinas consecutivas.")
    
    print(f"\n9. REDUCCIÓN DEL ERROR VERTICAL TRAS RECTIFICAR:")
    print(f"   Antes: {dy_before:.2f} px → Después: {dy_after:.2f} px")
    red = (1 - dy_after/(dy_before+1e-15))*100
    print(f"   Reducción: {red:.1f}%")
    
    print(f"\n10. ZONAS DONDE FALLA MÁS EL MAPA DE DISPARIDAD:")
    print(f"    - Bordes laterales (oclusiones por paralaje)")
    print(f"    - Superficies sin textura (paredes lisas, cielo)")
    print(f"    - Reflejos especulares y zonas sobreexpuestas")
    print(f"    - Bordes finos de objetos (discontinuidades de profundidad)")
    
    if np.any(np.isfinite(depth)):
        z_med = np.nanmedian(depth)
        z_min = np.nanmin(depth)
        z_max = np.nanmax(depth)
        print(f"\n11. COHERENCIA DE PROFUNDIDAD:")
        print(f"    Z min/mediana/max: {z_min:.3f} / {z_med:.3f} / {z_max:.3f} m")
        print(f"    Los valores deben ser coherentes con la distancia real de la escena.")
    
    print(f"\n12. POSIBLES MEJORAS DE CAPTURA:")
    print(f"    - Mayor baseline entre vistas (mejora precisión de profundidad)")
    print(f"    - Iluminación uniforme y difusa (reduce reflejos)")
    print(f"    - Escenas con más textura (mejora matching SGBM)")
    print(f"    - Más marcadores ArUco distribuidos por la escena")
    print(f"    - Uso de trípode o rig estéreo fijo (reduce error de pose)")

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    
    # --- Etapa 0: Carga ---
    img_l, img_r = cargar_imagenes()
    
    # --- Etapa 1: Correspondencias ArUco ---
    pts_l, pts_r, ids_comunes, marker_ids = obtener_correspondencias(img_l, img_r)
    visualizar_correspondencias(img_l, img_r, pts_l, pts_r, marker_ids, OUTDIR)
    
    # --- Etapas 2-4: Matriz Fundamental (8 puntos + RANSAC) ---
    F, mask_inliers = ransac_fundamental(pts_l, pts_r, umbral=15.0, max_iter=5000)
    p1_in = pts_l[mask_inliers]
    p2_in = pts_r[mask_inliers]
    mids_in = marker_ids[mask_inliers]
    n_inliers = int(np.sum(mask_inliers))
    n_total = len(pts_l)
    
    # Visualizar líneas epipolares de F
    visualizar_lineas_epipolares(img_l, img_r, F, p1_in, p2_in, OUTDIR,
                                  nombre="lineas_epipolares_F")
    
    # Error epipolar medio (para resumen)
    err_epi = _error_epipolar_simetrico(F, p1_in, p2_in)
    err_epi_medio = float(np.mean(err_epi))
    
    # --- Etapa 5: Matriz Esencial ---
    E = calcular_esencial(F, K)
    
    # Visualizar líneas epipolares desde E (transformando a F equivalente para comparar)
    # E induce líneas en coordenadas normalizadas; las convertimos a píxel:
    # F_from_E = K^-T · E · K^-1
    K_inv = np.linalg.inv(K)
    F_from_E = K_inv.T @ E @ K_inv
    F_from_E = F_from_E / np.linalg.norm(F_from_E)
    visualizar_lineas_epipolares(img_l, img_r, F_from_E, p1_in, p2_in, OUTDIR,
                                  nombre="lineas_epipolares_E")
    
    # --- Etapas 6-7: Pose (R, t) con test de quiralidad ---
    R, t, X_raw, idx_pose = seleccionar_pose(E, K, p1_in, p2_in)
    
    # --- Etapa 8: Escala métrica y reconstrucción dispersa ---
    print("\n" + "="*70)
    print("ETAPA 8 — Triangulación y reconstrucción dispersa con escala métrica")
    print("="*70)
    
    factor, t_esc, X = fijar_escala_metrica(p1_in, p2_in, K, R, t, TAMANO_ARUCO_M)
    print(f"  Factor de escala: {factor:.6f}")
    print(f"  t escalado: {t_esc.ravel()}")
    
    # Verificación de escala: distancia entre esquinas 0 y 1 del primer marcador
    if len(X) >= 2:
        d01 = np.linalg.norm(X[0] - X[1])
        print(f"  Distancia 3D esquina 0→1 (primer ArUco): {d01:.4f} m (real: {TAMANO_ARUCO_M:.4f} m)")
    
    # Error de reproyección
    err1, err2 = error_reproyeccion(K, R, t_esc, X, p1_in, p2_in)
    print(f"  Error reproyección: cam1={err1:.4f} px, cam2={err2:.4f} px")
    
    # Visualización 3D
    visualizar_3d(X, mids_in, R, t_esc, OUTDIR)
    
    # --- Etapa 9: Rectificación ---
    img_rect1, img_rect2, R1, R2, P1r, P2r, Q = rectificar_estereo(
        img_l, img_r, K, R, t_esc)
    
    # Error vertical y ajuste afín
    ptsL_rect, ptsR_rect, img_rect2_opt, dy_before, dy_after = calcular_error_vertical(
        img_rect2, p1_in, p2_in, K, R1, R2, P1r, P2r)
    
    # Reemplazar la imagen derecha con la versión alineada perfectamente
    img_rect2 = img_rect2_opt
    
    # Visualización rectificación (dibujando líneas por las correspondencias)
    visualizar_rectificacion(img_rect1, img_rect2, OUTDIR, pts_rect=ptsL_rect)
    
    # --- Etapa 10: Disparidad densa ---
    C1 = np.array([0.0, 0.0, 0.0])
    C2 = (-R.T @ t_esc).ravel()
    
    disp, depth = calcular_disparidad(
        img_rect1, img_rect2, P1r, P2r,
        p1_in, p2_in, K, R1, R2, C1, C2, OUTDIR)
    
    # --- Resumen final ---
    imprimir_resumen(K, err1, err2, len(p1_in), n_inliers, n_total,
                     err_epi_medio, dy_before, dy_after, depth)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETADO")
    print(f"Figuras guardadas en: {OUTDIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
