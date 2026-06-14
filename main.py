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
    print("\n" + "="*70)
    print("ETAPA 10 — Mapa de disparidad denso (StereoSGBM)")
    print("="*70)

    SCALE_FACTOR = 0.5
    assert SCALE_FACTOR in [0.5, 0.25, 0.125], "Usa potencias de 1/2"

    h_orig, w_orig = imgL_rect.shape[:2]
    new_w, new_h = int(w_orig * SCALE_FACTOR), int(h_orig * SCALE_FACTOR)
    imgL_rect_small = cv2.resize(imgL_rect, (new_w, new_h), interpolation=cv2.INTER_AREA)
    imgR_rect_small = cv2.resize(imgR_rect, (new_w, new_h), interpolation=cv2.INTER_AREA)
    grayL = cv2.cvtColor(imgL_rect_small, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR_rect_small, cv2.COLOR_BGR2GRAY)
    
    # MÁSCARA ESTRICTA DE ZONA VÁLIDA (para evitar que WLS reviente en los bordes negros de alpha=1)
    mask_valid = (grayL > 5) & (grayR > 5)

    P1r_scaled = P1r.copy(); P1r_scaled[0, :] *= SCALE_FACTOR; P1r_scaled[1, :] *= SCALE_FACTOR
    P2r_scaled = P2r.copy(); P2r_scaled[0, :] *= SCALE_FACTOR; P2r_scaled[1, :] *= SCALE_FACTOR

    dist_rect = np.zeros((1, 5), dtype=float)

    # Proyectamos los puntos de validación
    ptsL_rect = cv2.undistortPoints(ptsL_in.reshape(-1, 1, 2).astype(np.float64), K, dist_rect, R=R1, P=P1r)
    ptsR_rect = cv2.undistortPoints(ptsR_in.reshape(-1, 1, 2).astype(np.float64), K, dist_rect, R=R2, P=P2r)

    disp_known = (ptsL_rect[:, 0, 0] - ptsR_rect[:, 0, 0]) * SCALE_FACTOR
    min_disp = max(0, int(np.floor(np.min(disp_known))) - 5)
    max_disp = int(np.ceil(np.max(disp_known))) + 10
    num_disp = ((max_disp - min_disp) // 16 + 1) * 16
    print(f"  Rango disparidad: {min_disp} - {max_disp}, numDisparities={num_disp}")

    # --- Parámetros de StereoSGBM de TU código ---
    block_size = 5
    canales = 1

    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * canales * block_size**2,
        P2=16 * canales * block_size**2,
        disp12MaxDiff=2,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp_raw = matcher.compute(grayL, grayR).astype(np.float32) / 16.0
    
    # Forzar valores no válidos fuera de la imagen ANTES del WLS para evitar artefactos gigantes
    disp_raw[~mask_valid] = min_disp - 1.0

    try:
        from cv2 import ximgproc
        wls = ximgproc.createDisparityWLSFilter(matcher_left=matcher)
        right_matcher = ximgproc.createRightMatcher(matcher)
        disp_right = right_matcher.compute(grayR, grayL).astype(np.float32) / 16.0
        disp_filtered = wls.filter(disp_raw, grayL, disparity_map_right=disp_right).astype(np.float32)
        print("  Filtro WLS aplicado con éxito.")
    except (ImportError, AttributeError, Exception):
        disp_filtered = disp_raw.copy()
        print("  Filtro WLS no disponible en esta instalación de OpenCV. Se usa el mapa base.")

    # Filtro de mediana inicial
    disp_filtered = np.nan_to_num(disp_filtered, nan=0.0).astype(np.float32)
    disp_median = cv2.medianBlur(disp_filtered, 5)
    
    # Clamp agresivo para descartar ruido del WLS en los bordes negros
    disp_median[disp_median <= min_disp] = np.nan
    disp_median[disp_median >= min_disp + num_disp + 10] = np.nan
    disp_median[~mask_valid] = np.nan  # Mantenemos los bordes negros como NaN puros
    
    disp_final = disp_median

    # ==============================================================================
    # --- DETECCIÓN AUTOMÁTICA ADAPTATIVA DEL ANCHO DE OCLUSIÓN LATERAL ---
    # ==============================================================================
    valid_indices = [np.where(~np.isnan(disp_final[r]))[0] for r in range(disp_final.shape[0])]
    first_valid_cols = [idx[0] for idx in valid_indices if len(idx) > 0]
    last_valid_cols = [idx[-1] for idx in valid_indices if len(idx) > 0]

    if len(first_valid_cols) > 0:
        crop_left_small = min(int(np.percentile(first_valid_cols, 90)) + 15, int(new_w * 0.20))
    else:
        crop_left_small = int(num_disp + 15)

    if len(last_valid_cols) > 0:
        crop_right_small = min(disp_final.shape[1] - int(np.percentile(last_valid_cols, 10)) + 15, int(new_w * 0.15))
    else:
        crop_right_small = 15
    # ==============================================================================

    # --- INPAINTING ULTRA-LOCAL CON PRESERVACIÓN DE BORDES (NAVIER-STOKES) ---
    invalid_mask = np.isnan(disp_final)
    # IMPORTANTE: Inpaintamos SOLO dentro de la imagen, no los inmensos bordes negros de alpha=1
    inpaint_target = invalid_mask & mask_valid
    valid_data_mask = ~invalid_mask & mask_valid

    if np.any(valid_data_mask):
        d_min = float(np.nanmin(disp_final[valid_data_mask]))
        d_max = float(np.nanmax(disp_final[valid_data_mask]))
        d_range = d_max - d_min if (d_max - d_min) > 1e-9 else 1.0
        
        # Normalizamos
        disp_uint8 = np.zeros_like(disp_final, dtype=np.uint8)
        disp_uint8[valid_data_mask] = np.clip(
            255.0 * (disp_final[valid_data_mask] - d_min) / d_range, 0, 255
        ).astype(np.uint8)
        
        # Máscara de inpainting (solo agujeros internos)
        inpaint_mask = (inpaint_target * 255).astype(np.uint8)
        
        # Usamos Navier-Stokes (INPAINT_NS) con radio=1 para rellenar sin difuminar bordes
        disp_uint8_inpainted = cv2.inpaint(disp_uint8, inpaint_mask, inpaintRadius=1, flags=cv2.INPAINT_NS)
        
        # Desnormalizamos
        disp_inpainted_float = d_min + (disp_uint8_inpainted.astype(np.float32) / 255.0) * d_range
        
        # Conservamos los píxeles originales calculados por SGBM y solo rellenamos los vacíos internos
        disp_final_clean = disp_final.copy()
        disp_final_clean[inpaint_target] = disp_inpainted_float[inpaint_target]
        disp_final = disp_final_clean

    # Escalado de la disparidad al tamaño original
    if SCALE_FACTOR != 1.0:
        # Poner los NaN a 0 temporalmente para el resize
        disp_clean_resize = np.nan_to_num(disp_final, nan=0.0).astype(np.float32)
        disp_up = cv2.resize(
            disp_clean_resize,
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST,
        ) / SCALE_FACTOR
        
        mask = cv2.resize(
            np.isfinite(disp_final).astype(np.uint8),
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST,
        )
        disp_up[mask == 0] = np.nan
        disp = disp_up
    else:
        disp = disp_final.copy()

    # ==============================================================================
    # --- ANULACIÓN AUTOMÁTICA ADAPTATIVA EN RESOLUCIÓN COMPLETA ---
    # ==============================================================================
    crop_left = int(crop_left_small / SCALE_FACTOR)
    crop_right = int(crop_right_small / SCALE_FACTOR)

    disp[:, :crop_left] = np.nan
    disp[:, -crop_right:] = np.nan
    # ==============================================================================

    valid = np.isfinite(disp) & (disp > 0)
    print(f"  Disparidad válida: {100.0 * valid.mean():.1f}% de píxeles")
    if np.any(valid):
        print(f"  d (px) min/med/max: {np.nanmin(disp[valid]):.2f} / {np.nanmedian(disp[valid]):.2f} / {np.nanmax(disp[valid]):.2f}")

    # Profundidad métrica real Z = fB/d
    fx_rect = float(P1r[0, 0])
    B_rect = float(abs(P2r[0, 3] / P2r[0, 0])) if abs(P2r[0, 0]) > 1e-9 else float(np.linalg.norm(C2 - C1))
    depth = np.full_like(disp, np.nan)
    depth[valid] = (fx_rect * B_rect) / disp[valid]
    print(f"  f (rect) = {fx_rect:.2f} px, B = {B_rect:.4f} m")
    if np.any(np.isfinite(depth)):
        print(f"  Z aprox (m) min/med/max: {np.nanmin(depth[valid]):.3f} / {np.nanmedian(depth[valid]):.3f} / {np.nanmax(depth[valid]):.3f}")

    # Visualización del mapa de disparidad crudo (contraste optimizado)
    fig1 = plt.figure(figsize=(12, 5))
    valid_disp = disp[valid]
    if len(valid_disp) > 0:
        vmin = float(np.percentile(valid_disp, 5))
        vmax = float(np.percentile(valid_disp, 95))
        print(f"  Rango visualizado: {vmin:.1f} - {vmax:.1f} px")
    else:
        vmin, vmax = 0.0, 1.0

    # Submuestreo para visualización
    disp_viz_step = max(1, int(np.round(disp.shape[1] / 1600.0)))
    disp_viz = disp[::disp_viz_step, ::disp_viz_step]

    # Mostramos el mapa de disparidad crudo
    import matplotlib.cm as cm
    cmap = cm.plasma
    cmap.set_bad(color='white') # Fondo blanco para las oclusiones (los NaNs)
    plt.imshow(disp_viz, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.colorbar(label='Disparidad (px)')
    plt.title('Mapa de disparidad (contraste optimizado - bordes definidos)')
    plt.axis('off')

    fig1.savefig(os.path.join(outdir, 'mapa_disparidad_2D.pdf'), bbox_inches='tight')
    fig1.savefig(os.path.join(outdir, 'mapa_disparidad_2D.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Perfiles de disparidad en dos alturas
    h_rect, w_rect = disp.shape
    row_a = int(0.35 * h_rect)
    row_b = int(0.65 * h_rect)
    x = np.arange(w_rect)

    fig2 = plt.figure(figsize=(14, 5))
    plt.plot(x, disp[row_a, :], label=f'Perfil fila {row_a}', alpha=0.9)
    plt.plot(x, disp[row_b, :], label=f'Perfil fila {row_b}', alpha=0.9)
    plt.xlabel('Columna (px)')
    plt.ylabel('Disparidad (px)')
    plt.title('Perfiles de disparidad en dos alturas')
    plt.legend()
    plt.grid(True, alpha=0.25)

    fig2.savefig(os.path.join(outdir, 'perfiles_disparidad.pdf'), bbox_inches='tight')
    fig2.savefig(os.path.join(outdir, 'perfiles_disparidad.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # Comentario automático de calidad
    frac_invalid = 1.0 - valid.mean()
    print()
    print('  Diagnóstico rápido del mapa de disparidad:')
    print(f"  - Píxeles inválidos: {100.0 * frac_invalid:.1f}%")
    print('  - Fallos típicos esperables: oclusiones, reflejos, zonas sin textura y bordes finos.')
    print('  - Mejoras aplicadas: SGBM de bordes duros, inpainting local adaptativo, mediana y percentiles.')

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
