#!/usr/bin/env python3
"""
main.py — Pipeline completo de Visión 3D Estéreo (Trabajo 2)
=============================================================
Etapas 0-10: ArUco → F → E → Pose → Triangulación → Rectificación → SGBM
"""
import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

from geometria_epipolar import (obtener_correspondencias, ocho_puntos_normalizado,
                                 ransac_fundamental, calcular_esencial,
                                 normalizar_puntos, _error_epipolar_simetrico)
from reconstruccion import (seleccionar_pose, fijar_escala_metrica,
                             triangular_puntos, error_reproyeccion)
from visualizacion import (visualizar_correspondencias, visualizar_lineas_epipolares,
                            visualizar_rectificacion, visualizar_reconstruccion_3d,
                            visualizar_mapa_disparidad_2d, visualizar_perfiles_disparidad)
from pipeline_denso import (rectificar_estereo, calcular_error_vertical, 
                            computar_mapa_disparidad, calcular_profundidad,
                            imprimir_resumen)

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
SCALE_FACTOR = 0.2       # Forzamos la baja resolución para SGBM
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



def generar_grafica_ssd_diapositiva(imgL, imgR, ptL, outdir):
    print("\n" + "="*70)
    print("EXTRA 1 — Generando gráfica SSD (Búsqueda de Correspondencia 1D)")
    print("="*70)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    h, w = grayL.shape

    x_L, y_L = int(ptL[0]), int(ptL[1])
    window_size = 51 
    half_w = window_size // 2

    template = grayL[y_L-half_w:y_L+half_w+1, x_L-half_w:x_L+half_w+1].astype(np.float32)
    x_r_vals, ssd_vals = [], []

    rango_busqueda = 800
    x_min = max(half_w, x_L - rango_busqueda)
    x_max = min(w - half_w - 1, x_L + 50)
    
    for x_R in range(x_min, x_max, 5): 
        roi = grayR[y_L-half_w:y_L+half_w+1, x_R-half_w:x_R+half_w+1].astype(np.float32)
        ssd = np.sum((template - roi)**2)
        x_r_vals.append(x_R)
        ssd_vals.append(ssd)

    min_idx = np.argmin(ssd_vals)
    best_x_R = x_r_vals[min_idx]

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])
    zoom_margin = 800

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
    ax1.axhline(y=y_L, color='magenta', linewidth=1.5)
    ax1.text(x_L - zoom_margin + 50, y_L - 30, 'scanline', color='magenta', fontsize=14, fontweight='bold')
    rect = patches.Rectangle((x_L-half_w, y_L-half_w), window_size, window_size, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title("Izquierda", fontsize=16)
    ax1.axis('off')
    ax1.set_xlim(max(0, x_L - zoom_margin), min(w, x_L + zoom_margin//2))
    ax1.set_ylim(min(h, y_L + zoom_margin//2), max(0, y_L - zoom_margin//2)) 

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
    ax2.axhline(y=y_L, color='magenta', linewidth=1.5)
    ax2.axvline(x=best_x_R, color='red', linewidth=1.5)
    ax2.set_title("Derecha", fontsize=16)
    ax2.axis('off')
    ax2.set_xlim(max(0, x_L - zoom_margin), min(w, x_L + zoom_margin//2))
    ax2.set_ylim(min(h, y_L + zoom_margin//2), max(0, y_L - zoom_margin//2))

    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax3.plot(x_r_vals, ssd_vals, color='darkblue', linewidth=2)
    ax3.axvline(x=best_x_R, color='red', linewidth=1.5)
    ax3.set_title("SSD", fontsize=16)
    ax3.set_xlabel("Píxeles (Eje X)", fontsize=12)
    ax3.set_ylabel("Error Cuadrático", fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.suptitle("Búsqueda de correspondencia (SSD)", fontsize=24, fontweight='bold')
    plt.tight_layout()
    
    ruta_guardado = os.path.join(outdir, 'grafica_busqueda_SSD.png')
    fig.savefig(ruta_guardado, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Gráfica guardada en: {ruta_guardado}")


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
    
    # Visualizar líneas epipolares desde E 
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
    
    if len(X) >= 2:
        d01 = np.linalg.norm(X[0] - X[1])
        print(f"  Distancia 3D esquina 0→1 (primer ArUco): {d01:.4f} m (real: {TAMANO_ARUCO_M:.4f} m)")
    
    err1, err2 = error_reproyeccion(K, R, t_esc, X, p1_in, p2_in)
    print(f"  Error reproyección: cam1={err1:.4f} px, cam2={err2:.4f} px")
    
    visualizar_reconstruccion_3d(X, mids_in, R, t_esc)
    
    # --- Etapa 9: Rectificación ---
    img_rect1, img_rect2, R1, R2, P1r, P2r, Q = rectificar_estereo(
        img_l, img_r, K, R, t_esc)
    
    ptsL_rect, ptsR_rect, img_rect2_opt, dy_before, dy_after = calcular_error_vertical(
        img_rect2, p1_in, p2_in, K, R1, R2, P1r, P2r)
    
    img_rect2 = img_rect2_opt
    visualizar_rectificacion(img_rect1, img_rect2, OUTDIR, pts_rect=ptsL_rect)
    
    # Extraer gráfica SSD de validación en un punto con textura (ArUco)
    # Seleccionamos la primera esquina del primer ArUco detectado y rectificado
    if len(ptsL_rect) > 0:
        punto_test = ptsL_rect[0] 
        generar_grafica_ssd_diapositiva(img_rect1, img_rect2, punto_test, OUTDIR)
    
    # --- Etapa 10: Disparidad densa ---
    C1 = np.array([0.0, 0.0, 0.0])
    C2 = (-R.T @ t_esc).ravel()
    
    disp = computar_mapa_disparidad(
        img_rect1, img_rect2, P1r, P2r,
        p1_in, p2_in, K, R1, R2)
    depth = calcular_profundidad(disp, P1r, P2r, C1, C2)
    
    visualizar_mapa_disparidad_2d(disp, OUTDIR)
    visualizar_perfiles_disparidad(disp, OUTDIR)
    
    # --- Resumen final ---
    imprimir_resumen(K, err1, err2, len(p1_in), n_inliers, n_total,
                     err_epi_medio, dy_before, dy_after, depth, TAMANO_ARUCO_M)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETADO")
    print(f"Figuras guardadas en: {OUTDIR}/")
    print("="*70)

if __name__ == "__main__":
    main()