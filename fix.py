import os

new_code = """def calcular_disparidad(imgL_rect, imgR_rect, P1r, P2r, ptsL_in, ptsR_in,
                         K, R1, R2, C1, C2, outdir):
    print("\\n" + "="*70)
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
"""

with open('/home/isaac/Tercero/pruebavision3d/main.py', 'r') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith('def calcular_disparidad('):
        start_idx = i
    if line.startswith('# RESUMEN FINAL'):
        end_idx = i - 1
        break

if start_idx != -1 and end_idx != -1:
    with open('/home/isaac/Tercero/pruebavision3d/main.py', 'w') as f:
        f.writelines(lines[:start_idx])
        f.write(new_code)
        f.writelines(lines[end_idx:])
    print("Reemplazo exitoso.")
else:
    print(f"Error encontrando indices: {start_idx}, {end_idx}")
