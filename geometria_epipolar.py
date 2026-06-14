"""
geometria_epipolar.py
=====================
Etapas 1-5 del pipeline de visión 3D:
  1. Detección de ArUcos y correspondencias
  2. Normalización de puntos (Hartley)
  3. Algoritmo de los 8 puntos normalizado
  4. RANSAC para Matriz Fundamental robusta
  5. Cálculo de la Matriz Esencial

Todo implementado con NumPy (sin findFundamentalMat ni findEssentialMat).
"""

import numpy as np
import cv2


# ============================================================================
# ETAPA 1 — Detección de ArUcos y correspondencias
# ============================================================================

def detectar_aruco(imagen, diccionario=cv2.aruco.DICT_4X4_50, max_valid_id=8):
    """
    Detección robusta de ArUcos en 3 pasadas (estándar, agresiva, CLAHE).
    Fusiona resultados eligiendo la detección con mejor gradiente.
    Filtra IDs > max_valid_id. Aplica refinamiento subpíxel final.

    Retorna dict {id: esquinas (4x2)}.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(diccionario)
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    def make_std_params():
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 5
        p.cornerRefinementMaxIterations = 50
        return p

    def make_aggressive_params():
        p = make_std_params()
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 93
        p.adaptiveThreshWinSizeStep = 4
        p.adaptiveThreshConstant = 7
        p.minMarkerPerimeterRate = 0.005
        p.maxMarkerPerimeterRate = 4.0
        p.polygonalApproxAccuracyRate = 0.05
        p.minCornerDistanceRate = 0.005
        p.minDistanceToBorder = 0
        p.minMarkerDistanceRate = 0.005
        return p

    def run_pass(img_input, params):
        det = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = det.detectMarkers(img_input)
        result = {}
        if ids is not None:
            for i, mid in enumerate(ids.ravel()):
                result[int(mid)] = corners[i].reshape(4, 2)
        return result

    # 3 pasadas
    p1 = run_pass(imagen, make_std_params())
    p2 = run_pass(imagen, make_aggressive_params())
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    p3 = run_pass(enhanced, make_aggressive_params())

    # Gradiente para medir calidad
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    def sharpness(corners):
        total = 0.0
        for (x, y) in corners.reshape(-1, 2):
            xi, yi = int(round(x)), int(round(y))
            x0, x1 = max(0, xi-3), min(gray.shape[1]-1, xi+3)
            y0, y1 = max(0, yi-3), min(gray.shape[0]-1, yi+3)
            patch = mag[y0:y1+1, x0:x1+1]
            total += float(np.mean(patch)) if patch.size > 0 else 0.0
        return total

    # Fusión
    all_ids = set(p1) | set(p2) | set(p3)
    esquinas = {}
    for mid in all_ids:
        if mid > max_valid_id:
            continue
        cands = []
        if mid in p1: cands.append(p1[mid])
        if mid in p2: cands.append(p2[mid])
        if mid in p3:
            pts_ref = p3[mid].astype(np.float32).reshape(-1, 1, 2)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            cv2.cornerSubPix(gray, pts_ref, (7, 7), (-1, -1), crit)
            cands.append(pts_ref.reshape(4, 2))
        esquinas[mid] = max(cands, key=sharpness)

    # Refinamiento subpíxel final
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    for mid in list(esquinas):
        pts = esquinas[mid].astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(gray, pts, (7, 7), (-1, -1), crit)
        esquinas[mid] = pts.reshape(4, 2)

    return esquinas


def _alinear_esquinas_cyclic(c1, c2):
    """Alinea esquinas de c2 con c1 probando 4 rotaciones cíclicas."""
    best_s, best_c = 0, float('inf')
    for s in range(4):
        c = np.sum(np.linalg.norm(c1 - np.roll(c2, s, axis=0), axis=1))
        if c < best_c:
            best_c = c
            best_s = s
    return np.roll(c2, best_s, axis=0)


def obtener_correspondencias(img_left, img_right,
                             diccionario=cv2.aruco.DICT_4X4_50):
    """
    Detecta ArUcos en ambas imágenes, empareja por ID y extrae
    correspondencias de esquinas (4 por marcador).

    Retorna (pts_left Nx2, pts_right Nx2, ids_comunes, marker_ids N)
    """
    print("\n" + "="*70)
    print("ETAPA 1 — Detección de ArUcos y correspondencias")
    print("="*70)
    print("  Imagen izquierda:")
    esq_l = detectar_aruco(img_left, diccionario)
    print("  Imagen derecha:")
    esq_r = detectar_aruco(img_right, diccionario)

    ids_comunes = sorted(set(esq_l) & set(esq_r))
    if not ids_comunes:
        raise RuntimeError("No hay marcadores ArUco comunes.")

    pl, pr, mids = [], [], []
    for mid in ids_comunes:
        cl = esq_l[mid]
        cr = _alinear_esquinas_cyclic(cl, esq_r[mid])
        pl.append(cl); pr.append(cr)
        mids.extend([mid]*4)

    pts_l = np.vstack(pl)
    pts_r = np.vstack(pr)
    marker_ids = np.array(mids, dtype=int)

    print(f"\n  Marcadores en izquierda: {sorted(esq_l.keys())}")
    print(f"  Marcadores en derecha:  {sorted(esq_r.keys())}")
    print(f"  Marcadores comunes:     {ids_comunes}")
    print(f"  Total correspondencias: {len(pts_l)} puntos ({len(ids_comunes)} markers × 4 esquinas)")

    return pts_l, pts_r, ids_comunes, marker_ids


# ============================================================================
# ETAPA 2 — Normalización de puntos (Hartley)
# ============================================================================

def normalizar_puntos(pts):
    """
    Normalización de Hartley: traslada centroide al origen y escala
    para que la distancia media al origen sea √2.

    Retorna (pts_norm Nx2, T 3x3).
    """
    centroide = np.mean(pts, axis=0)
    pts_c = pts - centroide
    dist_media = np.mean(np.sqrt(np.sum(pts_c**2, axis=1)))
    if dist_media < 1e-10:
        dist_media = 1e-10
    s = np.sqrt(2.0) / dist_media

    T = np.array([
        [s, 0, -s*centroide[0]],
        [0, s, -s*centroide[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    pts_hom = np.column_stack([pts, np.ones(len(pts))])
    pts_norm = (T @ pts_hom.T).T[:, :2]
    return pts_norm, T


# ============================================================================
# ETAPA 3 — Algoritmo de los 8 puntos normalizado
# ============================================================================

def ocho_puntos_normalizado(ptsL, ptsR):
    """
    Estima la Matriz Fundamental con el algoritmo de los 8 puntos normalizado.

    Pasos:
      1. Normalizar ambos conjuntos → Tl, Tr
      2. Construir A tal que xr^T · F · xl = 0
      3. Resolver Af=0 por SVD (último vector singular de V^T)
      4. Imponer rango 2 (anular σ₃ mediante SVD de F)
      5. Desnormalizar: F = Tr^T · F_norm · Tl

    Requiere ≥ 8 correspondencias.
    Retorna F (3x3).
    """
    assert len(ptsL) >= 8, "Se necesitan al menos 8 correspondencias."

    ptsL_n, Tl = normalizar_puntos(ptsL)
    ptsR_n, Tr = normalizar_puntos(ptsR)

    n = len(ptsL_n)
    x1, y1 = ptsL_n[:, 0], ptsL_n[:, 1]
    x2, y2 = ptsR_n[:, 0], ptsR_n[:, 1]

    # Cada fila de A: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    A = np.column_stack([
        x2*x1, x2*y1, x2,
        y2*x1, y2*y1, y2,
        x1,    y1,    np.ones(n)
    ])

    # SVD → último vector singular
    _, _, Vt = np.linalg.svd(A)
    F_hat = Vt[-1].reshape(3, 3)

    # Forzar rango 2
    Uf, Sf, Vft = np.linalg.svd(F_hat)
    Sf[2] = 0.0
    F_norm = Uf @ np.diag(Sf) @ Vft

    # Desnormalizar
    F = Tr.T @ F_norm @ Tl
    F = F / np.linalg.norm(F)  # convención ||F||=1
    return F


# ============================================================================
# ETAPA 4 — RANSAC para Matriz Fundamental robusta
# ============================================================================

def _error_epipolar_simetrico(F, pts1, pts2):
    """
    Error epipolar simétrico (distancia punto-línea) para cada correspondencia.
    d = 0.5 * (|l2^T·x2| / ||l2[:2]|| + |l1^T·x1| / ||l1[:2]||)
    donde l2 = F·x1 (línea en img2), l1 = F^T·x2 (línea en img1).
    """
    n = len(pts1)
    ones = np.ones((n, 1))
    x1h = np.hstack([pts1, ones])  # (N,3)
    x2h = np.hstack([pts2, ones])

    # Líneas epipolares
    l2 = (F @ x1h.T).T   # en imagen 2
    l1 = (F.T @ x2h.T).T  # en imagen 1

    # Distancia punto-línea
    d2 = np.abs(np.sum(x2h * l2, axis=1)) / (np.sqrt(l2[:, 0]**2 + l2[:, 1]**2) + 1e-15)
    d1 = np.abs(np.sum(x1h * l1, axis=1)) / (np.sqrt(l1[:, 0]**2 + l1[:, 1]**2) + 1e-15)

    return 0.5 * (d1 + d2)


def ransac_fundamental(pts1, pts2, umbral=2.0, max_iter=5000, confianza=0.999):
    """
    Estimación robusta de F usando RANSAC con el algoritmo de 8 puntos.

    En cada iteración:
      - Muestrear 8 correspondencias aleatorias
      - Estimar F con ocho_puntos_normalizado
      - Calcular error epipolar simétrico
      - Contar inliers (error < umbral)
    Adapta el número de iteraciones dinámicamente.
    Al final, reestima F con todos los inliers.

    Retorna (F 3x3, mask_inliers N-bool).
    """
    print("\n" + "="*70)
    print("ETAPA 4 — RANSAC para Matriz Fundamental")
    print("="*70)

    n = len(pts1)
    assert n >= 8, f"Se necesitan ≥8 puntos, hay {n}"

    mejor_F = None
    mejor_mask = np.zeros(n, dtype=bool)
    mejor_num = 0
    rng = np.random.default_rng(42)

    iteraciones = max_iter
    for it in range(iteraciones):
        # Muestrear 8 puntos
        idx = rng.choice(n, 8, replace=False)
        try:
            F_cand = ocho_puntos_normalizado(pts1[idx], pts2[idx])
        except Exception:
            continue

        # Evaluar
        errores = _error_epipolar_simetrico(F_cand, pts1, pts2)
        mask = errores < umbral
        num_inliers = np.sum(mask)

        if num_inliers > mejor_num:
            mejor_num = num_inliers
            mejor_F = F_cand
            mejor_mask = mask.copy()

            # Adaptar iteraciones (fórmula RANSAC adaptativa)
            w = mejor_num / n
            if w > 0.999:
                break
            denom = 1.0 - w**8
            if denom < 1e-15:
                break
            iteraciones = min(max_iter, int(np.log(1 - confianza) / np.log(denom)) + 1)

    # Reestimar con todos los inliers
    if mejor_num >= 8:
        mejor_F = ocho_puntos_normalizado(pts1[mejor_mask], pts2[mejor_mask])

    # Verificar xr^T F xl ≈ 0
    errores_inliers = _error_epipolar_simetrico(mejor_F, pts1[mejor_mask], pts2[mejor_mask])
    err_medio = np.mean(errores_inliers)

    pct = 100.0 * mejor_num / n
    print(f"  Inliers: {mejor_num}/{n} ({pct:.1f}%)")
    print(f"  Error epipolar medio (inliers): {err_medio:.4f} px")
    print(f"  Iteraciones RANSAC: {min(it+1, iteraciones)}")
    print(f"\n  Matriz Fundamental F:\n{mejor_F}")

    # Verificación numérica xr^T F xl ≈ 0
    p1h = np.hstack([pts1[mejor_mask], np.ones((mejor_num, 1))])
    p2h = np.hstack([pts2[mejor_mask], np.ones((mejor_num, 1))])
    productos = np.abs(np.sum(p2h * (mejor_F @ p1h.T).T, axis=1))
    print(f"\n  Verificación xr^T·F·xl ≈ 0:")
    print(f"    Media |xr^T F xl|: {np.mean(productos):.6e}")
    print(f"    Max   |xr^T F xl|: {np.max(productos):.6e}")

    return mejor_F, mejor_mask


# ============================================================================
# ETAPA 5 — Matriz Esencial E
# ============================================================================

def calcular_esencial(F, K):
    """
    Calcula E = K^T · F · K y fuerza sus propiedades:
    dos valores singulares iguales (σ = media de σ₁,σ₂) y uno nulo.

    La Matriz Fundamental F opera en coordenadas de píxel:
      xr^T · F · xl = 0
    La Matriz Esencial E opera en coordenadas normalizadas (calibradas):
      x̂r^T · E · x̂l = 0   donde x̂ = K⁻¹·x
    E codifica la geometría epipolar independientemente de los parámetros
    intrínsecos de la cámara, mientras que F los incorpora.

    Retorna E (3x3).
    """
    print("\n" + "="*70)
    print("ETAPA 5 — Matriz Esencial E = K^T · F · K")
    print("="*70)

    E_raw = K.T @ F @ K

    U, S, Vt = np.linalg.svd(E_raw)
    print(f"  Valores singulares ANTES de corrección: {S}")

    # Forzar (σ, σ, 0)
    sigma = (S[0] + S[1]) / 2.0
    S_corr = np.array([sigma, sigma, 0.0])
    E = U @ np.diag(S_corr) @ Vt
    E = E / np.linalg.norm(E) * np.sqrt(2.0)

    _, S2, _ = np.linalg.svd(E)
    print(f"  Valores singulares DESPUÉS de corrección: {S2}")

    print("\n  NOTA CONCEPTUAL:")
    print("  • F (Fundamental): relaciona píxeles ↔ píxeles. Rango 2, 7 DoF.")
    print("  • E (Esencial): relaciona rayos normalizados. Rango 2, 5 DoF (R,t).")
    print("  • E contiene solo la geometría extrínseca (rotación + traslación).")

    return E
