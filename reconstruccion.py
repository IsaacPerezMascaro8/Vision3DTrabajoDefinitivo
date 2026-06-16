"""
reconstruccion.py
=================
Etapas 6-8 del pipeline de visión 3D:
  6. Descomposición de E en 4 poses candidatas (R, t)
  7. Test de quiralidad y selección de pose
  8. Triangulación lineal (DLT) y reconstrucción dispersa

Todo implementado con NumPy (sin recoverPose ni triangulatePoints).
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================================
# ETAPA 6 — Descomposición de E en 4 poses
# ============================================================================

def descomponer_esencial(E):
    """
    Descompone E mediante SVD (Hartley & Zisserman):
      E = U · diag(1,1,0) · V^T
      W = [[0,-1,0],[1,0,0],[0,0,1]]
      R1 = U·W·V^T,  R2 = U·W^T·V^T,  t = ±U[:,2]

    Genera las 4 combinaciones: (R1,t), (R1,-t), (R2,t), (R2,-t).
    Retorna lista de 4 tuplas (R 3x3, t 3x1).
    """
    print("\n" + "="*70)
    print("ETAPA 6 — Descomposición de E en 4 poses candidatas")
    print("="*70)

    U, S, Vt = np.linalg.svd(E)

    # Asegurar det > 0
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0],
                   [1,  0, 0],
                   [0,  0, 1]], dtype=np.float64)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2].reshape(3, 1)
    t = t / (np.linalg.norm(t) + 1e-15)

    poses = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    for i, (R, tv) in enumerate(poses):
        print(f"  Pose {i+1}: det(R)={np.linalg.det(R):.4f}, "
              f"t=[{tv[0,0]:.4f}, {tv[1,0]:.4f}, {tv[2,0]:.4f}]")

    return poses


# ============================================================================
# ETAPA 7 — Triangulación lineal (DLT) punto a punto
# ============================================================================

def triangular_punto(P1, P2, p1, p2):
    """
    Triangulación DLT de un punto 3D.
    Construye el sistema A·X=0 (4 ecuaciones, 4 incógnitas homogéneas)
    y resuelve por SVD.

    P1, P2: matrices de proyección 3x4.
    p1, p2: puntos 2D (x, y).
    Retorna X (3,) coordenadas 3D.
    """
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    A = np.array([
        x1 * P1[2, :] - P1[0, :],
        y1 * P1[2, :] - P1[1, :],
        x2 * P2[2, :] - P2[0, :],
        y2 * P2[2, :] - P2[1, :],
    ], dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    X_hom = Vt[-1]
    return X_hom[:3] / (X_hom[3] + 1e-15)


def triangular_puntos(P1, P2, pts1, pts2):
    """Triangula N correspondencias. Retorna puntos_3d (Nx3)."""
    return np.array([triangular_punto(P1, P2, pts1[i], pts2[i])
                     for i in range(len(pts1))])


# ============================================================================
# ETAPA 7 — Test de quiralidad y selección de pose
# ============================================================================

def _verificar_quiralidad(R, t, K, pts1, pts2):
    """
    Triangula puntos y cuenta cuántos tienen Z > 0 en AMBAS cámaras.
    Cámara 1 en el origen: P1 = K·[I|0].
    Cámara 2: P2 = K·[R|t].
    Retorna (num_delante, puntos_3d).
    """
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    X = triangular_puntos(P1, P2, pts1, pts2)

    z_cam1 = X[:, 2]
    z_cam2 = (R @ X.T + t).T[:, 2]
    delante = (z_cam1 > 0) & (z_cam2 > 0)
    return np.sum(delante), X


def seleccionar_pose(E, K, pts1, pts2):
    """
    Descompone E en 4 poses, triangula para cada una y selecciona
    aquella con más puntos con Z > 0 en ambas cámaras (quiralidad).

    Retorna (R 3x3, t 3x1, puntos_3d Nx3, idx_mejor).
    """
    print("\n" + "="*70)
    print("ETAPA 7 — Test de quiralidad y selección de pose")
    print("="*70)

    poses = descomponer_esencial(E)

    mejor_num = -1
    R_best, t_best, X_best, idx_best = None, None, None, -1

    print(f"\n  {'Pose':<8} {'Z>0 ambas':<15} {'%':<10}")
    print("  " + "-"*33)
    for i, (R, t) in enumerate(poses):
        num, X = _verificar_quiralidad(R, t, K, pts1, pts2)
        pct = 100.0 * num / len(pts1)
        print(f"  {i+1:<8} {num}/{len(pts1):<12} {pct:.1f}%")
        if num > mejor_num:
            mejor_num = num
            R_best, t_best, X_best, idx_best = R, t, X, i

    pct_best = 100.0 * mejor_num / len(pts1)
    print(f"\n  ✅ Pose seleccionada: {idx_best+1} ({mejor_num}/{len(pts1)} = {pct_best:.1f}%)")
    if pct_best < 90:
        print("  ⚠️  AVISO: Menos del 90% de puntos pasan el test de quiralidad.")
    print(f"  R =\n{R_best}")
    print(f"  t = {t_best.ravel()}")

    return R_best, t_best, X_best, idx_best


# ============================================================================
# ETAPA 8 — Reconstrucción dispersa con escala métrica
# ============================================================================

def fijar_escala_metrica(pts1, pts2, K, R, t, tamano_aruco_m):
    """
    Fija la escala métrica comparando la distancia 3D entre dos esquinas
    consecutivas de un ArUco con el tamaño real del marcador.

    Retorna (factor_escala, t_escalado 3x1, X_escalados Nx3).
    """
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    # Triangular las 2 primeras esquinas (lado superior del primer marcador)
    X0 = triangular_punto(P1, P2, pts1[0], pts2[0])
    X1 = triangular_punto(P1, P2, pts1[1], pts2[1])
    dist_3d = np.linalg.norm(X0 - X1)
    factor = tamano_aruco_m / dist_3d

    # Aplicar escala
    t_esc = t * factor
    X_all = triangular_puntos(P1, P2, pts1, pts2) * factor

    return factor, t_esc, X_all


def error_reproyeccion(K, R, t, X, pts1, pts2):
    """Calcula error de reproyección medio en ambas cámaras (píxeles)."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    err1, err2 = np.zeros(len(X)), np.zeros(len(X))
    for i in range(len(X)):
        Xh = np.append(X[i], 1.0)
        p1 = P1 @ Xh; p1 = p1[:2]/(p1[2]+1e-15)
        p2 = P2 @ Xh; p2 = p2[:2]/(p2[2]+1e-15)
        err1[i] = np.linalg.norm(p1 - pts1[i])
        err2[i] = np.linalg.norm(p2 - pts2[i])
    return np.mean(err1), np.mean(err2)



