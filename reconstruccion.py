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


# ============================================================================
# Visualización 3D
# ============================================================================

# Paleta de colores distinguibles para los IDs de ArUco
_ARUCO_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
    '#ffd8b1', '#000075', '#a9a9a9',
]

def visualizar_reconstruccion_3d(puntos_3d, aruco_ids, R, t_escalada,
                                  titulo="Reconstrucción 3D métrica"):
    """
    Genera un gráfico 3D agrupado por ArUco.

    Parámetros
    ----------
    puntos_3d : np.ndarray (Nx3)
        Puntos 3D triangulados y ya escalados a metros.
    aruco_ids : np.ndarray (N,)
        ID de ArUco correspondiente a cada punto (misma longitud que puntos_3d).
    R : np.ndarray (3x3)
        Rotación de la cámara 2 respecto a la cámara 1.
    t_escalada : np.ndarray (3x1)
        Vector de traslación escalado a metros.
    titulo : str
        Título del gráfico.
    """

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Nube de puntos agrupada por ArUco ID
    unique_ids = np.unique(aruco_ids)
    for idx, aid in enumerate(unique_ids):
        color = _ARUCO_COLORS[idx % len(_ARUCO_COLORS)]
        mask = aruco_ids == aid
        pts = puntos_3d[mask]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=color, marker='o', s=60, alpha=0.9,
                   edgecolors='k', linewidths=0.4,
                   label=f'ArUco ID {aid}')
        # Anotar las esquinas con números pequeños
        for j, (x, y, z) in enumerate(pts):
            ax.text(x, y, z, f' {j}', fontsize=7, color=color, alpha=0.7)

    # Cámara 1 — en el origen
    ax.scatter(0, 0, 0, c='red', marker='^', s=250, zorder=5,
               edgecolors='darkred', linewidths=1.2, label='Cámara 1 (origen)')

    # Cámara 2 — centro óptico real C₂ = −Rᵀ · t_escalada
    C2 = (-R.T @ t_escalada).ravel()
    ax.scatter(C2[0], C2[1], C2[2], c='blue', marker='^', s=250, zorder=5,
               edgecolors='darkblue', linewidths=1.2, label='Cámara 2')

    # Ejes de orientación de cada cámara (R, G, B → X, Y, Z)
    all_pts = np.vstack([puntos_3d, [[0, 0, 0]], [C2]])
    escala_ejes = np.max(np.ptp(all_pts, axis=0)) * 0.12
    if escala_ejes < 1e-6:
        escala_ejes = 0.05

    eje_labels = ['X', 'Y', 'Z']
    eje_colors = ['r', 'g', 'b']

    for eje, col, lbl in zip(np.eye(3), eje_colors, eje_labels):
        ax.quiver(0, 0, 0, eje[0], eje[1], eje[2],
                  length=escala_ejes, color=col, linewidth=2, arrow_length_ratio=0.15)

    for eje, col in zip(R.T, eje_colors):
        ax.quiver(C2[0], C2[1], C2[2], eje[0], eje[1], eje[2],
                  length=escala_ejes, color=col, linewidth=2,
                  alpha=0.6, arrow_length_ratio=0.15)

    ax.plot([0, C2[0]], [0, C2[1]], [0, C2[2]],
            'k--', linewidth=1.0, alpha=0.5, label='Baseline')

    def _dibujar_frustum(ax, centro, R_cam, escala, color, alpha_face=0.12):
        hw = escala * 0.55
        hh = escala * 0.40
        d  = escala * 0.9

        esquinas_local = np.array([
            [-hw, -hh, d],
            [ hw, -hh, d],
            [ hw,  hh, d],
            [-hw,  hh, d],
        ])

        esquinas_world = (R_cam @ esquinas_local.T).T + centro

        verts = [esquinas_world.tolist()]
        poly = Poly3DCollection(verts, alpha=alpha_face, facecolors=color,
                                edgecolors=color, linewidths=1.2)
        ax.add_collection3d(poly)

        for corner in esquinas_world:
            ax.plot([centro[0], corner[0]],
                    [centro[1], corner[1]],
                    [centro[2], corner[2]],
                    color=color, linewidth=0.8, alpha=0.45)

        for i in range(4):
            j = (i + 1) % 4
            ax.plot([esquinas_world[i, 0], esquinas_world[j, 0]],
                    [esquinas_world[i, 1], esquinas_world[j, 1]],
                    [esquinas_world[i, 2], esquinas_world[j, 2]],
                    color=color, linewidth=1.2, alpha=0.7)

    _dibujar_frustum(ax, np.array([0, 0, 0]), np.eye(3), escala_ejes, 'red')
    _dibujar_frustum(ax, C2, R.T, escala_ejes, 'blue')

    ax.set_xlabel('X (metros)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (metros)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z — Profundidad (metros)', fontsize=12, labelpad=10)
    ax.set_title(titulo, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.85)

    max_range = np.max(np.ptp(all_pts, axis=0)) * 0.6
    if max_range < 1e-6:
        max_range = 1.0
    mid = np.mean(all_pts, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.invert_yaxis()
    ax.invert_xaxis()

    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    plt.tight_layout()
    plt.show()
