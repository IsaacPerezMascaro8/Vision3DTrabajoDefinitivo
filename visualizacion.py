"""
visualizacion.py
================
Funciones de visualización para el pipeline de visión 3D:
  - Correspondencias ArUco
  - Líneas epipolares (F y E)
  - Imágenes rectificadas con líneas horizontales
"""

import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualizar_correspondencias(img_l, img_r, pts_l, pts_r, marker_ids, outdir):
    """Dibuja correspondencias ArUco numeradas en ambas imágenes."""
    step = max(1, img_l.shape[1] // 2000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.imshow(cv2.cvtColor(img_l[::step, ::step], cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img_r[::step, ::step], cv2.COLOR_BGR2RGB))

    unique = np.unique(marker_ids)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 1)))
    for idx, mid in enumerate(unique):
        c = cmap[idx % len(cmap)]
        m = marker_ids == mid
        pl, pr = pts_l[m]/step, pts_r[m]/step
        for j in range(len(pl)):
            ax1.plot(pl[j,0], pl[j,1], 'o', color=c, ms=5)
            ax1.annotate(f'{mid}.{j}', (pl[j,0], pl[j,1]), fontsize=6, color=c)
            ax2.plot(pr[j,0], pr[j,1], 'o', color=c, ms=5)
            ax2.annotate(f'{mid}.{j}', (pr[j,0], pr[j,1]), fontsize=6, color=c)
    ax1.set_title('Izquierda — ArUco'); ax2.set_title('Derecha — ArUco')
    ax1.axis('off'); ax2.axis('off')
    fig.tight_layout()
    p = os.path.join(outdir, 'correspondencias_aruco.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Guardado: {p}")


def _dibujar_linea(img, linea, color, grosor=8):
    """Dibuja línea epipolar ax+by+c=0 sobre imagen."""
    a, b, c = linea
    h, w = img.shape[:2]
    if abs(b) > 1e-8:
        y0 = int(-c/b)
        y1 = int(-(a*w+c)/b)
        cv2.line(img, (0, y0), (w, y1), color, grosor)
    elif abs(a) > 1e-8:
        x = int(-c/a)
        cv2.line(img, (x, 0), (x, h), color, grosor)


def visualizar_lineas_epipolares(img1, img2, F, pts1, pts2, outdir,
                                  nombre="lineas_epipolares_F", n_lineas=15):
    """Dibuja líneas epipolares en ambas imágenes inducidas por F."""
    vis1, vis2 = img1.copy(), img2.copy()
    n = len(pts1)
    indices = list(range(0, n, max(1, n//n_lineas)))[:n_lineas]

    for i, idx in enumerate(indices):
        color = (int((50+i*37)%256), int((150+i*53)%256), int((200+i*29)%256))
        p1h = np.array([pts1[idx,0], pts1[idx,1], 1.0])
        p2h = np.array([pts2[idx,0], pts2[idx,1], 1.0])
        l2 = F @ p1h  # línea en img2
        l1 = F.T @ p2h  # línea en img1
        
        # Grosor adaptativo según resolución
        grosor = max(3, img1.shape[1] // 500)
        _dibujar_linea(vis2, l2, color, grosor)
        _dibujar_linea(vis1, l1, color, grosor)
        
        radio = max(5, img1.shape[1] // 250)
        cv2.circle(vis1, (int(pts1[idx,0]), int(pts1[idx,1])), radio, color, -1)
        cv2.circle(vis2, (int(pts2[idx,0]), int(pts2[idx,1])), radio, color, -1)

    # Guardar concatenación con antialiasing (INTER_AREA)
    scale = 2000.0 / vis1.shape[1]
    if scale < 1.0:
        new_w = int(vis1.shape[1] * scale)
        new_h = int(vis1.shape[0] * scale)
        v1_s = cv2.resize(vis1, (new_w, new_h), interpolation=cv2.INTER_AREA)
        v2_s = cv2.resize(vis2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        v1_s, v2_s = vis1, vis2
        
    concat = np.hstack([v1_s, v2_s])
    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    ax.imshow(cv2.cvtColor(concat, cv2.COLOR_BGR2RGB))
    ax.set_title(nombre.replace('_',' '))
    ax.axis('off')
    p = os.path.join(outdir, f'{nombre}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Guardado: {p}")


def visualizar_rectificacion(img_rect1, img_rect2, outdir, pts_rect=None):
    """Concatena imágenes rectificadas con líneas horizontales de verificación.
    Si se pasan pts_rect, dibuja las líneas pasando exactamente por esos puntos."""
    
    # Redimensionado con antialiasing
    scale = 2000.0 / img_rect1.shape[1]
    if scale < 1.0:
        new_w = int(img_rect1.shape[1] * scale)
        new_h = int(img_rect1.shape[0] * scale)
        r1 = cv2.resize(img_rect1, (new_w, new_h), interpolation=cv2.INTER_AREA)
        r2 = cv2.resize(img_rect2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        r1, r2, scale = img_rect1.copy(), img_rect2.copy(), 1.0
        
    concat = np.hstack([r1, r2])
    h, w = concat.shape[:2]
    
    if pts_rect is not None:
        # Dibujar línea horizontal por cada punto
        n = len(pts_rect)
        for i in range(n):
            color = (int((50+i*37)%256), int((150+i*53)%256), int((200+i*29)%256))
            y = int(pts_rect[i, 1] * scale)
            if 0 <= y < h:
                cv2.line(concat, (0, y), (w, y), color, max(1, int(2*scale)))
                # Dibujar los puntos en ambas mitades
                x1 = int(pts_rect[i, 0] * scale)
                cv2.circle(concat, (x1, y), max(3, int(8*scale)), color, -1)
    else:
        # Dibujar grid espaciado
        paso = max(1, h // 20)
        for y in range(paso, h, paso):
            color = tuple(int(x) for x in np.random.randint(50, 255, 3))
            cv2.line(concat, (0, y), (w, y), color, 2)

    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    ax.imshow(cv2.cvtColor(concat, cv2.COLOR_BGR2RGB))
    ax.set_title('Rectificación estéreo — Verificación con líneas horizontales')
    ax.axis('off')
    p = os.path.join(outdir, 'rectificacion_lineas.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    cv2.imwrite(os.path.join(outdir, 'rectified_left.png'), img_rect1)
    cv2.imwrite(os.path.join(outdir, 'rectified_right.png'), img_rect2)
    print(f"  Guardado: {p}")
