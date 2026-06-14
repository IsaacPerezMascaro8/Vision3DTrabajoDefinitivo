import cv2
import numpy as np
from geometria_epipolar import obtener_correspondencias, ocho_puntos_normalizado
from visualizacion import _dibujar_linea

img1 = cv2.imread("FotosOriginales/1.png")
img2 = cv2.imread("FotosOriginales/2.png")
pts1, pts2, _, _ = obtener_correspondencias(img1, img2)

F = ocho_puntos_normalizado(pts1, pts2)

# cv2 epilines
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)

vis1_cv = img1.copy()
vis2_cv = img2.copy()
vis1_my = img1.copy()
vis2_my = img2.copy()

color = (0, 255, 0)
for i in range(10):
    _dibujar_linea(vis1_cv, lines1[i], color)
    _dibujar_linea(vis2_cv, lines2[i], color)
    
    p1h = np.array([pts1[i,0], pts1[i,1], 1.0])
    p2h = np.array([pts2[i,0], pts2[i,1], 1.0])
    l2 = F @ p1h
    l1 = F.T @ p2h
    
    _dibujar_linea(vis1_my, l1, (0, 0, 255))
    _dibujar_linea(vis2_my, l2, (0, 0, 255))

cv2.imwrite("output/test_lines1_cv.png", cv2.resize(vis1_cv, (0,0), fx=0.2, fy=0.2))
cv2.imwrite("output/test_lines1_my.png", cv2.resize(vis1_my, (0,0), fx=0.2, fy=0.2))

diff1 = np.abs(vis1_cv.astype(int) - vis1_my.astype(int))
print("Max diff img1:", np.max(diff1))
