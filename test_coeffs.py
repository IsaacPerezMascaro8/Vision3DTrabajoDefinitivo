import cv2
import numpy as np
from geometria_epipolar import obtener_correspondencias, ocho_puntos_normalizado

img1 = cv2.imread("FotosOriginales/1.png")
img2 = cv2.imread("FotosOriginales/2.png")
pts1, pts2, _, _ = obtener_correspondencias(img1, img2)

F = ocho_puntos_normalizado(pts1, pts2)

lines1_cv = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

lines1_my = []
for i in range(len(pts2)):
    p2h = np.array([pts2[i,0], pts2[i,1], 1.0])
    l1 = F.T @ p2h
    l1 = l1 / np.sqrt(l1[0]**2 + l1[1]**2)
    lines1_my.append(l1)
lines1_my = np.array(lines1_my)

print("lines1_cv:")
print(lines1_cv[:5])
print("lines1_my:")
print(lines1_my[:5])
