import cv2
import numpy as np
from geometria_epipolar import obtener_correspondencias, ocho_puntos_normalizado, normalizar_puntos
from visualizacion import visualizar_lineas_epipolares

img1 = cv2.imread("FotosOriginales/1.png")
img2 = cv2.imread("FotosOriginales/2.png")
pts1, pts2, _, _ = obtener_correspondencias(img1, img2)

# Mi implementacion (todos los puntos)
F_mio = ocho_puntos_normalizado(pts1, pts2)

# OpenCV 8-point
F_cv_8, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

# OpenCV RANSAC
F_cv_ransac, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
mask = mask.ravel().astype(bool)

print("F_mio:\n", F_mio)
print("F_cv_8:\n", F_cv_8)
print("F_cv_ransac:\n", F_cv_ransac)

visualizar_lineas_epipolares(img1, img2, F_mio, pts1, pts2, "output", "test_F_mio")
visualizar_lineas_epipolares(img1, img2, F_cv_8, pts1, pts2, "output", "test_F_cv_8")
if F_cv_ransac is not None:
    visualizar_lineas_epipolares(img1, img2, F_cv_ransac, pts1[mask], pts2[mask], "output", "test_F_cv_ransac")

