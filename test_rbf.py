import cv2
import numpy as np
import time
from scipy.interpolate import Rbf
from geometria_epipolar import obtener_correspondencias, ocho_puntos_normalizado

img1 = cv2.imread("FotosOriginales/1.png")
img2 = cv2.imread("FotosOriginales/2.png")
pts1, pts2, _, _ = obtener_correspondencias(img1, img2)

K = np.array([
    [3990.1574415286, 0.0, 2818.2271080190],
    [0.0, 3988.4486191714, 2135.8102621727],
    [0.0, 0.0, 1.0],
])

F = ocho_puntos_normalizado(pts1, pts2)
E = K.T @ F @ K
U, S, Vt = np.linalg.svd(E)
E = U @ np.diag([1, 1, 0]) @ Vt
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

dist_coeffs = np.zeros(5)
R1, R2, P1r, P2r, _, _, _ = cv2.stereoRectify(K, dist_coeffs, K, dist_coeffs, (5712, 4284), R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)

ptsL_rect = cv2.undistortPoints(pts1.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R1, P=P1r).reshape(-1,2)
ptsR_rect = cv2.undistortPoints(pts2.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R2, P=P2r).reshape(-1,2)

X_r = ptsR_rect[:, 0]
Y_r = ptsR_rect[:, 1]
Y_l = ptsL_rect[:, 1]

t0 = time.time()
h, w = 4284, 5712
grid_w, grid_h = 60, 45
gx, gy = np.meshgrid(np.linspace(0, w-1, grid_w), np.linspace(0, h-1, grid_h))

rbf = Rbf(X_r, Y_r, Y_l - Y_r, function='thin_plate')
dy_grid = rbf(gx, gy)

dy_full = cv2.resize(dy_grid.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)

map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
map2y_opt = map_y + dy_full

# evaluate new y for the points
Y_r_new = Y_r + rbf(X_r, Y_r)

print("RBF generation time:", time.time() - t0)
print("Max dy error after RBF:", np.max(np.abs(Y_l - Y_r_new)))

