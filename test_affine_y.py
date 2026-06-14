import cv2
import numpy as np
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

ptsL_rect = cv2.undistortPoints(pts1.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R1, P=P1r)[:,0,:]
ptsR_rect = cv2.undistortPoints(pts2.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R2, P=P2r)[:,0,:]

dy_before = np.mean(np.abs(ptsL_rect[:,1] - ptsR_rect[:,1]))
print("dy before affine:", dy_before)

# Y-only affine correction
X_r = ptsR_rect[:, 0]
Y_r = ptsR_rect[:, 1]
Y_l = ptsL_rect[:, 1]

A = np.column_stack([X_r, Y_r, np.ones_like(X_r)])
params, _, _, _ = np.linalg.lstsq(A, Y_l, rcond=None)

M = np.array([
    [1.0, 0.0, 0.0],
    [params[0], params[1], params[2]]
])

ptsR_new_Y = params[0]*X_r + params[1]*Y_r + params[2]
dy_after = np.mean(np.abs(Y_l - ptsR_new_Y))
print("dy after affine Y-only:", dy_after)
