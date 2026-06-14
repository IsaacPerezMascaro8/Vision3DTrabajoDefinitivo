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

ptsL_rect = cv2.undistortPoints(pts1.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R1, P=P1r).reshape(-1,2)
ptsR_rect = cv2.undistortPoints(pts2.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R2, P=P2r).reshape(-1,2)

# Source points: ptsR_rect
# Target points: (ptsR_rect.X, ptsL_rect.Y)  <- Keep X exact, force Y perfectly
target_pts = np.column_stack([ptsR_rect[:, 0], ptsL_rect[:, 1]])

# TPS matches exactly!
tps = cv2.createThinPlateSplineShapeTransformer()
matches = [cv2.DMatch(i, i, 0) for i in range(len(ptsR_rect))]

src_pts_res = ptsR_rect.reshape(1, -1, 2).astype(np.float32)
dst_pts_res = target_pts.reshape(1, -1, 2).astype(np.float32)

tps.estimateTransformation(src_pts_res, dst_pts_res, matches)

ptsR_tps = tps.applyTransformation(src_pts_res)[1].reshape(-1, 2)
print("TPS dy max error:", np.max(np.abs(ptsL_rect[:, 1] - ptsR_tps[:, 1])))
print("TPS dx max error:", np.max(np.abs(ptsR_rect[:, 0] - ptsR_tps[:, 0])))
