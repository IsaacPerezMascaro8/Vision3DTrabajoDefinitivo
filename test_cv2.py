import cv2
import numpy as np
from geometria_epipolar import obtener_correspondencias

img1 = cv2.imread("FotosOriginales/1.png")
img2 = cv2.imread("FotosOriginales/2.png")
pts1, pts2, _, _ = obtener_correspondencias(img1, img2)

F_cv, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
mask = mask.ravel().astype(bool)
pts1_m = pts1[mask]
pts2_m = pts2[mask]

K = np.array([
    [3990.1574415286, 0.0, 2818.2271080190],
    [0.0, 3988.4486191714, 2135.8102621727],
    [0.0, 0.0, 1.0],
])

E = K.T @ F_cv @ K
U, S, Vt = np.linalg.svd(E)
S = [1, 1, 0]
E = U @ np.diag(S) @ Vt

_, R, t, _ = cv2.recoverPose(E, pts1_m, pts2_m, K)

dist_coeffs = np.zeros(5)
R1, R2, P1r, P2r, _, _, _ = cv2.stereoRectify(K, dist_coeffs, K, dist_coeffs, (5712, 4284), R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)

ptsL_rect = cv2.undistortPoints(pts1_m.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R1, P=P1r)
ptsR_rect = cv2.undistortPoints(pts2_m.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R2, P=P2r)
dy = np.mean(np.abs(ptsL_rect[:,0,1] - ptsR_rect[:,0,1]))
print("dy with cv2.FM_RANSAC:", dy)
