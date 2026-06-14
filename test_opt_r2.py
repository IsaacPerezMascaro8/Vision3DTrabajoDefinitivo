import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_rot
from scipy.optimize import least_squares
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

ptsL_rect = cv2.undistortPoints(pts1.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R1, P=P1r)
ptsR_rect = cv2.undistortPoints(pts2.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R2, P=P2r)
print("Initial dy mean:", np.mean(np.abs(ptsL_rect[:,0,1] - ptsR_rect[:,0,1])))

def optimize_R2(rvec):
    dR = R_rot.from_rotvec(rvec).as_matrix()
    R2_new = dR @ R2
    pL = ptsL_rect[:,0,:]
    pR_new = cv2.undistortPoints(pts2.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R2_new, P=P2r)[:,0,:]
    return (pL[:,1] - pR_new[:,1])

res = least_squares(optimize_R2, np.zeros(3))
dR = R_rot.from_rotvec(res.x).as_matrix()
R2_opt = dR @ R2

ptsR_rect_opt = cv2.undistortPoints(pts2.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R2_opt, P=P2r)
print("Optimized dy mean:", np.mean(np.abs(ptsL_rect[:,0,1] - ptsR_rect_opt[:,0,1])))
