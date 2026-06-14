import cv2
import numpy as np
from scipy.optimize import least_squares
from geometria_epipolar import obtener_correspondencias, ocho_puntos_normalizado, _error_epipolar_simetrico

img1 = cv2.imread("FotosOriginales/1.png")
img2 = cv2.imread("FotosOriginales/2.png")
pts1, pts2, _, _ = obtener_correspondencias(img1, img2)

F_init = ocho_puntos_normalizado(pts1, pts2)

err_init = _error_epipolar_simetrico(F_init, pts1, pts2)
print("Error medio init:", np.mean(err_init))

def residual(f_vec):
    F = f_vec.reshape(3, 3)
    return _error_epipolar_simetrico(F, pts1, pts2)

res = least_squares(residual, F_init.flatten(), method='lm')
F_opt = res.x.reshape(3, 3)
U, S, Vt = np.linalg.svd(F_opt)
S[2] = 0.0
F_opt = U @ np.diag(S) @ Vt
F_opt = F_opt / np.linalg.norm(F_opt)

err_opt = _error_epipolar_simetrico(F_opt, pts1, pts2)
print("Error medio opt:", np.mean(err_opt))

# Let's see if this improves E and dy_after
K = np.array([
    [3990.1574415286, 0.0, 2818.2271080190],
    [0.0, 3988.4486191714, 2135.8102621727],
    [0.0, 0.0, 1.0],
])

def test_dy(F):
    E = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E)
    S = [1, 1, 0]
    E = U @ np.diag(S) @ Vt
    
    # Simple recoverPose mock
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    
    dist_coeffs = np.zeros(5)
    R1, R2, P1r, P2r, _, _, _ = cv2.stereoRectify(K, dist_coeffs, K, dist_coeffs, (5712, 4284), R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
    
    ptsL_rect = cv2.undistortPoints(pts1.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R1, P=P1r)
    ptsR_rect = cv2.undistortPoints(pts2.reshape(-1,1,2).astype(np.float64), K, dist_coeffs, R=R2, P=P2r)
    dy = np.mean(np.abs(ptsL_rect[:,0,1] - ptsR_rect[:,0,1]))
    return dy

print("dy_init:", test_dy(F_init))
print("dy_opt:", test_dy(F_opt))
