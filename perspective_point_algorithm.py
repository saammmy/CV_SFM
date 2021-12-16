import numpy as np

from final_camera_pose import EvaluateCheirality
from utils import Rotation2Quaternion
from utils import Quaternion2Rotation


def PnP(X, x):
    """
    Implement the linear perspective-n-point algorithm

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    """

    n = x.shape[0]

    A = []

    for i in range(n):
        A.append([X[i, 0], X[i, 1], X[i, 2], 1, 0, 0, 0, 0,
                  -x[i, 0] * X[i, 0], -x[i, 0] * X[i, 1], -x[i, 0] * X[i, 2], -x[i, 0]])

        A.append([0, 0, 0, 0, X[i, 0], X[i, 1], X[i, 2], 1,
                  -x[i, 1] * X[i, 0], -x[i, 1] * X[i, 1], -x[i, 1] * X[i, 2], -x[i, 0]])
    A = np.stack(A)

    _, _, Vh = np.linalg.svd(A)
    P = Vh[-1].reshape((3, 4))

    U, D, Vh = np.linalg.svd(P[:, :3])
    R = U @ Vh
    t = P[:, 3] / D[0]
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    C = -R.T @ t

    return R, C


def PnP_RANSAC(X, x, ransac_n_iter, ransacThreshold):
    """
    Estimate pose using PnP with RANSAC

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    inlier : ndarray of shape (n,)
        The indicator of inliers, i.e., the entry is 1 if the point is a inlier,
        and 0 otherwise
    """
    n = x.shape[0]
    maxInlier = 0

    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])

    print('Running PnP_RANSAC for %d pairs' % n)

    for _ in range(ransac_n_iter):
        sampleIdx = np.random.choice(n, 6)
        sampled_X = X[sampleIdx]
        sampled_x = x[sampleIdx]

        REstimate, CEstimate = PnP(sampled_X, sampled_x)

        twoD2threeD = (X - CEstimate) @ REstimate.T
        xEstimate = twoD2threeD[:, :2] / twoD2threeD[:, -1:]
        error = np.linalg.norm(x - xEstimate, axis=1)

        numInlier = np.sum(error < ransacThreshold)

        if numInlier > maxInlier:
            maxInlier = numInlier
            R, C = REstimate, CEstimate
            inlier = error < ransacThreshold

    return R, C, inlier


def ComputePoseJacobian(p, X):
    """
    Compute the pose Jacobian

    Parameters
    ----------
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion
    X : ndarray of shape (3,)
        3D point

    Returns
    -------
    dfdp : ndarray of shape (2, 7)
        The pose Jacobian
    """
    C = p[:3]
    q = p[3:]
    R = Quaternion2Rotation(q)

    uvw = R @ (X - C)

    duvw_dC = -R

    duvw_dR = np.zeros((3, 9))
    duvw_dR[0, :3] = X - C
    duvw_dR[1, 3:6] = X - C
    duvw_dR[2, 6:] = X - C

    qw, qx, qy, qz = q
    dR_dq = np.array([[0, 0, -4 * qy, -4 * qz],
                      [-2 * qz, 2 * qy, 2 * qx, -2 * qw],
                      [2 * qy, 2 * qz, 2 * qw, 2 * qx],
                      [2 * qz, 2 * qy, 2 * qx, 2 * qw],
                      [0, -4 * qx, 0, -4 * qz],
                      [-2 * qx, -2 * qw, 2 * qz, 2 * qy],
                      [-2 * qy, 2 * qz, -2 * qw, 2 * qx],
                      [2 * qx, 2 * qw, 2 * qz, 2 * qy],
                      [0, -4 * qx, -4 * qy, 0]])

    duvw_dq = duvw_dR @ dR_dq

    duvw_dp = np.hstack([duvw_dC, duvw_dq])
    assert duvw_dp.shape[0] == 3 and duvw_dp.shape[1] == 7

    u, v, w = uvw[0], uvw[1], uvw[2]
    du_dp, dv_dp, dw_dp = duvw_dp[0], duvw_dp[1], duvw_dp[2]

    dfdp = np.stack([(w * du_dp - u * dw_dp) / w ** 2, (w * dv_dp - v * dw_dp) / w ** 2])
    assert dfdp.shape[0] == 2 and dfdp.shape[1] == 7

    return dfdp


def ComputePnPError(R, C, X, b):
    """
    Compute nonlinear PnP estimation error and 1D vector f
    """
    f = (X - C) @ R.T
    f = f[:, :2] / f[:, -1:]  # non homogenize
    error = np.average(np.linalg.norm(f - b.reshape(-1, 2), axis=1))
    return error, f.reshape(-1)


def PnP_nl(R, C, X, x):
    """
    Update the pose using the pose Jacobian

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix refined by PnP
    c : ndarray of shape (3,)
        Camera center refined by PnP
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R_refined : ndarray of shape (3, 3)
        The rotation matrix refined by nonlinear optimization
    C_refined : ndarray of shape (3,)
        The camera center refined by nonlinear optimization
    """
    # maximum number of iterations
    maxIter = 50
    # threshold for terminating gradient update
    epsilon = 1e-3
    dampingLambda = 1
    n = x.shape[0]
    b = x.reshape(-1)

    previousError, f = ComputePnPError(R, C, X, b)

    for iter in range(maxIter):
        print('Running nonlinear PnP iteration %d' % iter)
        p = np.concatenate([C, Rotation2Quaternion(R)])

        # Compute Jacobian matrix and stack them
        dfdp = []
        for i in range(n):
            dfdp.append(ComputePoseJacobian(p, X[i]))
        dfdp = np.vstack(dfdp)  # 2n x 7
        assert dfdp.shape[0] == 2 * n and dfdp.shape[1] == 7

        dp = np.linalg.inv(dfdp.T @ dfdp + dampingLambda * np.eye(7)) @ dfdp.T @ (b - f)
        C += dp[:3]
        q = p[3:] + dp[3:]
        q = q / np.linalg.norm(q)
        R = Quaternion2Rotation(q)

        error, f = ComputePnPError(R, C, X, b)
        RRefined, CRefined = R, C

        if previousError - error < epsilon:
            break
        else:
            previousError = error

    return RRefined, CRefined
