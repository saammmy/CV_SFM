import numpy as np
import sys
from scipy.optimize import least_squares
from perspective_point_algorithm import ComputePoseJacobian
from utils import Quaternion2Rotation
from utils import Rotation2Quaternion


def FindMissingReconstruction(X, track_i):
    """
    Find the points that will be newly added

    Parameters
    ----------
    X : ndarray of shape (F, 3)
        3D points
    track_i : ndarray of shape (F, 2)
        2D points of the newly registered image

    Returns
    -------
    addedPoint : ndarray of shape (F,)
        The indicator of new points that are valid for the new image and are 
        not reconstructed yet
    """

    cnd1 = np.logical_and(np.logical_and(X[:, 0] != 0, X[:, 1] != 0), X[:, 2] != 0)

    cnd2 = np.logical_and(track_i[:, 0] != -1, track_i[:, 1] != -1)
    cnd3 = np.logical_and(cnd1, cnd2)

    N = X.shape[0]
    addedPoint = np.zeros(N)
    addedPoint[cnd3] = 1

    return addedPoint


def ComputeTriangulationError(X, P1, P2, b):
    """
    Compute averaged nonlinear triangulation error E  and vector f for each point in X
    """
    homo_X = np.insert(X, 3, 1, axis=1)
    x1 = homo_X @ P1.T
    x1 = x1[:, :2] / x1[:, -1:]
    x2 = homo_X @ P2.T
    x2 = x2[:, :2] / x2[:, -1:]

    error1 = np.linalg.norm(x1 - b[:, :2], axis=1)
    error2 = np.linalg.norm(x2 - b[:, 2:], axis=1)
    error = (np.average(error1) + np.average(error2)) / 2

    f = np.hstack([x1, x2])
    return error, f


def CompressP(P):
    """
    compress (3, 4) projection matrix to (7), C + quaternion
    """
    R = P[:, :3]
    q = Rotation2Quaternion(R)
    t = P[:, 3]
    C = -R.T @ t
    p = np.concatenate([C, q])
    return p


def Triangulation_nl(X, P1, P2, x1, x2):
    """
    Refine the triangulated points

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        3D points
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    x1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    x2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X_new : ndarray of shape (n, 3)
        The set of refined 3D points
    """

    maxIter = 50
    eps = 5e-5
    lmbd = 1e-2
    n = x1.shape[0]
    b = np.hstack([x1, x2])

    p1, p2 = CompressP(P1), CompressP(P2)

    previousError, f = ComputeTriangulationError(X, P1, P2, b)

    for iter in range(maxIter):
        print('Iteration %d of non-linear triangulation' % (iter))

        for i in range(n):
            J_X = [ComputePointJacobian(X[i], p1), ComputePointJacobian(X[i], p2)]
            J_X = np.vstack(J_X)
            if J_X.shape[0] != 4 or J_X.shape[1] != 3:
                sys.exit("Incorrect Jacobian.")

            dX = np.linalg.inv(J_X.T @ J_X + lmbd * np.eye(3)) @ J_X.T @ (b[i] - f[i])
            X[i] += dX

        error, f = ComputeTriangulationError(X, P1, P2, b)
        X_new = X

        if previousError - error < eps:
            break
        else:
            previousError = error
    return X_new


def ComputePointJacobian(X, p):
    """
    Compute the point Jacobian

    Parameters
    ----------
    X : ndarray of shape (3,)
        3D point
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion

    Returns
    -------
    J_X : ndarray of shape (2, 3)
        The point Jacobian
    """
    C = p[:3]
    q = p[3:]
    R = Quaternion2Rotation(q)

    S1 = R @ (X - C)
    dS1_dX = R

    u, v, w = S1[0], S1[1], S1[2]
    du_dX, dv_dX, dw_dX = dS1_dX[0], dS1_dX[1], dS1_dX[2]

    J_X = np.stack([(w * du_dX - u * dw_dX) / w ** 2, (w * dv_dX - v * dw_dX) / w ** 2])
    if J_X.shape[0] != 2 or J_X.shape[1] != 3:
        sys.exit("Incorrect Jacobian.")
    return J_X


def SetupBundleAdjustment(P, X, track):
    """
    Setup bundle adjustment

    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras

    Returns
    -------
    z : ndarray of shape (7K+3J,)
        The optimization variable that is made of all camera poses and 3D points
    b : ndarray of shape (2M,)
        The 2D points in track, where M is the number of 2D visible points
    S : ndarray of shape (2M, 7K+3J)
        The sparse indicator matrix that indicates the locations of Jacobian computation
    camera_index : ndarray of shape (M,)
        The index of camera for each measurement
    point_index : ndarray of shape (M,)
        The index of 3D point for each measurement
    """

    K, J = P.shape[0], X.shape[0]

    z = []
    for i in range(K):
        z.append(CompressP(P[i]))
    for i in range(J):
        z.append(X[i])
    z = np.hstack(z)

    cdn1 = np.logical_and(track[:, :, 0] != -1, track[:, :, 1] != -1)
    pointIndice, cameraIndice = np.nonzero(cdn1.transpose(1, 0))

    b = track.transpose(1, 0, 2)[cdn1.transpose(1, 0)].reshape(-1)
    M = int(b.shape[0] / 2)

    S = np.zeros((2 * M, 7 * K + 3 * J))
    for m in range(M):
        c_id, p_id = cameraIndice[m], pointIndice[m]
        if c_id != 0 and c_id != 1:
            S[2 * m: 2 * (m + 1), 7 * c_id: 7 * (c_id + 1)] = 1
        S[2 * m: 2 * (m + 1), 7 * K + 3 * p_id: 7 * K + 3 * (p_id + 1)] = 1

    return z, b, S, cameraIndice, pointIndice


def MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index):
    """
    Evaluate the reprojection error

    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    b : ndarray of shape (2M,)
        2D measured points
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points
    camera_index : ndarray of shape (M,)
        Index of camera for each measurement
    point_index : ndarray of shape (M,)
        Index of 3D point for each measurement

    Returns
    -------
    err : ndarray of shape (2M,)
        The reprojection error
    """

    P, X = UpdatePosePoint(z, n_cameras, n_points)

    projections = []
    for c, p in zip(camera_index, point_index):
        projection = P[c] @ np.insert(X[p], 3, 1)
        projections.append(projection[:2] / projection[2])
    rays = np.vstack(projections).reshape(-1)

    err = np.abs(rays - b)
    return err


def UpdatePosePoint(z, n_cameras, n_points):
    """
    Update the poses and 3D points

    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points

    Returns
    -------
    addedP : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    addedX : ndarray of shape (J, 3)
        The set of refined 3D points
    """

    addedP = np.zeros((n_cameras, 3, 4))
    for i in range(n_cameras):
        p = z[7 * i: 7 * (i + 1)]
        C = p[:3]
        R = Quaternion2Rotation(p[3:] / np.linalg.norm(p[3:]))
        t = - R @ C
        addedP[i] = np.hstack([R, t.reshape((3, 1))])

    addedX = z[7 * n_cameras:].reshape((n_points, 3))
    return addedP, addedX


def F(z, b, n_cameras, n_points, camera_index, point_index):
    error = MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index)
    return np.average(error)


def RunBundleAdjustment(P, X, track):
    """
    Run bundle adjustment

    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras

    Returns
    -------
    addedP : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    addedX : ndarray of shape (J, 3)
        The set of refined 3D points
    """

    allCameras, allPoints = P.shape[0], X.shape[0]
    print('%d Camera Bundle Adjustment for %d points' % (allCameras, allPoints))
    z0, b, S, cameraIndice, pointIndice = SetupBundleAdjustment(P, X, track)
    r = least_squares(MeasureReprojection, z0, args=(b, allCameras, allPoints, cameraIndice, pointIndice),
                      jac_sparsity=S)

    addedP, addedX = UpdatePosePoint(r.x, allCameras, allPoints)
    return addedP, addedX
