import numpy as np

from feature import EstimateE_RANSAC


def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """

    # TODO Your code goes here
    U, S, V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # ! create matrix W and Z (Hartley's Book pp 258)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    t = U[:, 2].reshape(1, -1).T

    # ! 4 possible transformations
    # transformations = [
    #     np.vstack((np.hstack((U @ W.T @ V, t)), [0, 0, 0, 1])),
    #     np.vstack((np.hstack((U @ W.T @ V, -t)), [0, 0, 0, 1])),
    #     np.vstack((np.hstack((U @ W @ V, t)), [0, 0, 0, 1])),
    #     np.vstack((np.hstack((U @ W @ V, -t)), [0, 0, 0, 1])),
    # ]
    # return transformations

    R_set = np.vstack([np.hstack(U @ W.T @ V).reshape((3,3)), np.hstack(U @ W.T @ V).reshape((3,3)),
                      np.hstack(U @ W @ V).reshape((3,3)), np.hstack(U @ W @ V).reshape((3,3))]).reshape((4,3,3))

    C_set = np.vstack([t, -t, t, -t]).reshape((4, 3, 1))

    return R_set, C_set


def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """

    # TODO Your code goes here
    n = track1.shape[1]
    if track2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    for i in range(n):
        M = np.zeros((6, 6))
        M[:3, :4] = P1
        M[3:, :4] = P2
        M[:3, 4] = -track1[:, i]
        M[3:, 5] = -track2[:, i]

        U, S, V = np.linalg.svd(M)
        X = V[-1, :4]
        X = X / X[3]
        X = np.array(X).T

    return X


def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """

    # TODO Your code goes here
    count = 0

    # third row of R
    r3 = R[2]

    for X_i in X:
        if np.dot(r3, X_i - C) > 0:
            count += 1

    return count
    # valid_index = np.zeros
    # X.shape[0])
    return valid_index


def EstimateCameraPose(track1, track2):
    """
    Return the best pose configuration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """

    # TODO Your code goes here
    F = track1.shape[0]

    # print(np.sum(track1, axis=1))
    # Select only features which are matched
    mask = np.logical_and(np.sum(track1, axis=1) != -2, np.sum(track2, axis=1) != -2)
    x1, x2 = track1[mask], track2[mask]

    # Homogenize coordinates to [x1 y1 1]
    X1 = []
    idx = 0
    while idx < len(x1):
        X1.append([x1[idx][0], x1[idx][1], 1])
        idx += 1

    # Homogenize coordinates to [x2 y2 1]
    X2 = []
    idx = 0
    while idx < len(x2):
        X2.append([x2[idx][0], x2[idx][1], 1])
        idx += 1

    E, inlier = EstimateE_RANSAC(np.asarray(X1), np.asarray(X2), ransac_n_iter=500, ransac_thr=0.005)

    R, C = GetCameraPoseFromE(E)

    # print(C)

    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    validIndex = []
    for i in range(4):
        print('R ', R[i])
        print('C ', C[i])
        print('The other ', (np.dot(-(R[i], C[i]))))
        # A possible transformations
        P2 = np.hstack([R[i], -(R[i] @ C[i]).reshape((3, 1))])
        X = Triangulation(P1, P2, x1, x2)
        cheiralityIndex = EvaluateCheirality(P1, P2, X)
        validIndex.append(X[cheiralityIndex])
        print('Found %d valid points after triangulation ' %(np.sum(cheiralityIndex)))

    return R, C, X
