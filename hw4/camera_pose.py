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
    transformations = [
        np.vstack((np.hstack((U @ W.T @ V, t)), [0, 0, 0, 1])),
        np.vstack((np.hstack((U @ W.T @ V, -t)), [0, 0, 0, 1])),
        np.vstack((np.hstack((U @ W @ V, t)), [0, 0, 0, 1])),
        np.vstack((np.hstack((U @ W @ V, -t)), [0, 0, 0, 1])),
    ]
    # return transformations

    R_set = np.vstack(np.hstack(U @ W.T @ V), np.hstack(U @ W.T @ V),
                      np.hstack(U @ W @ V), np.hstack(U @ W @ V))

    C_set = np.vstack(t, -t, t, -t)

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

    GetCameraPoseFromE(E)

    return R, C, X
