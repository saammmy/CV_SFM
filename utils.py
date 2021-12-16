import numpy as np
from pyquaternion import Quaternion


def Rotation2Quaternion(R):
    """
    Convert a rotation matrix to quaternion

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix

    Returns
    -------
    q : ndarray of shape (4,)
        The unit quaternion (w, x, y, z)
    """

    q = Quaternion(matrix=R).elements

    return q


def Quaternion2Rotation(q):
    """
    Convert a quaternion to rotation matrix

    Parameters
    ----------
    q : ndarray of shape (4,)
        Unit quaternion (w, x, y, z)

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    """

    q = Quaternion(q)
    R = q.rotation_matrix

    return R
