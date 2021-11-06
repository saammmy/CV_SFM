import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """

    # TODO Your code goes here
    NNDR = 0.8
    nbrs = NearestNeighbors(n_neighbors=2).fit(des2)
    distances, indices = nbrs.kneighbors(des1, 2, return_distance=True)

    # print(len(distances))
    # print(indices, distances)
    pairlist = []

    x1, x2, ind1 = [], [], []
    for i, distance in enumerate(distances):
        if distance[0] < NNDR * distance[1]:
            x1.append(loc1[i])
            x2.append(loc2[indices[i][0]])
            ind1.append(i)

    # print(x1, x2)

    return x1, x2, ind1


def EstimateE(x1, x2):
    """
    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """

    # TODO Your code goes here
    X = np.zeros((8, 9))
    for i in range(8):
        X[i] = [x1[i, 0] * x2[i, 0], x1[i, 0] * x2[i, 1], x1[i, 0] * x2[i, 2],
                x1[i, 1] * x2[i, 0], x1[i, 1] * x2[i, 1], x1[i, 1] * x2[i, 2],
                x1[i, 2] * x2[i, 0], x1[i, 2] * x2[i, 1], x1[i, 2] * x2[i, 2]]

    X = np.matmul(X.transpose(), X)
    eigenValues, eigenVectors = np.linalg.eig(X)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    # solution of X*E = 0 is F:
    F = eigenVectors[8].reshape(3, 3)
    U, sigma, V = np.linalg.svd(F)
    S = np.zeros((3, 3))
    S[0, 0] = S[1, 1] = (sigma[0] + sigma[1]) / 2
    E = np.matmul((np.matmul(U, S)), V.transpose())

    return E


def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """

    # TODO Your code goes here
    EstimateE(x1, x2)
    return E, inlier


def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """

    # TODO Your code goes here
    # print(Im.shape)
    # print(Im)

    loc, des = [], []
    loc = np.array([])
    for i in range(len(Im[:, :, :, :]) - 1):
        # Load consecutive images I_i and I_{i+1}

        # TODO Your code goes here
        img = Im[i, :, :, :]
        # Extract SIFT features
        # TODO Your code goes here
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des_temp = sift.detectAndCompute(gray, None)
        loc1 = np.array([kp[idx].pt for idx in range(0, len(kp))]).reshape(-1, 2)
        print(loc1)
        np.append(loc, loc1)
        des.append(des_temp)

    print('Loc: ', loc)
    print('Desc: ', des)
    # print('Matched Features.')
    # trace = torch.full((len(Im[:, :, :, :]), feature_lenght, 2), -1)

    for i in range(len(Im[i, :, :, :]) - 1):
        # Find the matches between two images (x1 <--> x2)
        x1, x2, ind = MatchSIFT(loc1, des1, loc2, des2)

    return track
