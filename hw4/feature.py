import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import torch
import pickle, pandas as pd


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

    # print(x2)

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
    assert x1.shape == x2.shape, f"Shapes do not match {x1.shape} != {x2.shape}"
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
    # print(eigenVectors[8])
    # solution of X*E = 0 is F:
    F = eigenVectors[8].reshape(3, 3)
    # print(F)
    U, sigma, V = np.linalg.svd(F)
    S = np.zeros((3, 3))
    S[0, 0] = S[1, 1] = (sigma[0] + sigma[1]) / 2
    E = np.matmul((np.matmul(U, S)), V.transpose())

    # lines1 = cv2.computeCorrespondEpilines(x2, F)
    # lines1 = lines1.reshape(-1, 3)
    # print(lines1)
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

    ransac_n_iter = 500
    # kk = 0

    highest_number_of_happy_points = -1
    best_estimated_essential_matrix = np.identity(3)

    for r in range(ransac_n_iter):
        s1 = np.random.randint(x1.shape[0], size=8)
        s2 = np.random.randint(x2.shape[0], size=8)
        x1_s = x1[s1]
        x2_s = x2[s1]

        E = EstimateE(x1_s, x2_s)

        ''' get the eMat with the lowest error
        X'EX = 0 or in our case the error
        *[x2, y2, 1] * E * [x1, y1, 1] = 0 Found here
        Found here https://www.cc.gatech.edu/~hays/compvision/proj3/
        '''
        # sample_evaluation = []
        inlier_idx = []
        for i in range(len(x1)):
            E_dot_x2 = np.matmul(E, x2[i])
            dst = np.sum(x1[i] * E_dot_x2, axis=0)

            error = dst / (np.linalg.norm(x1[i]) * np.linalg.norm(E_dot_x2))
            if np.abs(error) < ransac_thr:
                inlier_idx.append(i)

            if len(inlier_idx) > highest_number_of_happy_points:
                highest_number_of_happy_points = len(inlier_idx)
                best_estimated_essential_matrix = E
                inlier = np.array(inlier_idx)

    E = best_estimated_essential_matrix
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

    loc, des = [], []
    n_features = 0

    for i in range(len(Im[:, :, :, :])):
        # TODO Your code goes here
        img = Im[i, :, :, :]
        # Extract SIFT features
        # TODO Your code goes here
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp_temp, des_temp = sift.detectAndCompute(gray, None)

        # print(des_temp.shape[0])
        n_features += des_temp.shape[0]
        des_temp = des_temp.tolist()

        kp_temp = [list(kp_temp[idx].pt) for idx in range(0, len(kp_temp))]

        loc.append(kp_temp)
        des.append(des_temp)

    track = torch.full((len(Im[:, :, :, :]), n_features, 2), -1)

    K_inv = np.linalg.inv(K)

    for i in range(len(Im[:, :, :, :])):
        # track_i = torch.full((len(Im[:, :, :, :]), n_features, 2), -1)
        track_i = np.ones((len(Im[:, :, :, :]), n_features, 2)) * -1
        for j in range(i + 1, len(Im[:, :, :, :])):
            # Find the matches between two images (x1 <--> x2)
            x1, x2, ind = MatchSIFT(loc[i], des[i], loc[j], des[j])

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

            # Normalize coordinate by multiplying the inverse of intrinsics
            x1 = [np.dot(K_inv, X1[idx]) for idx in range(len(X1))]
            x2 = [np.dot(K_inv, X2[idx]) for idx in range(len(X2))]

            x1 = np.array(x1)
            x2 = np.array(x2)
            print(x1.shape)

            E, inlier = EstimateE_RANSAC(x1, x2, ransac_n_iter=500, ransac_thr=0.5)

            # Removing Homogenization
            x1 = x1[:, :-1]
            x2 = x2[:, :-1]

            # print(torch.from_numpy(x1[inlier]))

            for inlier_index in inlier:
                print(inlier_index)
                # row, col = np.where(np.isclose(x2, x2[index]))
                print(track_i[i, ind[inlier_index], :])
                print(torch.from_numpy(x1[inlier_index]))
                # track_i[i, ind[inlier_index], :] = torch.from_numpy(x1[inlier_index])
                # track_i[j, ind[inlier_index], :] = torch.from_numpy(x2[inlier_index])
                track_i[i, ind[inlier_index], :] = x1[inlier_index]
                track_i[j, ind[inlier_index], :] = x2[inlier_index]
                print(track_i[i, ind[inlier_index], :])
                print(track_i[j, ind[inlier_index], :])
        #     Points1 = pd.DataFrame(x1[inlier])
        #     Points1.to_csv('Test.csv')
        # print("Current Track_i", track_i)

        # px = pd.DataFrame(track_i)
        # px.to_csv('Test1.csv')
        # pickle.dump(track_i, open('test.pkl', 'wb'))

    return track
