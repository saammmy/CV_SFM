import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors


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
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    """

    # TODO Your code goes here
    # Nearest neighbor distance ratio:
    NNDR = 0.7
    nbrs = NearestNeighbors(n_neighbors=2).fit(des2)
    distances, indices = nbrs.kneighbors(des1, 2, return_distance=True)
    print(len(distances))
    print(indices, distances)
    pass


def EstimateH(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the homography between images using RANSAC
    
    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """

    # TODO Your code goes here
    pass


def EstimateR(H, K):
    """
    Compute the relative rotation matrix
    
    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """

    # TODO Your code goes here
    pass


def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface
    
    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Hc, Hc, 3)
        The 3D points corresponding to all pixels in the canvas
    """

    # TODO Your code goes here
    pass


def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane
    
    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """

    # TODO Your code goes here
    pass


def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas
    
    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """

    # TODO Your code goes here
    pass


def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image
    
    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """

    # TODO Your code goes here
    pass


if __name__ == '__main__':
    ransac_n_iter = 500
    ransac_thr = 3
    K = np.asarray([
        [320, 0, 480],
        [0, 320, 270],
        [0, 0, 1]
    ])

    # Read all images
    imList = []
    for i in range(1, 9):
        im_file = '{}.jpg'.format(i)
        im = cv.imread(im_file)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        imList.append(im)

    rot_list = []
    rot_list.append(np.eye(3))
    for i in range(len(imList) - 1):
        # Load consecutive images I_i and I_{i+1}
        # TODO Your code goes here
        img1, img2 = imList[i], imList[i + 1]

        # Extract SIFT features
        # TODO Your code goes here
        grayImg1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        sift1 = cv.SIFT_create()
        loc1, des1 = sift1.detectAndCompute(grayImg1, None)

        grayImg2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        sift2 = cv.SIFT_create()
        loc2, des2 = sift2.detectAndCompute(grayImg2, None)

        print(loc2, des2)

        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2)

        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)

        # Compute the relative rotation matrix R
        R = EstimateR(H, K)

        # Compute R_new (or R_i+1)
        # TODO Your code goes here

        rot_list.append(R_new)

    Him = imList[0].shape[0]
    Wim = imList[0].shape[1]

    Hc = Him
    Wc = len(imList) * Wim // 2

    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)

    fig = plt.figure('HW1')
    plt.axis('off')
    plt.ion()
    plt.show()
    for i, (im_i, rot_i) in enumerate(zip(imList, rot_list)):
        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)
        # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)
        # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)
        plt.imshow(canvas)
        plt.savefig('output_{}.png'.format(i + 1), dpi=600, bbox_inches='tight', pad_inches=0)
