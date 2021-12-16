import os
import cv2
import argparse
import numpy as np

import open3d as o3d
from scipy.interpolate import RectBivariateSpline

from feature_detection import BuildFeatureTrack
from final_camera_pose import EstimateCameraPose
from final_camera_pose import Triangulation
from final_camera_pose import EvaluateCheirality
from perspective_point_algorithm import PnP_RANSAC
from perspective_point_algorithm import PnP_nl
from reconstruction_3D import FindMissingReconstruction
from reconstruction_3D import Triangulation_nl
from reconstruction_3D import RunBundleAdjustment


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='im')
    args = parser.parse_args()

    intrinsic_parameters = np.asarray([
        [3228.13099, 0, 600],
        [0, 3238.72283, 450],
        [0, 0, 1]
    ])

    # Load input images
    List_Img = os.listdir(args.img_dir)
    List_Img.sort()
    Num_Images = len(List_Img)
    img_shape = cv2.imread(os.path.join(args.img_dir, List_Img[0])).shape
    height = img_shape[0]
    width = img_shape[1]

    Images = np.empty((Num_Images, height, width, 3), dtype=np.uint8)
    for i in range(Num_Images):
        im = cv2.imread(os.path.join(args.img_dir, List_Img[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Images[i,:,:,:] = im

    # Build feature track
    track = BuildFeatureTrack(Images, intrinsic_parameters)
    #track = np.load('track.pkl', allow_pickle=True)

    track_first = track[0,:,:]
    track_second = track[1,:,:]

    # Estimate ﬁrst two camera poses
    R, C, X = EstimateCameraPose(track_first, track_second)

    output_dir = 'output'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set of camera poses
    Poses = np.zeros((Num_Images, 3, 4))
    # Set first two camera poses
    Poses[0] = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0]])
    Poses[1] = np.hstack([R, -(R@C).reshape((3,1))])

    ransac_iteration = 1000
    ransac_threshold = 0.5
    for i in range(2, Num_Images):
        # Estimate new camera pose
        X_mask = np.logical_and( np.logical_and(X[:,0]!=-1, X[:,1]!=-1), X[:,2]!=-1)

        track_i = track[i,:,:]
        track_mask_i = np.logical_and(track_i[:,0]!=-1, track_i[:,1]!=-1)

        mask = np.logical_and(X_mask, track_mask_i)
        R, C, inlier = PnP_RANSAC(X[mask], track_i[mask], ransac_iteration, ransac_threshold)
        R, C = PnP_nl(R, C, X[mask], track_i[mask])

        # Add new camera pose to the set
        Poses[i] = np.hstack([R, -(R@C).reshape((3,1))])

        for j in range(i):
            # Fine new points to reconstruct
            track_j = track[j,:,:]
            track_mask_j = np.logical_and(track_j[:,0]!=-1, track_j[:,1]!=-1)

            # get mask for points that exist in both track_i and track_j but not in X
            mask = np.logical_and( np.logical_and(track_mask_i, track_mask_j), ~X_mask)
            # get correspoinding index in X and track
            mask_pos = np.asarray(np.nonzero(mask)[0])

            # Triangulate points
            print('Running linear triangulation between image %d and %d'%(i, j))
            missing_X = Triangulation(Poses[i], Poses[j], track_i[mask], track_j[mask])
            missing_X = Triangulation_nl(missing_X, Poses[i], Poses[j], track_i[mask], track_j[mask])

            # Filter out points based on cheirality
            valid_pos = EvaluateCheirality(Poses[i], Poses[j], missing_X)

            # Update 3D points
            X[mask_pos[valid_pos]] = missing_X[valid_pos]
        
        # Run bundle adjustment
        valid_ps = X[:, 0] != -1
        X_current = X[valid_ps, :]
        track_current = track[:i + 1, valid_ps, :]
        P_latest, X_latest = RunBundleAdjustment(Poses[:i + 1, :, :], X_current, track_current)
        Poses[:i + 1, :, :] = P_latest
        X[valid_ps, :] = X_latest

        # X_latest = X[valid_ps, :]

        ###############################################################
        # Save the camera coordinate frames as meshes for visualization
        m_cam = None
        for j in range(i+1):
            R_d = Poses[j, :, :3]
            C_d = -R_d.T @ Poses[j, :, 3]
            T = np.eye(4)
            T[:3, :3] = R_d.T
            T[:3, 3] = C_d
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(T)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m
        o3d.io.write_triangle_mesh('{}/cameras_{}.ply'.format(output_dir, i+1), m_cam)

        # Save the reconstructed points as point cloud for visualization
        X_latest_h = np.hstack([X_latest, np.ones((X_latest.shape[0],1))])
        pixel_colors = np.zeros_like(X_latest)
        for j in range(i, -1, -1):
            x = X_latest_h @ Poses[j,:,:].T
            x = x / x[:, 2, np.newaxis]
            correct_mask = (x[:,0] >= -1) * (x[:,0] <= 1) * (x[:,1] >= -1) * (x[:,1] <= 1)
            uv = x[correct_mask,:] @ intrinsic_parameters.T
            for k in range(3):
                interp_fun = RectBivariateSpline(np.arange(height), np.arange(width), Images[j,:,:,k].astype(float)/255, kx=1, ky=1)
                pixel_colors[correct_mask, k] = interp_fun(uv[:,1], uv[:,0], grid=False)

        ind = np.sqrt(np.sum(X_current ** 2, axis=1)) < 200
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_latest[ind]))
        pcd.colors = o3d.utility.Vector3dVector(pixel_colors[ind])
        o3d.io.write_point_cloud('{}/points_{}.ply'.format(output_dir, i+1), pcd)