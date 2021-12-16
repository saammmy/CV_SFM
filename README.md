# SfM
Name: Structure from Motion
Team: Samarth Shah, Aditya Mehrotra, Anujay Sharma, Aditya Patil

This project follows the traditional pipeline to perform incremental Structure from Motion (SfM).

To run this code, execute the run.py file, ensure there are 6 images in "im" folder.
Also if you are using your personal dataset please perform the camera callibration to obtain the intrinsic parameters and change this in run.py.

Running the run.py file will create camera and points file in output for 3,4,5,6 cameras.

To visualise each of the point cloud download the meshlab application from this link:
https://www.meshlab.net/#download

This code has the following functions:

feature_detection.py: It perform feature extraction and matching. Finally it develops a feature track that contains corresponding matches.
final_camera_pose.py: It estimates the camera poses for all images using Chierality condition.
perspective_point_algorithm: It minimizes the reprojection error to refine camera poses
3d_reconstruction: Its an optimizer for minimizing the reprojection error through non linear techniques while accounting for camera poses and 3d point triangulations

Future Work:
Make a more dense point cloud
