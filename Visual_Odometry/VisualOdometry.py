import open3d as o3d
import numpy as np
import cv2
import sys, os, argparse, glob
import multiprocessing as mp
from FeatureMatcher import FeatureMatcher
from pointCloud import PointCloud
from scipy.optimize import least_squares
import shutil

feature_matcher = FeatureMatcher('orb')
point_cloud = PointCloud()

class Visual_Odometry:
    def __init__(self, args):
        # Camera intrinsics
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']

        if args.input.endswith('.mp4') or args.input.endswith('.avi') or args.input.endswith('.mov') or args.input.endswith('.MOV'):
            # Video input
            cap = cv2.VideoCapture(args.input)
            self.frame_paths = []
            idx = 0
            os.makedirs('./temp_frames', exist_ok=True)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                path = f'./temp_frames/temp_frame_{idx:04d}.png'
                cv2.imwrite(path, frame)
                self.frame_paths.append(path)
                idx += 1
        else:
            # All images paths
            self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))
 
        # Image size
        img = cv2.imread(self.frame_paths[0])
        self.w = img.shape[1]
        self.h = img.shape[0]

    def construct3DPoints(self, feature1, feature2, pose1, pose2): # Triangulate
        pts1, pts2, des1, des2 = feature_matcher.feature_matching(feature1[0], feature1[1], feature2[0], feature2[1], return_des=True)
        pts1 = cv2.undistortPoints(pts1.reshape(-1, pts1.shape[0], 2), self.K, self.dist, P=self.K).reshape(-1, 2)
        pts2 = cv2.undistortPoints(pts2.reshape(-1, pts2.shape[0], 2), self.K, self.dist, P=self.K).reshape(-1, 2)
        pts3D = cv2.triangulatePoints(self.K@pose1, self.K@pose2, pts1.T, pts2.T).T
        pts3D = pts3D[:,:3] / pts3D[:,3].reshape(-1, 1)
        return pts3D, pts1, pts2, des1, des2

    def getRelativeScale(self, R, t, R_so_far, t_so_far, pts3D, pts2D):
        pts2D = cv2.undistortPoints(pts2D.reshape(-1, pts2D.shape[0], 2), self.K, self.dist, P=self.K).reshape(-1, 2)
        def fun(s):
            return np.linalg.norm(pts2D - cv2.projectPoints(pts3D, R@R_so_far, R@t_so_far + t*s, self.K, self.dist)[0].reshape(-1, 2), axis=1)
        res = least_squares(fun, 1, bounds=(0, 5))
        return res.x[0]

    def getRelativePose(self, pts1, pts2):
        # Undistort points
        pts1 = cv2.undistortPoints(pts1.reshape(-1, pts1.shape[0], 2), self.K, self.dist, P=self.K).reshape(-1, 2)
        pts2 = cv2.undistortPoints(pts2.reshape(-1, pts2.shape[0], 2), self.K, self.dist, P=self.K).reshape(-1, 2)
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, cv2.RANSAC, 0.999, 1.5)
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        return R, t

    def creatCamera(self, R, t):
        # Corners on Image Plane
        vertice = np.array([[0,           0],
                            [0,      self.h],
                            [self.w, self.h],
                            [self.w,      0]]).astype(float)

        # Transform to Camera Coordinate
        vertice = cv2.undistortPoints(vertice, self.K, self.dist)
        vertice = np.concatenate((vertice.reshape(-1,2), np.ones((4,1))), axis=1)
        # Add Camera Center
        vertice = np.concatenate((np.array([[0,0,0]]),vertice), axis=0)
        # Transform to World Coordinate
        world_cordinate = R.T @ (vertice.T - t)
        # Draw Pyramid
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(world_cordinate.T.squeeze().tolist())
        line_set.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(len(line_set.lines))])
        # Draw Image Plane
        mesh  = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(world_cordinate.T.tolist()[1:])
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])
        mesh.compute_vertex_normals()
        return line_set, mesh, world_cordinate.T.tolist()[0]

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        o3d.visualization.ViewControl.set_lookat(vis.get_view_control(), [0, 0, 1])
        o3d.visualization.ViewControl.set_up(vis.get_view_control(), [0, 1, 0])

        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    cameraInfo = self.creatCamera(R, t)
                    vis.add_geometry(cameraInfo[0])
                    vis.add_geometry(cameraInfo[1])
            except:
                pass

            keep_running = keep_running and vis.poll_events()

        vis.destroy_window()
        p.join()


    def process_frames(self, queue):
        # Initialize
        ## First frame & Second frame
        img_0 = cv2.imread(self.frame_paths[0])
        img_1 = cv2.imread(self.frame_paths[1])
        ## Feature matching
        pts0, pts1, feature0, feature1 = feature_matcher.feature_matching_image_pair(img_0, img_1, return_feature=True)
        ## Find relative pose
        R_so_far, t_so_far = self.getRelativePose(pts0, pts1)
        ## Point cloud
        pts3D, pts2D_0, pts2D_1, des0, des1 = self.construct3DPoints(feature0, feature1, np.concatenate((np.eye(3), np.zeros((3,1))), axis=1), np.concatenate((R_so_far, t_so_far), axis=1))
        point_cloud.addPoints(0, pts2D_0, des0, pts3D)
        point_cloud.addPoints(1, pts2D_1, des1, pts3D)

        # Loop
        for idx, frame_path in enumerate(self.frame_paths[2:], start=2):
            img = cv2.imread(frame_path)
            # Feature matching
            kp2, des2 = feature_matcher.get_keypoints(img)
            pts1, pts2 = feature_matcher.feature_matching(feature1[0], feature1[1], kp2, des2)
            # Find relative pose
            R, t = self.getRelativePose(pts1, pts2)
            # Triangulate points to get relative scale
            if idx > 2:
                pts3D, des = point_cloud.findImgPoints([idx-3, idx-2, idx-1, idx])
            else:
                pts3D, des = point_cloud.findImgPoints([idx-2, idx-1])
            pts3D, pts2D = feature_matcher.feature_matching3D(pts3D, des, kp2, des2)
            scale = self.getRelativeScale(R, t, R_so_far, t_so_far, pts3D, pts2D)
            # Absolute pose
            bufR, buft = R_so_far, t_so_far
            R_so_far, t_so_far = R @ R_so_far, R @ t_so_far + t*scale
            queue.put((R_so_far, t_so_far))

            # Update state
            pts3D, _, pts2D, _, des1 = self.construct3DPoints(feature1, (kp2, des2), np.concatenate((bufR, buft), axis=1), np.concatenate((R_so_far, t_so_far), axis=1))
            point_cloud.addPoints(idx, pts2D, des1, pts3D)
            feature1 = (kp2, des2)
            
            # Draw KeyPoints
            img = cv2.drawKeypoints(img, cv2.KeyPoint_convert(pts2D), None, color=(0,255,0), flags=0)
            cv2.imshow('frame', img)
            if cv2.waitKey(30) == 27: break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_params.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vis_od = Visual_Odometry(args)
    vis_od.run()
    shutil.rmtree('./temp_frames')
