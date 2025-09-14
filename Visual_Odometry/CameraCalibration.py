import cv2
import numpy as np
import argparse

import open3d as o3d


class CameraCalibrator:
    def __init__(self, config):
        self.config = config
        self.cols = config.w
        self.rows = config.h

        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Generate object points grid: (0,0,0), (1,0,0), ..., (cols-1, rows-1, 0)
        self.object_points_template = np.zeros((self.cols * self.rows, 3), np.float32)
        self.object_points_template[:, :2] = np.mgrid[0:self.rows, 0:self.cols].T.reshape(-1, 2)

        self.all_obj_points = []
        self.all_img_points = []
        self.saved_frames = []

    def execute(self):
        self.cap = cv2.VideoCapture(self.config.input)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.saved_frames.append(frame.copy())

        # Randomly sample 100 frames if too many
        if len(self.saved_frames) > 100:
            self.saved_frames = np.array(self.saved_frames)
            self.saved_frames = list(self.saved_frames[np.random.choice(len(self.saved_frames), 100, replace=False)])
        assert len(self.saved_frames) >= 4, "At least 4 images are required for calibration."

        rms, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = self.perform_calibration()
        print(f"Re-projection error (RMS): {rms:.5f}")
        print("Camera matrix:\n", self.camera_matrix)
        print("Distortion coefficients:\n", self.dist_coeffs)
        np.save(self.config.output, {'K': self.camera_matrix, 'dist': self.dist_coeffs})

        if self.config.show:
            self.visualize_calibration()

    def perform_calibration(self):
        last_gray_shape = None
        for frame in self.saved_frames:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            last_gray_shape = gray_frame.shape[::-1]

            found, corners = cv2.findChessboardCorners(gray_frame, (self.rows, self.cols), None)
            if found:
                self.all_obj_points.append(self.object_points_template)
                refined_corners = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), self.criteria)
                self.all_img_points.append(refined_corners)

        if len(self.all_obj_points) < 4:
            raise RuntimeError("Not enough corner-detected images for calibration.")

        return cv2.calibrateCamera(self.all_obj_points, self.all_img_points, last_gray_shape, None, None)

    def visualize_calibration(self):
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Draw chessboard squares
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                color = [0, 0, 0] if (i + j) % 2 == 0 else [0.9, 0.9, 0.9]
                box = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.2)
                box.translate((i, j, -0.2))
                box.paint_uniform_color(color)
                vis.add_geometry(box)

        # Draw camera positions
        for rvec, tvec in zip(self.rvecs, self.tvecs):
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
            frustum, mesh = self.createCamera(R, t)
            vis.add_geometry(frustum)
            vis.add_geometry(mesh)

        vis.run()

    def createCamera(self, R, t, scale=1.0):
        # Camera pyramid vertices in camera frame
        vertices = np.array([
            [0, 0, 0],          # Camera center
            [0, 0, scale],      # Top-left
            [scale, 0, scale],  # Top-right
            [scale, scale, scale],  # Bottom-right
            [0, scale, scale]       # Bottom-left
        ])

        # Apply rotation and translation
        R_w = R.T
        t_w = -R.T @ t
        vertices = (R_w @ vertices.T).T + t_w.T

        # Define lines connecting vertices
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # Camera center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Image plane edges
        ]
        colors = [[1, 0, 0] for _ in lines]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Optional image plane mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices[1:])  # 4 corners
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.8, 0.8])

        return line_set, mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to input video for calibration")
    parser.add_argument("--output", default="camera_params.npy", help="File to save camera parameters")
    parser.add_argument("--w", type=int, default=9, help="Chessboard inner corner width")
    parser.add_argument("--h", type=int, default=6, help="Chessboard inner corner height")
    parser.add_argument("--show", action="store_true", help="Show 3D visualization (Open3D required)")

    cfg = parser.parse_args()
    calibrator = CameraCalibrator(cfg)
    calibrator.execute()