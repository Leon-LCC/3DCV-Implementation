from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

from utils import read_pointcloud, get_descriptors_from_db, read_camera, get_query_images
from PNPSolver import P3PRansac


# Median pose error (N,4),(N,3)
def median_pose_error(rotq, tvec, rotq_gt, tvec_gt):
    # Translation (N,3)
    t_err = np.linalg.norm((tvec-tvec_gt).astype(float), axis=1)
    # Rotation (N,4)
    rotq = R.from_quat(rotq)
    rotq_gt = R.from_quat(rotq_gt)
    rot_rel = rotq_gt * rotq.inv()
    # Axis angle representation
    rot_rel = rot_rel.as_rotvec()
    r_err = np.linalg.norm(rot_rel, axis=1)
    return np.median(r_err), np.median(t_err)


# 2D-3D matching
def pnpsolver(query, model, cameraMatrix, distCoeffs, solver='Fis_Arun'):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    if solver == 'cv2':
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)
        return rvec, tvec
    elif solver == 'Fis_Arun' or solver == 'Fis_tril' or solver == 'Grun_Arun' or solver == 'Grun_tril':
        solver = solver.split('_')
        return P3PRansac(points3D, points2D, cameraMatrix, distCoeffs, solver[0], solver[1])
    else:
        raise ValueError('Unknown solver')


# Draw Points Cloud
def creatCamera(rotq, tvec, w, h, cameraMatrix, distCoeffs): #Quaternion Rotation and Translation Vector
    rotq = R.from_quat(rotq).as_matrix()
    # Corners on Image Plane
    vertice = np.array([[0, 0],
                        [0, h],
                        [w, h],
                        [w, 0]]).astype(float)

    # Transform to Camera Coordinate
    vertice = cv2.undistortPoints(vertice, cameraMatrix, distCoeffs)
    vertice = np.concatenate((vertice.reshape(-1,2), np.ones((4,1))), axis=1)
    # Add Camera Center
    vertice = np.concatenate((np.array([[0,0,0]]),vertice), axis=0)

    # Transform to World Coordinate
    world_cordinate = np.linalg.inv(rotq) @ (vertice.T - tvec.T)

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
    mesh.paint_uniform_color([0.9, 0.1, 0.1])

    return line_set, mesh, world_cordinate.T.tolist()[0]


def main():
    # Load Data
    PC_path = "./data/sparse/points3D.txt"
    db_path = "./data/sparse/database.db"
    images_path = "./data/sparse/images.txt"
    camera_params_path = "./data/sparse/cameras.txt"
    solver = 'Fis_Arun'  # 'cv2', 'Fis_Arun', 'Fis_tril', 'Grun_Arun', 'Grun_tril'

    # Get descriptors from database
    descriptors_by_image = get_descriptors_from_db(db_path)
    
    # Get point cloud with descriptors
    points3D_df = read_pointcloud(PC_path, descriptors_by_image)

    # Camera Parameters
    cameraMatrix, distCoeffs, w, h = read_camera(camera_params_path)

    # Process model descriptors
    kp_model = np.array(points3D_df["XYZ"].to_list())
    desc_model = np.array(points3D_df["DESCRIPTORS"].to_list()).astype(np.float32)
    
    # Query Images
    query_images = get_query_images(images_path, db_path)

    rotqVal, tvecVal = [], []
    rotq_gt, tvec_gt = [], []
    for i in tqdm(range(len(query_images))):
        # Load query keypoints and descriptors
        kp_query = np.array(query_images.loc[i]["XY"])
        desc_query = np.array(query_images.loc[i]["DESCRIPTORS"]).astype(np.float32)

        # Find correspondance and solve pnp
        rvec, tvec = pnpsolver((kp_query, desc_query),(kp_model, desc_model), cameraMatrix, distCoeffs, solver=solver)
        rotqVal.append(R.from_rotvec(rvec.reshape(1,3)).as_quat())
        tvecVal.append(tvec.reshape(1,3))

        # Get camera pose groundtruth
        rotq_gt.append(query_images.loc[i][["QX","QY","QZ","QW"]].to_numpy())
        tvec_gt.append(query_images.loc[i][["TX","TY","TZ"]].to_numpy())

    # Compute Median Pose Error
    r_err, t_err = median_pose_error(np.vstack(rotqVal), np.vstack(tvecVal), np.vstack(rotq_gt), np.vstack(tvec_gt))
    print("Median Pose Error | ", "Rotaion:", np.median(r_err), "Translation:", np.median(t_err))
    


    # Visualizations
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Draw Points Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D_df['XYZ'].to_list())
    pcd.colors = o3d.utility.Vector3dVector(np.array(points3D_df['RGB'].to_list())/255.0)
    vis.add_geometry(pcd)
    # Draw Camera
    cameraCenter = []
    for i in range(len(rotqVal)):
        cameraInfo = creatCamera(rotqVal[i], tvecVal[i], w, h, cameraMatrix, distCoeffs)
        vis.add_geometry(cameraInfo[0])
        vis.add_geometry(cameraInfo[1])
        cameraCenter.append(cameraInfo[2])
    # Draw Camera Trajectory
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(cameraCenter)
    line_set.lines = o3d.utility.Vector2iVector([[i,i+1] for i in range(len(line_set.points)-1)])
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(len(line_set.lines))])
    vis.add_geometry(line_set)
    # Set View Pose
    o3d.visualization.ViewControl.set_front(vis.get_view_control(), [0.23833787489693181, -0.69576063439641966, -0.67757818516677926])
    o3d.visualization.ViewControl.set_lookat(vis.get_view_control(), [0.52527069999999998, -0.5, 1.4795769999999999])
    o3d.visualization.ViewControl.set_up(vis.get_view_control(), [-0.079073005436627597, -0.70926954959991573, 0.70048851940738599])
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.080000000000000002)
    vis.run()



if __name__ == "__main__":
    main()