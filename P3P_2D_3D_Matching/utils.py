import os
import pandas as pd
import numpy as np
import sqlite3



def get_descriptors_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Get all descriptors from the database
    cursor.execute("SELECT image_id, data FROM descriptors")
    rows = cursor.fetchall()

    descriptors_by_image = {}
    for image_id, data in rows:
        desc = np.frombuffer(data, dtype=np.uint8)
        desc = desc.reshape(-1, 128)  # 128-d SIFT descriptors
        descriptors_by_image[image_id] = desc

    conn.close()

    return descriptors_by_image


def read_pointcloud(PC_path, descriptors_by_image):
    # Load point cloud from .txt file
    with open(PC_path, 'r') as f:
        lines = f.readlines()
    
    # Parse lines
    points3D = []
    for line in lines:
        if line[0] == '#':
            continue
        elems = line.split(' ')
        point_id = int(elems[0])
        x, y, z = float(elems[1]), float(elems[2]), float(elems[3])
        r, g, b = int(elems[4]), int(elems[5]), int(elems[6])

        # Get descriptors for this point
        descriptors = []
        for i in range(8, len(elems), 2):
            image_id = int(elems[i])
            feature_idx = int(elems[i+1])
            if image_id in descriptors_by_image:
                descriptors.append(descriptors_by_image[image_id][feature_idx])

        # Average descriptors
        descriptors = np.mean(descriptors, axis=0).astype(np.uint8)

        points3D.append([point_id, [x, y, z], [r, g, b], descriptors])

    # To DataFrame
    points3D_df = pd.DataFrame(points3D, columns=["POINT_ID","XYZ","RGB","DESCRIPTORS"])
    points3D_df["POINT_ID"] = points3D_df["POINT_ID"].astype(int)

    return points3D_df


def read_camera(cameras_path):
    # Load camera parameters from cameras.txt
    with open(cameras_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line[0] == '#':
            continue
        elems = line.split(' ')
        width, height = int(elems[2]), int(elems[3])
        params = list(map(float, elems[4:]))

    cameraMatrix = np.array([[params[0], 0, params[2]],
                             [0, params[1], params[3]],
                             [0, 0, 1]]).astype(np.float64)
    distCoeffs = np.array(params[4:]).astype(np.float64)

    return cameraMatrix, distCoeffs, width, height


def get_query_images(images_path, db_path):
    # Load image list from images.txt
    with open(images_path, 'r') as f:
        lines = f.readlines()

    # Load descriptors from database
    descriptors_by_image = get_descriptors_from_db(db_path)

    image_list = []
    for i in range(0, len(lines), 2):
        if lines[i][0] == '#':
            continue
        
        image_id = int(lines[i].split(' ')[0])
        image_name = lines[i].split(' ')[-1]
        if not image_name.startswith('val'):
            continue
        
        QX, QY, QZ, QW = map(float, lines[i].split(' ')[1:5])
        TX, TY, TZ = map(float, lines[i].split(' ')[5:8])
        points = lines[i+1].split(' ')
        points = [([float(points[j]), float(points[j+1])]) for j in range(0, len(points), 3)]
        image_list.append((image_id, points, descriptors_by_image.get(image_id, None), QX, QY, QZ, QW, TX, TY, TZ))

    images_df = pd.DataFrame(image_list, columns=["IMAGE_ID","XY","DESCRIPTORS","QX","QY","QZ","QW","TX","TY","TZ"])
    images_df["IMAGE_ID"] = images_df["IMAGE_ID"].astype(int)
    images_df["XY"] = images_df["XY"].apply(lambda x: np.array(x).astype(float))
    images_df["DESCRIPTORS"] = images_df["DESCRIPTORS"].apply(lambda x: np.array(x).astype(int) if x is not None else None)
    images_df["QX"] = images_df["QX"].astype(float)
    images_df["QY"] = images_df["QY"].astype(float)
    images_df["QZ"] = images_df["QZ"].astype(float)
    images_df["QW"] = images_df["QW"].astype(float)
    images_df["TX"] = images_df["TX"].astype(float)
    images_df["TY"] = images_df["TY"].astype(float)
    images_df["TZ"] = images_df["TZ"].astype(float)

    return images_df



if __name__ == "__main__":
    PC_path = "./data/sparse/points3D.txt"
    db_path = "./data/sparse/database.db"
    images_path = "./data/sparse/images.txt"
    camera_params_path = "./data/sparse/cameras.txt"

    # Get descriptors from database
    descriptors_by_image = get_descriptors_from_db(db_path)
    print(descriptors_by_image[list(descriptors_by_image.keys())[0]].shape)

    # Read point cloud and attach descriptors
    points3D_df = read_pointcloud(PC_path, descriptors_by_image)
    print(points3D_df.head())

    # Get camera parameters
    cameraMatrix, distCoeffs, w, h = read_camera(camera_params_path)
    print("Camera Matrix:\n", cameraMatrix)
    print("Distortion Coefficients:\n", distCoeffs)
    print("Image Width:", w)
    print("Image Height:", h)

    # Get query images
    query_images = get_query_images(images_path, db_path)
    print(query_images.head())
    print(query_images.iloc[0]["XY"].shape)
    print(query_images.iloc[0]["DESCRIPTORS"].shape if query_images.iloc[0]["DESCRIPTORS"] is not None else None)
    print(query_images.iloc[0][["QX","QY","QZ","QW","TX","TY","TZ"]])
    

