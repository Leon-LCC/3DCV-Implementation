# P3P 2D-3D Matching


## Quick Start

##### Requirements
- numpy
- pandas
- scipy
- opencv-python
- open3d
- tqdm
- sqlite3

##### Usage
The p3p solver can be found in `PNPSolver.py` . There are three functions:
- `solveRT`: Given 2D-3D correspondences and distance, solve the rotation and translation. You can choose between two methods: [Arun's method](https://jingnanshi.com/blog/arun_method_for_3d_reg.html) and [Trilateration method](https://en.wikipedia.org/wiki/Trilateration).
```python
def solveRT(pts2D, pts3D, distances, method="Arun"):
    # Implementation
    return R, t
```
- `solvep3p`: Given 4 2D-3D correspondences, solve the possible camera poses using the [P3P algorithm](https://en.wikipedia.org/wiki/Perspective-n-Point). Fischler and Bolles's method and Grunert’s method are implemented.
```python
def solvep3p(points3D, points2D, cameraMatrix, distCoeffs, method='Fis', RTsolver='Arun'):
    # Implementation
    return best_R, best_t
```
- `P3PRansac`: use RANSAC to robustly estimate the camera pose from 2D-3D correspondences. The function internally calls `solvep3p` to generate possible solutions.
```python
def P3PRansac(points3D, points2D, cameraMatrix, distCoeffs, method='Grun', RTsolver='Arun'):
    # Implementation
    return best_R, best_t
```

---

## Application 
##### Introduction
- Given a 3D point cloud, the P3P algorithm can be used to estimate the camera pose of any 2D image that captures the same scene. The 2D–3D correspondences are established by matching SIFT features between the input image and the 3D point cloud.

##### Dataset
- The following dataset is generated with [COLMAP](https://colmap.github.io/index.html). Images with the prefix val- serve as query images for testing the P3P solver. The dataset structure is organized as follows:
```bash
-- data
  |-- sparse
  |  |-- cameras.txt
  |  |-- database.db
  |  |-- images.txt
  |  |-- points3D.txt
  |
  |-- train-000000.png
  |-- train-000001.png
  |-- ...
  |-- val-000000.png
  |-- val-000001.png
  |-- ...
```
- Download the `database.db` file from [here](https://github.com/Leon-LCC/3DCV-Implementation/releases/tag/p3p-v0) and place it in the `data/sparse` folder.

- You can also create your own dataset using COLMAP. Run the following command to reconstruct a sparse point cloud from a set of input images. Once reconstruction is complete, organize the generated files to match the directory structure above.
```bash
colmap automatic_reconstructor --workspace_path PATH_TO_WORKSPACE --image_path PATH_TO_IMAGES --camera_model OPENCV --single_camera 1
# e.g., colmap automatic_reconstructor --workspace_path ./data --image_path ./data/images --camera_model OPENCV --single_camera 1
```

##### Usage
```bash
python application_findCameraPose.py
```
- All the paths and parameters are hardcoded in the python script.
    - PC_path: path to the 3D point cloud file (points3D.txt)
    - db_path: path to the database file (database.db)
    - images_path: path to the images file (images.txt)
    - camera_params_path: path to the camera parameters file (cameras.txt)
    - solver: choose the P3P solver and RT solver. Options are 'cv2', 'Fis_Arun', 'Fis_tril', 'Grun_Arun', 'Grun_tril'. 'cv2' uses OpenCV's built-in PnP solver.


##### Output
![output](./result.gif)