# Visual Odometry

### Requirements
- numpy
- pandas
- opencv-contrib-python
- [open3d](https://github.com/isl-org/Open3D)
- scipy


### Data
The example data can be found in the `video` folder. You can use your own video for testing.
```bash
-- video
  |-- chessboard.MOV
  |-- walking.mp4
```

### Usage
- Step 1: Camera Calibration
```bash
python CameraCalibration.py --input PATH_TO_CHESSBOARD_VIDEO \ 
                            --output SAVE_PATH_FOR_CAMERA_PARAMS \
                            --w CHESSBOARD_WIDTH \
                            --h CHESSBOARD_HEIGHT \
                            [--show] (Flag to show the camera poses)

# E.g., python CameraCalibration.py --input ./video/chessboard.MOV --output camera_params.npy --w 9 --h 6 --show
```
- Step 2: Visual Odometry
```bash
python VisualOdometry.py --input PATH_TO_VIDEO_OR_IMAGE_FOLDER \ 
                        --camera_parameters PATH_TO_CAMERA_PARAMS

# E.g., python VisualOdometry.py --input ./video/walking.mp4 --camera_parameters camera_params.npy
```

### Files
- `CameraCalibration.py`: Camera calibration using a chessboard video.
- `FeatureMatcher.py`: Feature detection and matching using SIFT or ORB.
- `PointCloud.py`: Point cloud management.
- `VisualOdometry.py`: Main visual odometry pipeline.

### Result
The gif below only shows a part of the result. The full result can be found on [YouTube](https://www.youtube.com/playlist?list=PLSIKdwlRdCdpnYpHhOrUgJNaXJZIai6CE).
![Demo](./demo.gif)