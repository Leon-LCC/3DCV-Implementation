import numpy as np
import cv2
from cv2 import aruco
from tqdm import tqdm

from homography import estimate_homography
from warping import warping
    
def main(image_path, video_path):
    # Initialize video capture and writer
    video = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    videowriter = cv2.VideoWriter("./output/ArUco_output.mp4", fourcc, video.get(cv2.CAP_PROP_FPS), (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Initialize ArUco marker detection
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParameters)

    # Load source image
    sImg = cv2.imread(image_path)
    h, w, c = sImg.shape
    src_corns = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    # Find homography per frame
    pbar = tqdm(total = int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    while (video.isOpened()):
        ret, frame = video.read()
        if ret:  
            corners, ids, rejected = detector.detectMarkers(frame)

            # Find homography
            H = estimate_homography(src_corns, np.array(corners[0]).squeeze())

            # Apply backward warp
            warping(sImg, frame, H, direction='b')

            videowriter.write(frame)
            pbar.update(1)

        else:
            break

    pbar.close()
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main('./data/cat.jpg', './data/ArUco.mp4')