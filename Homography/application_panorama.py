
import numpy as np
import cv2


from homography import RANSAC
from warping import warping


def panorama(imgs):
    h_max = imgs[0].shape[0]
    w_max = sum([x.shape[1] for x in imgs])


    # Create the canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)

    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # Feature Detection & Matching (The whole process can be replaced by any homography estimation method)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY),None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY),None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1,des2)
        
        match_point = []
        for match in matches:
            match_point.append([list(kp2[match.trainIdx].pt) + list(kp1[match.queryIdx].pt)])
        match_point = np.array(match_point).squeeze()

        # Apply RANSAC to choose best H
        best_H = RANSAC(match_point)

        # Update last H
        last_best_H = np.matmul(last_best_H, best_H)

        # Apply warping
        dst = warping(im2, dst, last_best_H, direction='b')

    # Post-processing: crop the black area
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    dst = dst[y+20:y+h-50, x+10:x+w-10]

    return dst




if __name__ == "__main__":
    imgs = [cv2.imread('./data/frame{:d}.jpg'.format(x)) for x in range(1, 4)]
    output = panorama(imgs)
    cv2.imwrite('./output/panorama.png', output)
    cv2.imshow('panorama', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()