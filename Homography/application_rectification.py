import numpy as np
import cv2 as cv
import argparse

from homography import estimate_homography
from warping import warping

def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])  

def markPoints(img):
    sPoints= []
    cv.namedWindow('Select Corner', cv.WINDOW_NORMAL)
    cv.setMouseCallback('Select Corner', on_mouse, [sPoints])

    while True:
        img_ = img.copy()
        for i, p in enumerate(sPoints):
            # draw points on img_
            cv.circle(img_, tuple(p), 2, (0, 255, 0), -1)
        cv.imshow('Select Corner', img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27 or len(sPoints) == 4: break # exist when pressing ESC

    cv.destroyAllWindows()
    print('{} Points added'.format(len(sPoints)))

    return np.array(sPoints)


def main(args):
    sImg = cv.imread(args.img_path)
    dWidth = args.width
    dHeight = args.height

    # Manual mark the points
    sPoints = markPoints(sImg)
    dPoints = np.array([[0, 0],
                        [dWidth, 0],
                        [dWidth, dHeight],
                        [0, dHeight]])


    # Compute homography
    H = estimate_homography(sPoints, dPoints)

    # Destination image
    dImg = np.zeros((dHeight, dWidth, 3), dtype=np.uint8)

    # Warp image
    dImg = warping(sImg, dImg, H, direction=args.warp_direction)

    # Show the result and save it
    cv.imwrite(args.save_path, dImg)
    cv.imshow('rectified', dImg)
    cv.waitKey(0)
    cv.destroyAllWindows()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./data/textbook.jpeg')
    parser.add_argument('--save_path', type=str, default='./output/rectified.jpg')
    parser.add_argument('--width', type=int, default=380) #Keep the ratio of the image. h:26, w:19
    parser.add_argument('--height', type=int, default=520) #Keep the ratio of the image. h:26, w:19
    parser.add_argument('--warp_direction', type=str, default='b', choices=['b', 'f'])
    args = parser.parse_args()

    main(args)