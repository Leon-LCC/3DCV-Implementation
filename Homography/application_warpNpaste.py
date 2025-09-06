# Put your Ad at Times Square
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
    print('Top-Left Point: {}'.format(sPoints[0]))
    print('Top-Right Point: {}'.format(sPoints[1]))
    print('Bottom-Right Point: {}'.format(sPoints[2]))
    print('Bottom-Left Point: {}'.format(sPoints[3]))

    return np.array(sPoints)


def main(args):
    # Source image: Ad
    sImg = cv.imread(args.img_path)
    sWidth, sHeight = sImg.shape[1], sImg.shape[0]

    # Destination image: Times Square photo
    dImg = cv.imread('./data/times.png')

    # Manual mark the points (where the ad should be placed, must be a planar surface in the scene)
    sPoints = np.array([[0, 0],
                        [sWidth, 0],
                        [sWidth, sHeight],
                        [0, sHeight]])
    if args.manual:
        dPoints = markPoints(dImg)
    else:
        dPoints = np.array([[984, 751],
                            [1540, 1015],
                            [1467, 1640],
                            [834, 1389]])

    # Compute homography
    H = estimate_homography(sPoints, dPoints)

    # Warp image
    warpedImg = warping(sImg, dImg, H, direction=args.warp_direction)


    # Show the result and save it
    cv.imwrite(args.save_path, warpedImg)
    cv.imshow('Result', warpedImg)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./data/ad.jpg')
    parser.add_argument('--save_path', type=str, default='./output/timesWithAd.jpg')
    parser.add_argument('--manual', action='store_true')
    parser.add_argument('--warp_direction', type=str, default='b', choices=['b', 'f'])
    args = parser.parse_args()

    main(args)