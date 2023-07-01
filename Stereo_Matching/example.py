import numpy as np
import cv2

import argparse
import os

from computeDisp import computeDisp


def evaluate(disp_input, disp_gt, scale_factor, threshold=1.0):
    disp_input = np.uint8(disp_input * scale_factor)
    disp_input = np.int32(disp_input / scale_factor)
    disp_gt = np.int32(disp_gt / scale_factor)

    nr_pixel = 0
    nr_error = 0
    h, w = disp_gt.shape
    for y in range(0, h):
        for x in range(0, w):
            if disp_gt[y, x] > 0:
                nr_pixel += 1
                if np.abs(disp_gt[y, x] - disp_input[y, x]) > threshold:
                    nr_error += 1

    return float(nr_error)/nr_pixel



def main(args):
    # Load image
    img_left = cv2.imread(os.path.join(args.img_dir, 'img_left.png'))
    img_right = cv2.imread(os.path.join(args.img_dir, 'img_right.png'))
    
    # Compute disparity
    labels = computeDisp(img_left, img_right, args.max_disp)

    # Save disparity
    if args.save_path is None:
        save_path = os.path.join(args.img_dir, 'disp.png')
    else:
        save_path = args.save_path

    cv2.imwrite(save_path, np.uint8(labels * args.scale_factor))

    # Evaluate
    if args.gt_path is not None:
        img_gt = cv2.imread(args.gt_path, -1)
        error = evaluate(labels, img_gt, args.scale_factor)
        print('[Bad Pixel Ratio] %.2f%%' % (error*100))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stereo Matching')
    parser.add_argument('--img_dir', required=True, help='input image directory')
    parser.add_argument('--save_path', default=None, help='save path')
    parser.add_argument('--gt_path', default=None, help='ground truth path')
    parser.add_argument('--max_disp', default=60, type=int, help='maximum disparity')
    parser.add_argument('--scale_factor', default=4, type=int, help='scale factor')
    args = parser.parse_args()
    main(args)