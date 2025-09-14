import cv2
import numpy as np

class FeatureMatcher:
    def __init__(self, detector_type='sift'):
        if detector_type == 'sift':
            self.detector = cv2.SIFT_create()
            self.type = 'sift'
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
        else:
            self.detector = cv2.ORB_create(nfeatures=5000)
            self.type = 'orb'
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def feature_matching(self, kp1, des1, kp2, des2, return_des=False):
        if self.type  == 'sift': #SIFT feature matching
            # KNN match with KDTree
            matches = self.matcher.knnMatch(des1,des2,k=2)
            # Ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            # Extract matched points
            pts1 = np.array([kp1[m.queryIdx].pt for m in good])
            pts2 = np.array([kp2[m.trainIdx].pt for m in good])
            des1 = np.array([des1[m.queryIdx] for m in good])
            des2 = np.array([des2[m.trainIdx] for m in good])

        else: # ORB feature matching
            matches = self.matcher.match(des1, des2)
            # Sort
            matches = sorted(matches, key = lambda x:x.distance)
            # Extract matched points
            pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
            des1 = np.array([des1[m.queryIdx] for m in matches])
            des2 = np.array([des2[m.trainIdx] for m in matches])
        
        if return_des:
            return pts1, pts2, des1, des2
        return pts1, pts2
    
    def feature_matching3D(self, pts3D, des1, kp2, des2):
        if self.type  == 'sift': #SIFT feature matching
            # KNN match with KDTree
            matches = self.matcher.knnMatch(des1,des2,k=2)
            # Ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            # Extract matched points
            pts1 = np.array([pts3D[m.queryIdx] for m in good])
            pts2 = np.array([kp2[m.trainIdx].pt for m in good])

        else: # ORB feature matching
            matches = self.matcher.match(des1, des2)
            # Sort
            matches = sorted(matches, key = lambda x:x.distance)
            # Extract matched points
            pts1 = np.array([pts3D[m.queryIdx] for m in matches])
            pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

        return pts1, pts2


    def get_keypoints(self, img):
        return self.detector.detectAndCompute(img,None)
    
    def feature_matching_image_pair(self, img1, img2, return_feature=False):
        kp1, des1 = self.get_keypoints(img1)
        kp2, des2 = self.get_keypoints(img2)

        if return_feature:
            pts = self.feature_matching(kp1, des1, kp2, des2)
            return pts[0], pts[1], (kp1, des1), (kp2, des2)

        return self.feature_matching(kp1, des1, kp2, des2)