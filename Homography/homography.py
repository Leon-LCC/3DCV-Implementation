import numpy as np
import random

def estimate_homography(source, destination):
    '''
    Estimate the homography matrix H such that destination = H * source
    source: N x 2 source points
    destination: N x 2 destination points
    '''
    # Number of points
    N = source.shape[0]
    assert destination.shape[0] == N, 'Number of points do not match'
    assert N >= 4, 'At least 4 points are required to estimate homography'

    # Direct Linear Transform (DLT) algorithm
    M = np.zeros((2*N, 9))
    # Fill in the M matrix
    for i in range(N):
        M[i*2, 0] = M[i*2+1, 3] = source[i,0]
        M[i*2, 1] = M[i*2+1, 4] = source[i,1]
        M[i*2, 2] = M[i*2+1, 5] = 1
        M[i*2, 6:8] = -(destination[i,0] * source[i,:])
        M[i*2+1, 6:8] = -(destination[i,1] * source[i,:])
        M[i*2, -1] = -destination[i, 0]
        M[i*2+1, -1] = -destination[i, 1]

    # Solve the linear system
    u, s, vh = np.linalg.svd(M, full_matrices=True)
    H = vh.T[:,-1].reshape(3,-1)

    return H



def RANSAC(match_point, iter_num=2000):
    '''
    Random Sample Consensus (RANSAC) for robust homography estimation
    match_point: Nx4 array of matched points [x1, y1, x2, y2]
    iter_num: Number of RANSAC iterations
    '''
    best_H = None
    max_num = 0
    # RANSAC loop
    for _ in range(iter_num):
        # Sample random points
        sample_idx = random.sample([i for i in range(match_point.shape[0])], k=4)
        src_corns = np.array([[match_point[sample_idx[0],0], match_point[sample_idx[0],1]],
                              [match_point[sample_idx[1],0], match_point[sample_idx[1],1]],
                              [match_point[sample_idx[2],0], match_point[sample_idx[2],1]],
                              [match_point[sample_idx[3],0], match_point[sample_idx[3],1]]])

        dst_corns = np.array([[match_point[sample_idx[0],2], match_point[sample_idx[0],3]],
                              [match_point[sample_idx[1],2], match_point[sample_idx[1],3]],
                              [match_point[sample_idx[2],2], match_point[sample_idx[2],3]],
                              [match_point[sample_idx[3],2], match_point[sample_idx[3],3]]])

        # Estimate the homography matrix
        H = estimate_homography(src_corns, dst_corns)
        H_inv = np.linalg.inv(H)

        original = np.concatenate((match_point[:,0:2],np.ones((match_point.shape[0],1))), axis=1)
        projected = np.concatenate((match_point[:,2:4],np.ones((match_point.shape[0],1))), axis=1)

        projected_hat = np.matmul(H, original.T)
        projected_hat[0,:] /= projected_hat[2,:]
        projected_hat[1,:] /= projected_hat[2,:]
        projected_hat = projected_hat[0:2,:].T

        original_hat = np.matmul(H_inv, projected.T)
        original_hat[0,:] /= original_hat[2,:]
        original_hat[1,:] /= original_hat[2,:]
        original_hat = original_hat[0:2,:].T

        # Compute the reprojection error when H (H_inv) is applied
        error = np.linalg.norm(original[:,0:2]-original_hat, axis=1)**2 + np.linalg.norm(projected[:,0:2]-projected_hat, axis=1)**2
        inliner_num = np.sum(np.where(error < 10, 1, 0))

        if inliner_num > max_num:
            max_num = inliner_num
            best_H = H

    return best_H
