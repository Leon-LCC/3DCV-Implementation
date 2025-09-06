import numpy as np

def forward_warping(src, dst, H):
    '''
    Forward warping: Fill in the destination image using the source image and the homography matrix (Some pixels may be left empty)
    src: h x w x c source image
    dst: h x w x c destination image
    H: 3 x 3 homography matrix
    '''
    # Get the source and destination image sizes
    h_src, w_src, c = src.shape
    h_dst, w_dst, c = dst.shape

    # Meshgrid the (x,y) coordinate pairs
    x_src, y_src = np.mgrid[0:w_src,0:h_src]
    x_dst, y_dst = np.mgrid[0:w_dst,0:h_dst]

    # Reshape the destination pixels as N x 3
    x_src, y_src = x_src.reshape(-1,1), y_src.reshape(-1,1)
    x_dst, y_dst = x_dst.reshape(-1,1), y_dst.reshape(-1,1)
    coordinate_src = np.concatenate((x_src,y_src,np.ones((w_src*h_src,1))), axis=1)
    coordinate_dst = np.concatenate((x_dst,y_dst,np.ones((w_dst*h_dst,1))), axis=1)

     # Apply H to the source pixels
    transform_coordinate = np.matmul(H, coordinate_src.T)
    transform_coordinate[0,:] /= transform_coordinate[2,:]
    transform_coordinate[1,:] /= transform_coordinate[2,:]
    transform_coordinate = transform_coordinate.T

    # Filter the valid coordinates
    valid_idx = np.where((transform_coordinate[:,0]>=0) & (transform_coordinate[:,0] < w_dst) & (transform_coordinate[:,1] >= 0) & (transform_coordinate[:,1] < h_dst))
    valid_xy = transform_coordinate[valid_idx].astype(np.int64)

    # Assign to destination image
    dst[valid_xy[:,1], valid_xy[:,0]] = src[y_src[valid_idx,:].squeeze(), x_src[valid_idx,:].squeeze()]

    return dst



def backward_warping(src, dst, H):
    '''
    Backward warping: Fill in the destination image using the source image and the inverse of the homography matrix
    src: h x w x c source image
    dst: h x w x c destination image
    H: 3 x 3 homography matrix
    '''
    # Get the source and destination image sizes
    h_src, w_src, c = src.shape
    h_dst, w_dst, c = dst.shape

    # Compute the inverse homography matrix
    H_inv = np.linalg.inv(H)

    # Meshgrid the (x,y) coordinate pairs
    x_src, y_src = np.mgrid[0:w_src,0:h_src]
    x_dst, y_dst = np.mgrid[0:w_dst,0:h_dst]

    # Reshape the destination pixels as N x 3
    x_src, y_src = x_src.reshape(-1,1), y_src.reshape(-1,1)
    x_dst, y_dst = x_dst.reshape(-1,1), y_dst.reshape(-1,1)
    coordinate_src = np.concatenate((x_src,y_src,np.ones((w_src*h_src,1))), axis=1)
    coordinate_dst = np.concatenate((x_dst,y_dst,np.ones((w_dst*h_dst,1))), axis=1)

    # Apply H_inv to the destination pixels
    transform_coordinate = np.matmul(H_inv, coordinate_dst.T)
    transform_coordinate[0,:] /= transform_coordinate[2,:]
    transform_coordinate[1,:] /= transform_coordinate[2,:]
    transform_coordinate = transform_coordinate.T

    # Sample the source image with the reshaped transformed coordinates
    valid_idx = np.where((transform_coordinate[:,0]>=0) & (transform_coordinate[:,0] < w_src) & (transform_coordinate[:,1] >= 0) & (transform_coordinate[:,1] < h_src))
    valid_xy = transform_coordinate[valid_idx].astype(np.int64)

    # Assign to destination image with proper masking
    dst[y_dst[valid_idx,:].squeeze(), x_dst[valid_idx,:].squeeze()] = src[valid_xy[:,1], valid_xy[:,0]]

    return dst


def warping(src, dst, H, direction='b'):
    if direction == 'b':
        return backward_warping(src, dst, H)
    else:
        return forward_warping(src, dst, H)
