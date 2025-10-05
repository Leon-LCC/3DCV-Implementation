# Reference: https://github.com/akshayb6/trilateration-in-3d

import numpy as np

from math import sqrt

# Let 3 points form the xy-plane, and one point is at the origin
def trilateration(points, distances): #(N,3), (N,)
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]

    # v2 projection on v1
    v2Pv1 = np.dot(v1, v2) / np.dot(v1, v1) * v1
    # v2 orthogonal to v1
    v2Ov1 = v2 - v2Pv1
    
    #New axis
    nx = v1 / np.linalg.norm(v1)
    ny = v2Ov1 / np.linalg.norm(v2Ov1)
    nz = np.cross(v1, v2)
    nz = nz / np.linalg.norm(nz)
    
    # 3D tilateration
    # https://en.wikipedia.org/wiki/True-range_multilateration
    U = np.linalg.norm(v1)
    Vx = np.linalg.norm(v2Pv1) * np.sign(np.dot(v1, v2))
    Vy = np.linalg.norm(v2Ov1)
    r1_square = distances[0]*distances[0]
    r2_square = distances[1]*distances[1]
    r3_square = distances[2]*distances[2]
    x = (r1_square-r2_square+U*U) / (2*U)
    y = (r1_square-r3_square+Vx*Vx+Vy*Vy-2*Vx*x) / (2*Vy)
    z1 = sqrt(abs(r1_square-x*x-y*y))
    z2 = -z1
    
    # Transform (x,y,z) back to original coordinate system
    vec1 = x * nx + y * ny + z1 * nz
    vec2 = x * nx + y * ny + z2 * nz

    return -(points[0] + vec1), -(points[0] + vec2)



if __name__ == '__main__':
    points = np.array([[0, 0, 1], [0, 1, 2], [-1, 0, 1]])
    distance = np.array([1, 1, 1])
    print(trilateration(points, distance))