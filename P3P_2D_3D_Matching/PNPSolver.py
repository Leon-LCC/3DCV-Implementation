import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from math import sqrt

from trilateration3D import trilateration
from polynomialSolver import solve_deg4



# Solve Rotation and Translation
def solveRT(points3D, points2D, distances, method="Arun"):
    # Scale distances
    points2D_scaled = np.zeros(points2D.shape)
    points2D_scaled[0] = points2D[0] * distances[0]
    points2D_scaled[1] = points2D[1] * distances[1]
    points2D_scaled[2] = points2D[2] * distances[2]

    sol = []
    if method == "Arun":  # ICP Method
        # Scale distances
        points2D_scaled = np.zeros(points2D.shape)
        points2D_scaled[0] = points2D[0] * distances[0]
        points2D_scaled[1] = points2D[1] * distances[1]
        points2D_scaled[2] = points2D[2] * distances[2]

        # Centroid
        centroid2D = ((np.sum(points2D_scaled, axis=0))/3)
        centroid3D = ((np.sum(points3D, axis=0))/3)

        # Difference
        vec2D = points2D_scaled - centroid2D
        vec3D = points3D - centroid3D

        # Covariance
        cov = vec3D.T @ vec2D

        # SVD
        U, S, Vt = np.linalg.svd(cov)

        # Rotation
        rotM = Vt.T @ np.diag([1, 1, np.linalg.det(Vt) * np.linalg.det(U)]) @ U.T

        # translation estimation
        tVec = centroid2D.T - rotM @ centroid3D.T
        sol.append((rotM, tVec))


    elif method == "tril": # Trilateration
        # Translation (Trilateration)
        _tVec = trilateration(points3D, np.array([abs(distances[0]), abs(distances[1]), abs(distances[2])]))

        # Rotation
        rotM1 = points2D_scaled.T @ np.linalg.inv((points3D+_tVec[0]).T)
        rotM2 = points2D_scaled.T @ np.linalg.inv((points3D+_tVec[1]).T)

        # Correct Translation
        tVec1 = _tVec[0] @ rotM1.T
        tVec2 = _tVec[1] @ rotM2.T
        sol.append((rotM1, tVec1))
        sol.append((rotM2, tVec2))


    else:
        raise ValueError("Unknown method")

    return sol




# P3P
def solvep3p(points3D, points2D, cameraMatrix, distCoeffs, method='Fis', RTsolver='Arun'): #(4,3),(4,2)   
    # Transform 2D points to Camera Coordinate
    p2DCamCoord = cv2.undistortPoints(points2D, cameraMatrix, distCoeffs)
    p2DCamCoord = np.concatenate((p2DCamCoord.reshape(-1,2), np.ones((p2DCamCoord.shape[0],1))), axis=1)
    p2DCamCoord = p2DCamCoord / np.linalg.norm(p2DCamCoord, axis=1).reshape(-1,1)

    # Distances
    R1 = np.linalg.norm(points3D[0] - points3D[1])
    R2 = np.linalg.norm(points3D[1] - points3D[2])
    R3 = np.linalg.norm(points3D[2] - points3D[0])

    # Angles
    C1 = np.dot(p2DCamCoord[0], p2DCamCoord[1])
    C2 = np.dot(p2DCamCoord[1], p2DCamCoord[2])
    C3 = np.dot(p2DCamCoord[2], p2DCamCoord[0])

    # Distances too small
    if R1 < 1e-15 or R2 < 1e-15 or R3 < 1e-15:
        return None, None
    
    # Calculate Scales
    if method == 'Fis': # Fischerler & Bolles
        # Coefficients
        K1, K2 = (R1/R3)**2, (R1/R2)**2
        K1K2, K1_K2, K1pK2 = K1*K2, K1-K2, K1+K2
        G4 = (K1K2-K1pK2)**2-4*K1K2*C1*C1
        G3 = 4*(K1K2-K1pK2)*K2*(1-K1)*C2 + 4*K1*C1*((K1K2-K1_K2)*C3+2*K2*C2*C1)
        G2 = (2*K2*(1-K1)*C2)**2 + 2*(K1K2-K1pK2)*(K1K2+K1_K2) + 4*K1*((K1_K2)*C1*C1+K1*(1-K2)*C3*C3-2*(1+K1)*K2*C1*C2*C3)
        G1 = 4*(K1K2+K1pK2)*K2*(1-K1)*C2 + 4*K1*((K1K2-K1_K2)*C3*C1+2*K1K2*C2*C3*C3)
        G0 = (K1K2+K1_K2)**2-4*K1*K1K2*C3*C3
        
        # Solve polynomial
        roots, n = solve_deg4(G4, G3, G2, G1, G0)

        # All possible Scales
        scales = []
        for i in range(n):
            # Skip invalid solutions
            _cu = (1-K1)*(roots[i]*roots[i]*(1-K2)+2*roots[i]*K2*C2-K2)-(roots[i]*roots[i]-K1)
            _cd = 2*K1*(C3-roots[i]*C1)
            if _cd == 0:
                continue

            # Distances to Camera
            a = sqrt(R2*R2 / abs(1+roots[i]*roots[i]-2*roots[i]*C2))
            b = roots[i] * a
            c = _cu/_cd * a
            scales.append([a, b, c])


    elif method == 'Grun': # Grunert
        # Coefficients
        R1R1, R2R2, R3R3 = R1*R1, R2*R2, R3*R3
        C1C1, C2C2, C3C3 = C1*C1, C2*C2, C3*C3
        t1 = (R1R1-R3R3) / R2R2
        t2 = (R1R1+R3R3) / R2R2
        G4 = (t1-1)*(t1-1) - 4*R3R3/R2R2*C1C1
        G3 = 4*(t1*(1-t1)*C2 - (1-t2)*C1*C3 + 2*R3R3/R2R2*C1C1*C2)
        G2 = 2*(t1*t1 - 1 + 2*t1*t1*C2C2 + 2*(R2R2-R3R3)/R2R2*C1C1 - 4*t2*C1*C2*C3 + 2*(R2R2-R1R1)/R2R2*C3C3)
        G1 = 4*(-t1*(1+t1)*C2 + 2*R1R1/R3R3*C3C3*C2 - (1-t2)*C1*C3)
        G0 = (1+t1)*(1+t1) - 4*R1R1/R2R2*C3C3

        # Solve polynomial
        roots, n = solve_deg4(G4, G3, G2, G1, G0)

        # All possible Scales
        scales = []
        for i in range(n):
            _u = ((-1+t1)*roots[i]*roots[i]-2*t1*C2*roots[i] + 1 + t1) / (2*(C3-roots[i]*C1))
            # Distances to Camera
            a = sqrt(R1R1 / abs(1 + roots[i]*roots[i] - 2*roots[i]*C2))
            b = roots[i] * a
            c = _u * a
            scales.append([a, b, c])

    else:
        raise ValueError("Unknown method")
    
    
    # Compute Possible Rotation and Translation
    solutions = []
    for a, b, c in scales:
        solutions += solveRT(points3D[:3], p2DCamCoord[:3], [c, b, a], RTsolver)
        solutions += solveRT(points3D[:3], p2DCamCoord[:3], [-c, -b, -a], RTsolver)

    # Select Rotation and Translation with smallest reprojection error
    if len(solutions) == 0:
        return None, None
    else:
        min_error = np.inf
        for i in range(len(solutions)):
            # Reprojection error
            points2D_reproj = (cameraMatrix @ (solutions[i][0] @ points3D.T + solutions[i][1].reshape(-1,1))).T
            points2D_reproj = points2D_reproj / points2D_reproj[:,2].reshape(-1,1)
            undispoints2D = cv2.undistortPoints(points2D, cameraMatrix, distCoeffs, None, cameraMatrix)
            error = np.linalg.norm(points2D_reproj[:,:2] - undispoints2D)
            if error < min_error:
                min_error = error
                best_solution = solutions[i]

    return best_solution[0], best_solution[1]




#P3P + RANSAC
def P3PRansac(points3D, points2D, cameraMatrix, distCoeffs, method='Grun', RTsolver='Arun'):
    # Number of points
    N = points3D.shape[0]

    # Undistort points
    undispoints2D = cv2.undistortPoints(points2D, cameraMatrix, distCoeffs, None, cameraMatrix).reshape(N, 2)

    max_num = 0
    best_rot = None
    best_tvec = None
    for i in range(10000):
        # Random Sample
        sample_idx = np.random.choice(N, 4, replace=False)

        # Solve P3P
        rotM, tVec = solvep3p(points3D[sample_idx], points2D[sample_idx], cameraMatrix, distCoeffs, method=method, RTsolver=RTsolver)
        if rotM is None:
            continue

        # Reprojection error
        points2D_reproj = (cameraMatrix @ (rotM @ points3D.T + tVec.reshape(-1,1))).T
        points2D_reproj = points2D_reproj / points2D_reproj[:,2].reshape(-1,1)
        error = np.linalg.norm(points2D_reproj[:,:2] - undispoints2D, axis=1)
        inliner_num = np.sum(np.where(error < 5, 1, 0))

        # Update
        if inliner_num > max_num:
            max_num = inliner_num
            best_rot, best_tvec = rotM, tVec

    return R.from_matrix(best_rot).as_rotvec(), best_tvec





if __name__ == '__main__':
    # Camera Matrix
    cameraMatrix = np.array([[2, 0, 100],
                             [0, 3, 200],
                             [0, 0, 1]]).astype(np.float64)
    # Camera Distortion
    distCoeffs = np.array([0, 0, 0, 0]).astype(np.float64)
        
    # 3D Points
    points3D = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5], [2, 2, 2]]).astype(np.float64)
    
    # 2D Points
    points2D = np.array([[ 95.5763854 , 200.98976395],
                         [ 89.9856876 , 216.81463785],
                         [ 99.39037456, 198.57991566],
                         [ 98.0000008 , 200.00000060]]).astype(np.float64)
    
    # Solve P
    R, T = solvep3p(points3D, points2D, cameraMatrix, distCoeffs, method="Grun", RTsolver="Arun")
    print(R, T)
    print(cv2.solvePnP(points3D, points2D, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_EPNP))

    ''' Answer
    R =  array([[-0.3016087,  0.8267868,  0.4748218],
                [ 0.4748218, -0.3016087,  0.8267868],
                [ 0.8267868,  0.4748218, -0.3016087]])
    T = array([[-1],
               [-2],
               [-3]])
    '''