import numpy as np
import cv2.ximgproc as xip
import cv2


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    grayIr = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
    grayIl = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)

    Ir = xip.jointBilateralFilter(joint=grayIr, src=Ir, d=-1, sigmaColor=2, sigmaSpace=5)
    Il = xip.jointBilateralFilter(joint=grayIl, src=Il, d=-1, sigmaColor=2, sigmaSpace=5)

    # Cost Computation: Compute matching cost
    census_map_r = np.zeros((h-2, w-2, ch), dtype=np.uint8)
    census_map_l = np.zeros((h-2, w-2, ch), dtype=np.uint8)
    coords = [(u, v) for v in range(3) for u in range(3) if not (u == 1 and v == 1)]

    for u,v in coords:
        census_map_r = (census_map_r << 1) | (Ir[v:v+h-2, u:u+w-2] >= Ir[1:h-1, 1:w-1])
        census_map_l = (census_map_l << 1) | (Il[v:v+h-2, u:u+w-2] >= Il[1:h-1, 1:w-1])

    cost_r = np.zeros((max_disp, h-2, w-2), dtype=np.uint8)
    cost_l = np.zeros((max_disp, h-2, w-2), dtype=np.uint8)

    binary = [2**i for i in range(8)]
    for i in range(max_disp):
        hamming_dist = np.zeros((h-2, w-3-i))
        for bit in binary:
            for channel in range(3):
                hamming_dist += (np.bitwise_xor(census_map_l[:,i+1:,channel], census_map_r[:,:w-i-3,channel]) & bit != 0).astype(np.float)
        
        cost_r[i,:,:] = np.pad(hamming_dist, ((0,0),(0,i+1)), 'edge')
        cost_l[i,:,:] = np.pad(hamming_dist, ((0,0),(i+1,0)), 'edge')

    cost_r = np.pad(cost_r, ((0,0),(1,1),(1,1)), 'edge').astype(np.float32)
    cost_l = np.pad(cost_l, ((0,0),(1,1),(1,1)), 'edge').astype(np.float32)


    # Cost Aggregation: Refine the cost according to nearby costs
    for i in range(max_disp):
        cost_r[i,:,:] = xip.guidedFilter(guide=Ir, src=cost_r[i,:,:], radius=12, eps=9, dDepth=-1)
        cost_l[i,:,:] = xip.guidedFilter(guide=Il, src=cost_l[i,:,:], radius=12, eps=9, dDepth=-1)


    # Disparity Optimization: Determine disparity based on estimated cost.
    disparity_map_r = np.argmin(cost_r, axis=0)+1
    disparity_map_l = np.argmin(cost_l, axis=0)+1


    # Disparity Refinement: Left-right consistency check -> Hole filling -> Weighted median filtering
    y = np.repeat(np.arange(h).reshape(-1,1), w, axis=1)
    x = np.repeat(np.arange(w).reshape(1,-1), h, axis=0) + disparity_map_r
    x = np.clip(x, None, w-1)
    valid = np.where(disparity_map_r==disparity_map_l[y,x], True, False)
    Fr = np.where(valid, disparity_map_r, 0)
    for y in range(h):
        for x in range(w):
            if Fr[y,x] == 0:
                idx = x+1
                while Fr[y,x] == 0:
                    if idx == w:
                        Fr[y,x] = max_disp
                    elif Fr[y,idx] != 0:
                        Fr[y,x] = Fr[y,idx]
                    else:
                        idx += 1

    y = np.repeat(np.arange(h).reshape(-1,1), w, axis=1)
    x = np.repeat(np.arange(w).reshape(1,-1), h, axis=0) - disparity_map_l
    x = np.clip(x, 0, None)
    valid = np.where(disparity_map_l==disparity_map_r[y,x], True, False)
    Fl = np.where(valid, disparity_map_l, 0)
    for y in range(h):
        for x in range(w-1,-1,-1):
            if Fl[y,x] == 0:
                idx = x-1
                while Fl[y,x] == 0:
                    if idx == -1:
                        Fl[y,x] = max_disp
                    elif Fl[y,idx] != 0:
                        Fl[y,x] = Fl[y,idx]
                    else:
                        idx -= 1

    checkandfill_dismap = np.where(valid, disparity_map_l, np.minimum(Fl,Fr)).astype(np.uint8)
    labels = xip.weightedMedianFilter(grayIl.astype(np.uint8), checkandfill_dismap, r=2)

    return labels.astype(np.uint8)
    