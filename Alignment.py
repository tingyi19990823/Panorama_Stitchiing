from itertools import count
import random
from re import X
import numpy as np
import cv2
import math
import os

result_dir = "./result_parrington"

# random選四個matching points
def randompair(correspond):
    pairs = np.zeros((4, 4))
    idx = random.sample(range(correspond.shape[0]), 4)
    for i in range(4):
        pairs[i][0] = correspond[idx[i]][1]
        pairs[i][1] = correspond[idx[i]][2]
        pairs[i][2] = correspond[idx[i]][3]
        pairs[i][3] = correspond[idx[i]][4]
    
    return pairs

# 以隨機選出的四個點建出的變換式
def alignment(pairs):
    R = np.zeros((8, 8))
    B = np.zeros((8, 1))
    # (x, y) = (0, 1), (u, v) = (2, 3)
    for i in range(pairs.shape[0]):
        R[2*i][0:2] = pairs[i][0:2]  # x, y
        R[2*i][2] = 1              
        R[2*i][6] = -1 * pairs[i][2] * pairs[i][0] # -u * x
        R[2*i][7] = -1 * pairs[i][2] * pairs[i][1] # -u * y

        R[2*i+1][3:5] = pairs[i][0:2]
        R[2*i+1][5] = 1
        R[2*i+1][6] = -1 * pairs[i][3] * pairs[i][0] # -v * x
        R[2*i+1][7] = -1 * pairs[i][3] * pairs[i][1] # -v * y

        B[2*i][0] = pairs[i][2]
        B[2*i+1][0] = pairs[i][3]

    RR = np.dot(R.T, R)
    RR = np.linalg.inv(RR)
    T = np.dot(RR, R.T)
    T = np.dot(T, B)
    H = np.append(T, 1).reshape(3, 3)
   
    return H

# 計算其他點透過變換式得到的變換點與原始點的差異
def error(H, pair):
    # (x, y)：需轉換的點 (u, v)：轉換後比較的點
    P1 = np.ones((3, 1))
    P2 = np.ones((3, 1))

    P1[0][0] = pair[0]
    P1[1][0] = pair[1]
    P2[0][0] = pair[2]
    P2[1][0] = pair[3]

    # print('P1 = ', P1)
    # print('P2 = ', P2)

    # 預測出的position
    estimate_P = np.dot(H, P1)
    estimate_P[0:3] = estimate_P[0:3]/estimate_P[2]

    # print('estimate_P = ', estimate_P)

    # 計算出預測與實際第二張圖的位置之差異
    get_error = np.linalg.norm(P2 - estimate_P)

    # print('error = ', get_error)
    
    return get_error

# 計算變換式的inliers
def InlierPartial(H, correspond):
    n = correspond.shape[0] # 有多少個matching point
    inlier = 0              # inlier數量
    threshold = 3           # Threshold

    for i in range(n):
        tempPair = np.zeros((4))
        tempPair[0:4] = correspond[i][1:5]
        if(error(H, tempPair) < threshold):
            inlier = inlier + 1

    return inlier/n

# 取得inlier最多的變換式
def FinalH(correspond):
    TH = alignment(randompair(correspond))
    max = 0
    for i in range(100):
        H = alignment(randompair(correspond))
        if InlierPartial(H, correspond) > max:
            max = InlierPartial(H, correspond)
            TH = H
    
    print(max)
    return TH

if __name__ == '__main__':
    correspond = np.load('./result_parrington/correspond_0.npy')
    
    print(FinalH(correspond))

    np.save(os.path.join(result_dir,'Best_Homography_Matrix'),FinalH)