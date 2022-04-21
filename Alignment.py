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

def BoundaryCompute(img1, img2):

    # 找出要變換之圖片轉換後的邊界
    boundary1_00 = [0, 0 ,1]
    boundary1_01 = [0, img1.shape[1], 1]
    boundary1_11 = [img1.shape[0], img1.shape[1], 1] 
    boundary1_10 = [img1.shape[0] , 0, 1]

    newboundary1_00 = np.dot(H, boundary1_00)
    newboundary1_01 = np.dot(H, boundary1_01)
    newboundary1_11 = np.dot(H, boundary1_11)
    newboundary1_10 = np.dot(H, boundary1_10)
    newboundary1_00[0:3] = newboundary1_00[0:3]/newboundary1_00[2]
    newboundary1_01[0:3] = newboundary1_01[0:3]/newboundary1_01[2]
    newboundary1_11[0:3] = newboundary1_11[0:3]/newboundary1_11[2]
    newboundary1_10[0:3] = newboundary1_10[0:3]/newboundary1_10[2]
    print('newboundary1_00 = ', newboundary1_00)
    print('newboundary1_01 = ', newboundary1_01)
    print('newboundary1_11 = ', newboundary1_11)
    print('newboundary1_10 = ', newboundary1_10)

    # max height & weight
    h1 = newboundary1_10 - newboundary1_00
    h2 = newboundary1_11 - newboundary1_01
    w1 = newboundary1_01 - newboundary1_00
    w2 = newboundary1_11 - newboundary1_10
    h = int(np.ceil(max(h1[0], h2[0])))
    w = int(np.ceil(max(w1[1], w2[1])))
    print('h = ', h, 'w = ', w)

    # 位移保留完整圖片(還未用到)
    offset = 0

    if newboundary1_00[0] < 0 :
        offset = -1*newboundary1_00[0]
        # newboundary1_00[0] = newboundary1_00[0] + offset
        # newboundary1_01[0] = newboundary1_01[0] + offset
        # newboundary1_11[0] = newboundary1_11[0] + offset
        # newboundary1_10[0] = newboundary1_10[0] + offset
        

    if newboundary1_01[0] < 0 :
        offset = -1*newboundary1_01[0]
        # newboundary1_00[0] = newboundary1_00[0] + offset
        # newboundary1_01[0] = newboundary1_01[0] + offset
        # newboundary1_11[0] = newboundary1_11[0] + offset
        # newboundary1_10[0] = newboundary1_10[0] + offset
    
    return h, w, offset

def MergeImg(H, h, w, img1, img2):
    inverseH = np.linalg.inv(H)
    print(inverseH)
    img = np.zeros((max(img1.shape[0], h), img1.shape[1]+w, 3), np.uint8)
    img.fill(200)

    for i in range(max(img1.shape[0], h)):
        for j in range(img1.shape[1]+w):
            invP = np.dot(inverseH, [i, j, 1])
            invP[0:3] = invP[0:3]/invP[2]
            if int(invP[0]) < 0 or int(invP[1]) < 0 or int(invP[0]) >= img1.shape[0] or int(invP[1]) >= img1.shape[1]:
                if i >= 0 and i < img1.shape[0]-1 and j >= 0 and j < img1.shape[1]-1:
                    img[i][j] = img2[i][j]
                else: 
                    img[i][j] = 0
            else:
                img[i][j] = img1[int(invP[0])][int(invP[1])]


    cv2.imshow('img', img)
    cv2.imwrite('./result_parrington/AlignmentTest.jpg', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    correspond = np.load('./result_parrington/correspond_0.npy')
    
    # Find the Best Homography Matrix
    H = FinalH(correspond)
    print(H)
    np.save(os.path.join(result_dir,'Best_Homography_Matrix'),H)

    # Stitching Image
    # 讀圖片
    img1 = cv2.imread('./result_parrington/corner0.jpg')  # 右圖, 要轉換的圖
    img2 = cv2.imread('./result_parrington/corner1.jpg')  # 左圖

    h, w, offset = BoundaryCompute(img1, img2)
    MergeImg(H, h, w, img1, img2)
    

