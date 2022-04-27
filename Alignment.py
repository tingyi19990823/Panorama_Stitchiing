from itertools import count
import random
from re import X
from cv2 import pyrUp
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
    try:
        RR = np.linalg.inv(RR)
    except:
        print('error: Inverse Matrix Does not exist!')
        return 0
    else:
        T = np.dot(RR, R.T)
        T = np.dot(T, B)
        H = np.append(T, 1).reshape(3, 3)
        return H

    # R = np.array(R)
    # H = np.linalg.lstsq(R, B, rcond=None)[0]
    # H = np.concatenate((H, [1]), axis=-1)
    # H = np.reshape(H, (3, 3))

    

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
    threshold = 2.0           # Threshold

    for i in range(n):
        tempPair = np.zeros((4))
        tempPair[0:4] = correspond[i][1:5]
        if(error(H, tempPair) < threshold):
            inlier = inlier + 1

    return inlier/n

# 取得inlier最多的變換式
def FinalH(correspond, times):
    TH = alignment(randompair(correspond))
    max = 0
    for i in range(times):
        H = alignment(randompair(correspond))
        if InlierPartial(H, correspond) > max:
            max = InlierPartial(H, correspond)
            TH = H
    
    print(max)
    return TH

def BoundaryCompute(img1, img2, H):

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
    
    w1 = img2.shape[1] - newboundary1_00[1]
    w2 = img2.shape[1] - newboundary1_10[1]
    overlap = int(max(w1, w2))
    print('h = ', h, 'w = ', w, 'overlap = ', overlap)


    
    return h, w, overlap

def MergeImg(H, h, w, img1, img2, overlapw):
    inverseH = np.linalg.inv(H)
    print(inverseH)
    # 型態問題
    newimg1 = np.ones((max(img1.shape[0], img2.shape[0], h), img1.shape[1]+w-overlapw, 3), float)
    newimg2 = np.ones((max(img1.shape[0], img2.shape[0], h), img2.shape[1], 3), float)
    newimg2.fill(0)

    for i in range(max(img1.shape[0], img2.shape[0], h)):
        for j in range(img1.shape[1]+w-overlapw):
            invP = np.dot(inverseH, [i, j, 1])
            invP[0:3] = invP[0:3]/invP[2]
            if int(invP[0]) < 0 or int(invP[1]) < 0 or int(invP[0]) >= img1.shape[0] or int(invP[1]) >= img1.shape[1]:
                newimg1[i][j] = 0
            else:
                newimg1[i][j] = img1[int(invP[0])][int(invP[1])]

    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            newimg2[i][j] = img2[i][j]


    return newimg1, newimg2


def GenerateMask(newimg1, newimg2, overlapw):
    if newimg1.shape[0] != newimg2.shape[0]:
        print("error: image dim error")
        return 
    
    w1 = newimg1.shape[1] # 右圖
    w2 = newimg2.shape[1] # 左圖

    shape = np.array(newimg1.shape)
    shape[1] = max(w1, w2)
    # 型態問題
    subimg1 = np.zeros(shape, float)
    start = w2 - overlapw
    subimg1[:, start:] = newimg1[:, start:]

    subimg2 = np.zeros(shape, float)
    subimg2[:, :w2] = newimg2[:, :]

    mask = np.zeros(shape)
    mask[:, w2 - int(overlapw/2):] = 1
    return subimg1, subimg2, mask


def GussianPyramid(img, levels):

    _GP = []
    _GP.append(img)

    currentImg = img


    for i in range(1, levels):
        downsampleImg = 0
        downsampleImg = cv2.pyrDown(currentImg)
        _GP.append(downsampleImg)
        currentImg = downsampleImg
        

    return _GP

def LaplacianPyramid(GP):

    levels = len(GP)
    _LP = []
    _LP.append(GP[levels-1])

    for i in range(levels - 2, -1, -1):
        upsampleImg = 0
        size = (GP[i].shape[1], GP[i].shape[0])
        upsampleImg = pyrUp(GP[i+1], dstsize=size)
        currentImg = cv2.subtract(GP[i], upsampleImg)
        _LP.append(currentImg)

    return _LP

def BlendPyramid(pyrA, pyrB, pyrMask):
    levels = len(pyrA)
    blendedP = []

    for i in range(0, levels):
        blendedImg = pyrA[i]*(1.0 - pyrMask[levels-1-i]) + pyrB[i]*(pyrMask[levels-1-i])
        print('blendedImg = ', blendedImg.shape)
        
        blendedP.append(blendedImg)
    
    return blendedP

# 重建
def CollapsePyramid(blendedP):
    levels = len(blendedP)
    currentImg = blendedP[0]
    for i in range(1, levels):
        size = (blendedP[i].shape[1], blendedP[i].shape[0])
        currentImg = pyrUp(currentImg, dstsize=size)
        
        currentImg = blendedP[i] + currentImg

    blendedImg = cv2.convertScaleAbs(currentImg)
    return blendedImg


def MultiBandBlending(img1, img2, correspond, timers):
    
    # Find the Best Homography Matrix
    H = FinalH(correspond, timers)
    np.save(os.path.join(result_dir,'Best_Homography_Matrix'),H) # save the H Matrix

    # conpute img1(which will be transformed) new height & weight
    h, w, overlap = BoundaryCompute(img1, img2, H)
     
    # Merge two image
    newimg1, newimg2 = MergeImg(H, h, w, img1, img2, overlap)
    subimg1, subimg2, mask = GenerateMask(newimg1, newimg2, overlap)

    levels = int(np.floor(np.log2(min(newimg1.shape[0], newimg1.shape[1], newimg2.shape[0], newimg2.shape[1]))))

    print('Levels = ', levels)

    MaskP = GussianPyramid(mask, levels)
    GP1 = GussianPyramid(subimg1, levels)
    GP2 = GussianPyramid(subimg2, levels)
    LP1 = LaplacianPyramid(GP1)
    LP2 = LaplacianPyramid(GP2)

    BlendedP = BlendPyramid(LP2, LP1, MaskP)

    Result = CollapsePyramid(BlendedP)
    cv2.imwrite('result.jpg',Result)
    cv2.imshow('result', Result)
    cv2.waitKey(0)
    return Result


if __name__ == '__main__':
    correspond = np.load('./result_parrington/correspond_0.npy')
    
    # Stitching Image
    # 讀圖片
    img1 = cv2.imread('./parrington/prtn00.jpg')  # 右圖, 要轉換的圖
    img2 = cv2.imread('./parrington/prtn01.jpg')  # 左圖


    MultiBandBlending(img1, img2, correspond)