from cgi import test
from pickletools import uint8
from cv2 import blur
import numpy as np
import cv2
from numpy import unravel_index
import time

corner_threshold = 10 # 1000比較正常
kernel_size = 10
keypoint_count = 250

# input: 要pooling的圖片(灰階)、Kernel_Size
# output: pooling完的圖片
def Max_Pooling(input,kernel_size):
    pooledImg = np.zeros((input.shape[0],input.shape[1]))
    max_row = 0
    max_col = 0
    corner_count = 0
    for row in range(0,input.shape[0] - (kernel_size + 1), kernel_size):
        for col in range(0,input.shape[1] - (kernel_size + 1), kernel_size):
            max = 0
            for i in range(kernel_size):
                for j in range(kernel_size):
                    if input[row + i,col + j] > max:
                        max = input[row + i,col + j]
                        max_row = row + i
                        max_col = col + j
            if max > corner_threshold:
                pooledImg[max_row,max_col] = max
                corner_count = corner_count + 1
    print('corner count after pooling: ',corner_count)
    return pooledImg

# input: 原圖、遮罩
# output: 原圖上加紅點點的圖
def Create_CornerImg(img,maskImg):
    maskImg = maskImg.astype(np.int)
    not_mask = np.bitwise_not(maskImg)
    not_mask[not_mask == -1] = 255
    cornerImg = img

    cornerImg[:,:,2] = np.bitwise_or(cornerImg[:,:,2],maskImg)
    cornerImg[:,:,1] = np.bitwise_and(cornerImg[:,:,1],not_mask)
    cornerImg[:,:,0] = np.bitwise_and(cornerImg[:,:,0],not_mask)
    return cornerImg

# input: 特徵點圖片、要取的特徵點數量
# output: 根據要的特徵點的數量輸出mask
# 未完成
def Non_Maximal_Suppression(input, keypoint_count):
    start = time.clock()
    mask = np.zeros((input.shape[0],input.shape[1]),dtype = float)
    keypoints = 0   # 找到的特徵點數量
    radius = 0
    if input.shape[0] > input.shape[1]:
        radius = input.shape[0]
    else:
        radius = input.shape[1]
    while keypoints < keypoint_count:
        radius = radius - 10
        keypoints = 0
        mask = np.copy(input)
        while np.any(mask > corner_threshold) :
            current_keypoint_index = unravel_index(mask.argmax(),input.shape)
            x_center = current_keypoint_index[0]
            y_center = current_keypoint_index[1]

            x_offset_right = x_center + radius
            x_offset_left = x_center - radius
            y_offset_up = y_center - radius
            y_offset_down = y_center + radius

            if x_offset_right > input.shape[0]:
                x_offset_right = input.shape[0]
            if x_offset_left < 0:
                x_offset_left = 0
            if y_offset_up < 0:
                y_offset_up = 0
            if y_offset_down > input.shape[1]:
                y_offset_down = input.shape[1]

            mask[x_offset_left:(x_offset_right+1), y_offset_up:(y_offset_down+1)] = 0   # 在範圍內的都變0
            mask[x_center,y_center] = -1                                                # 變-1，下次就不會找到
            keypoints = keypoints + 1
            if keypoints == keypoint_count:
                break
        if radius < 0:
            break
    mask[mask == -1] = 255
    end = time.clock()
    if keypoints < 500:
        print('\n not enough keypoints , count: {} \n'.format(np.sum(mask == 255)))
    else:
        print('\n Non_Maximal_Suppression done,keypoint count: {}, time cost: {} \n'.format(np.sum(mask == 255),end-start))
    return mask

# input: 要detect的圖片
# output: 灰階、遮罩、角落偵測圖片
def Corner_detection(img):
    
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg,(3,3),1.0)

    Ix = cv2.Sobel(blurImg,cv2.CV_64F,1,0)
    Iy = cv2.Sobel(blurImg,cv2.CV_64F,0,1)
    
    Ix2 = np.power(Ix,2)
    Iy2 = np.power(Iy,2)
    Ixy = Ix * Iy

    Sx2 = cv2.GaussianBlur(Ix2,(3,3),1.5)
    Sy2 = cv2.GaussianBlur(Iy2,(3,3),1.5)
    Sxy = cv2.GaussianBlur(Ixy,(3,3),1.5)

    corner_response = np.zeros((img.shape[0],img.shape[1]),dtype = float)

    HL_mat_dict = np.zeros((img.shape[0],img.shape[1],3),dtype = float)
    HL_mat_dict[:,:,0] = Sx2
    HL_mat_dict[:,:,1] = Sy2
    HL_mat_dict[:,:,2] = Sxy
    corner_response = (HL_mat_dict[:,:,0] * HL_mat_dict[:,:,1] - np.power(Sxy,2)) / (HL_mat_dict[:,:,0] + HL_mat_dict[:,:,1])       # det/trace
    corner_response = np.nan_to_num(corner_response)

    pooledImg = Max_Pooling(corner_response,kernel_size)

    maskImg = Non_Maximal_Suppression(pooledImg,keypoint_count)

    cornerImg = Create_CornerImg(img,maskImg)

    return grayImg, maskImg, cornerImg