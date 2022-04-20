from cgi import test
from operator import truediv
from pickletools import uint8
from tkinter import CENTER
from cv2 import blur
import numpy as np
import cv2
from numpy import unravel_index
import time

corner_threshold = 10 # 1000比較正常
kernel_size = 10
# keypoint_count = 250

# input: 要pooling的圖片(灰階)、Kernel_Size
# output: pooling完的圖片,回傳所有可能的特徵點
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

    cornerImg[:,:,0] = np.bitwise_or(cornerImg[:,:,0],maskImg)
    cornerImg[:,:,1] = np.bitwise_and(cornerImg[:,:,1],not_mask)
    cornerImg[:,:,2] = np.bitwise_and(cornerImg[:,:,2],not_mask)
    return cornerImg

# input: 特徵點圖片、要取的特徵點數量
# output: 根據要的特徵點的數量輸出mask
# 未完成
def Non_Maximal_Suppression(input, keypoint_count):
    height,width = input.shape[:2]
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

            if (x_center + 20) > height or (x_center - 20) < 0 or (y_center + 20) > width or (y_center - 20) < 0:
                mask[x_center,y_center] = 0
                continue

            height_offset_down = x_center + radius
            height_offset_up = x_center - radius
            width_offset_right = y_center + radius
            width_offset_left = y_center - radius
            
            if height_offset_down > height:
                height_offset_down = height
            if height_offset_up < 0:
                height_offset_up = 0
            if width_offset_right > width:
                width_offset_right = width
            if width_offset_left < 0:
                width_offset_left = 0

            mask[height_offset_up:(height_offset_down+1), width_offset_left:(width_offset_right+1)] = 0   # 在範圍內的都變0
            mask[x_center,y_center] = -1                                                                  # 變-1，下次就不會找到
            keypoints = keypoints + 1
            if keypoints == keypoint_count:
                break
        if radius < 0:
            break
    mask[mask != -1] = 0
    mask[mask == -1] = 255
    end = time.clock()
    if keypoints < keypoint_count:
        print('\n not enough keypoints , count: {} \n'.format(np.sum(mask == 255)))
    else:
        print('\n Non_Maximal_Suppression done,keypoint count: {}, radius: {}\n'.format(np.sum(mask == 255),radius))
    return mask

# input: 要detect的圖片
# output: 灰階、遮罩、角落偵測圖片
def Corner_detection(img,keypoint_count):
    
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
    det = HL_mat_dict[:,:,0] * HL_mat_dict[:,:,1] - np.power(Sxy,2)
    trace = HL_mat_dict[:,:,0] + HL_mat_dict[:,:,1]
    corner_response = np.divide(det,trace,out = np.zeros_like(det), where=trace != 0)

    pooledImg = Max_Pooling(corner_response,kernel_size)

    maskImg = Non_Maximal_Suppression(pooledImg,keypoint_count)

    cornerImg = Create_CornerImg(img,maskImg)

    return grayImg, maskImg, cornerImg