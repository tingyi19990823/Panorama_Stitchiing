from pickletools import uint8
from cv2 import blur
import numpy as np
import cv2

corner_threshold = 10 # 1000比較正常

# input: 要pooling的圖片
# output: pooling完的圖片(當作遮罩使用)、特徵點數量
def Max_Pooling(input):
    output = np.zeros((input.shape[0],input.shape[1]))
    max_row = 0
    max_col = 0
    corner_count = 0
    for row in range(0,input.shape[0] - 2, 3):
        for col in range(0,input.shape[1] - 2, 3):
            max = 0
            for i in range(3):
                for j in range(3):
                    if input[row + i,col + j] > max:
                        max = input[row + i,col + j]
                        max_row = row + i
                        max_col = col + j
            if max > corner_threshold:
                output[max_row,max_col] = 255
                corner_count = corner_count + 1
            
    return output,corner_count

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


# input: 要detect的圖片
# output: 灰階、遮罩、角落偵測圖片
def Corner_detection(img):
    
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    Ix = cv2.Sobel(grayImg,cv2.CV_64F,1,0)
    Iy = cv2.Sobel(grayImg,cv2.CV_64F,0,1)
    
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
    corner_response = (HL_mat_dict[:,:,0] * HL_mat_dict[:,:,1] - np.power(Sxy,2)) / (HL_mat_dict[:,:,0] + HL_mat_dict[:,:,1])

    corner_response = np.nan_to_num(corner_response)

    maskImg, corner_count = Max_Pooling(corner_response)
    print('corner count: ',corner_count)
    cornerImg = Create_CornerImg(img,maskImg)


    return grayImg, maskImg, cornerImg