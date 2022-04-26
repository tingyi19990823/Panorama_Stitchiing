
from operator import index
from itertools import count
from re import X
import numpy as np
import cv2
import math
import os
import sys


result_dir = "./result_parrington"

def Matcher(img1,img2,feature1,feature2,index1,index2):
    feature_count = feature1.shape[2]

    # concatenate_img = np.concatenate((img2,img1),axis = 1)

    # Threshold
    threshold = 2.5

    # 儲存對應點
    # correspond = np.zeros((feature_count,4))
    correspondA = np.zeros((1, 5))
    correspondB = np.zeros((1, 5))
    counter = 0

    counter = 0
    for i in range(feature_count):
        min_dist = math.inf
        min_index = -1 
        for j in range(feature_count):
            dist = math.sqrt(np.sum(np.square(feature1[:,:,i] - feature2[:,:,j])))
            if min_dist > dist:
                min_dist = dist
                min_index = j
        origin_x = index1[i][0]
        origin_y = index1[i][1]
        match_x = index2[min_index][0]
        match_y = index2[min_index][1]


        if min_dist < threshold:
            # concatenate_img = cv2.line(concatenate_img,(origin_y,origin_x),(match_y + img1.shape[1],match_x),(int(255*min_dist),0,0),2)
            # concatenate_img = cv2.circle(concatenate_img,(origin_y,origin_x),5,(255,0,0),1)                  # 左圖
            # concatenate_img = cv2.circle(concatenate_img,(match_y + img1.shape[1],match_x),5,(255,0,0),1)    # 右圖
            # concatenate_img = cv2.circle(concatenate_img,(origin_y + img2.shape[1],origin_x),5,(255,0,0),1)    # 右圖
            # concatenate_img = cv2.circle(concatenate_img,(match_y,match_x),5,(255,0,0),1)                      # 左圖
            if counter == 0:
                correspondA = [i, origin_x, origin_y, match_x, match_y]
            else:
                correspondB = [i, origin_x, origin_y, match_x, match_y]
                correspondA = np.append(correspondA, correspondB, axis=0)
                
            counter = counter + 1
    if counter == 0:
        sys.exit('沒有 Match 點 !!!')
    correspondA = np.reshape(correspondA, (counter, 5))    
    print('# of matching point = ', counter)
    # print(correspondA)
    # cv2.imshow('test',concatenate_img)
    # cv2.waitKey(0)
    # cv2.imwrite('match.jpg',concatenate_img)
    
    np.save(os.path.join(result_dir,'correspond_0'),correspondA)
    return correspondA

if __name__ == '__main__':
    # 讀圖片
    img1 = cv2.imread('./result_parrington/corner0.jpg')
    img2 = cv2.imread('./result_parrington/corner1.jpg')
    # 讀 feature跟index進來
    feature1 = np.load('./result_parrington/feature_0.npy')
    index1 = np.load('./result_parrington/feature_index0.npy')
    feature2 = np.load('./result_parrington/feature_1.npy')
    index2 = np.load('./result_parrington/feature_index1.npy')

    Matcher(img1,img2,feature1,feature2,index1,index2)

