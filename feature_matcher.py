from re import X
import numpy as np
import cv2
import math

def Matcher(img1,img2,feature1,feature2,index1,index2):
    feature_count = feature1.shape[2]

    concatenate_img = np.concatenate((img1,img2),axis = 1)

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
        if min_dist < 3:
            # concatenate_img = cv2.line(concatenate_img,(origin_y,origin_x),(match_y + img1.shape[1],match_x),(int(255*min_dist),0,0),2)
            concatenate_img = cv2.circle(concatenate_img,(origin_y,origin_x),5,(255,0,0),1)
            concatenate_img = cv2.circle(concatenate_img,(match_y + img1.shape[1],match_x),5,(255,0,0),1)
            print(min_dist)
    # cv2.imshow('test',concatenate_img)
    # cv2.waitKey(0)
    cv2.imwrite('match.jpg',concatenate_img)

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

