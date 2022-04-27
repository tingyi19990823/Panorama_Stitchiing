from unittest import result
import feature_detector
import feature_descriptor
import Cylindrical_Projector
import feature_matcher
import Alignment
import cv2
import os
import time
import numpy as np

# img_dir = "./grail"
# img_dir = "./pic"
img_dir = "./data"
img_list = []

# result_dir = "./result_grail"
# result_dir = "./result_pic"
result_dir = "./result_parrington"

keypoint_count = 1000

focal_length_all = [666.95,667.752,668.862,669.435,669.253,668.272,668.224,667.011]

# input : img
# output : 沒有黑邊的圖片
def ClearBorder(test, cut):
    left_up = 0
    left_down = test.shape[0]
    right_up = 0
    right_down = test.shape[0]
    test = test[:,15:test.shape[1]-cut]
    
    for i in range(test.shape[0]):
        if test[i,0,0] != 0:
            break
        else:
            left_up = left_up + 1

    for i in range(int(test.shape[0]/2) , test.shape[0]):
        if test[i,0,0] == 0:
            left_down = i
            break
    for j in range(0,test.shape[0]):
        if test[j,test.shape[1]-1,0] != 0:
            break
        else:
            right_up = right_up + 1
    for j in range(int(test.shape[0]/2),test.shape[0]):
        if test[j,test.shape[1]-1,0] == 0:
            right_down = j
            break
    up = max(left_up,right_up)
    down = min(left_down,right_down)
    
    test = test[up:down,:]
    return test

if __name__ == '__main__':
    start = time.clock()
    # 讀圖片
    for filename in os.listdir(r"./" + img_dir):
        img_list.append(cv2.imread(os.path.join(img_dir,filename)))

    # Threshold
    threshold = 2.0

    # Homography 次數
    times = 1500
    

    for i in range(0,len(img_list)):
        img_list[i] = cv2.resize(img_list[i], (int(img_list[i].shape[1]/10),int(img_list[i].shape[0]/10)), interpolation = cv2.INTER_AREA)

    stitched_img = []

    # stitched_img[:] = img_list[:8]
    # k = 0
    # for i in range(3):
    #     result_list = []
    #     for j in range(0, len(stitched_img), 2):
    #         print('index = ', k)
    #         # if i == 0:
    #         #     img0 = Cylindrical_Projector.CylindricalProjection(stitched_img[j],focal_length_all[j])
    #         #     img1 = Cylindrical_Projector.CylindricalProjection(stitched_img[j+1],focal_length_all[j+1])
    #         # else:
    #         img0 = stitched_img[j]
    #         img1 = stitched_img[j+1]
    #         img0 = ClearBorder(img0)
    #         img1 = ClearBorder(img1)
    #         height = min(img0.shape[0],img1.shape[0])
    #         img0 = img0[0:height,:]
    #         img1 = img1[0:height,:]

    #         cv2.imshow('img0',img0)
    #         cv2.imshow('img1',img1)
    #         cv2.waitKey(0)
    #         if i == 1:
    #             keypoint_count = 3000
    #         elif i == 2:
    #             keypoint_count = 5000
    #             threshold = 1.5

            
            
    #         print('\n --- 找{}個特徵點, 圖 {} --- '.format(keypoint_count,i))
    #         _ , mask0, corner0 = feature_detector.Corner_detection(img0,keypoint_count)
    #         description0,description_index0 = feature_descriptor.MSOP_descriptor_vector(img0,mask0,keypoint_count)

    #         print('\n --- 找{}個特徵點, 圖 {} --- '.format(keypoint_count,i+1))
    #         _ , mask1, corner1 = feature_detector.Corner_detection(img1,keypoint_count)
    #         description1,description_index1 = feature_descriptor.MSOP_descriptor_vector(img1,mask1,keypoint_count)

    #         print('\n --- 找Match點 --- ')
    #         correspond = feature_matcher.Matcher(img0,img1,description0,description1,description_index0,description_index1, threshold)

    #         print('\n --- Align圖片 --- ')
    #         result = Alignment.MultiBandBlending(img0, img1, correspond, times)
    #         print('\n --- Align 結束 ---')
    #         result_list.append(result)

    #         cv2.imwrite('./20220427result/panorama_' + str(k) + '.jpg',result_list[len(result_list)-1])
    #         k = k+1
    #     stitched_img = result_list
    #     print('lenght = ', len(stitched_img))

    for i in range(0,len(img_list)-3):
        if i == 0:
            img0 = Cylindrical_Projector.CylindricalProjection(img_list[0],focal_length_all[i])
            img1 = Cylindrical_Projector.CylindricalProjection(img_list[1],focal_length_all[i+1])
        else:
            img0 = stitched_img[i-1]
            img1 = Cylindrical_Projector.CylindricalProjection(img_list[i+1],focal_length_all[i])

        cut = [0, 20, 200, 1000, 2000]
        img0 = ClearBorder(img0, cut[i])
        img1 = ClearBorder(img1, 0)
        height = min(img0.shape[0],img1.shape[0])
        img0 = img0[0:height,:]
        img1 = img1[0:height,:]

        # cv2.imshow('img0',img0)
        # cv2.imshow('img1',img1)
        # cv2.waitKey(0)
        print('\n --- 找{}個特徵點, 圖 {} --- '.format(keypoint_count,i))
        _ , mask0, corner0 = feature_detector.Corner_detection(img0,keypoint_count)
        description0,description_index0 = feature_descriptor.MSOP_descriptor_vector(img0,mask0,keypoint_count)

        print('\n --- 找{}個特徵點, 圖 {} --- '.format(keypoint_count,i+1))
        _ , mask1, corner1 = feature_detector.Corner_detection(img1,keypoint_count)
        description1,description_index1 = feature_descriptor.MSOP_descriptor_vector(img1,mask1,keypoint_count)

        print('\n --- 找Match點 --- ')
        correspond = feature_matcher.Matcher(img0,img1,description0,description1,description_index0,description_index1, threshold)

        print('\n --- Align圖片 --- ')
        result = Alignment.MultiBandBlending(img0, img1, correspond, times)
        print('\n --- Align 結束 ---')
        stitched_img.append(result)

        cv2.imwrite('./result_final/panorama00_' + str(i) + '.jpg',stitched_img[len(stitched_img) - 1])

    
    end = time.clock()
    print("total time: ",end - start)