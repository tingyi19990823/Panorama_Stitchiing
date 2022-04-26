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

focal_length = 705  # 704.916

if __name__ == '__main__':
    start = time.clock()
    # 讀圖片
    for filename in os.listdir(r"./" + img_dir):
        img_list.append(cv2.imread(os.path.join(img_dir,filename)))
    for i in range(0,len(img_list)):
        img_list[i] = cv2.resize(img_list[i], (int(img_list[i].shape[1]/10),int(img_list[i].shape[0]/10)), interpolation = cv2.INTER_AREA)
    
    # index = 0
    # for img in img_list:
    #     # 找 corner
    #     grayimg, maskimg, cornerimg = feature_detector.Corner_detection(img,keypoint_count)
    #     # cv2.imwrite(os.path.join(result_dir,'gray' + str(index) + '.jpg'),grayimg)
    #     cv2.imwrite(os.path.join(result_dir,'mask' + str(index) + '.jpg'),maskimg)
    #     cv2.imwrite(os.path.join(result_dir,'corner' + str(index) + '.jpg'),cornerimg)
    #     np.save(os.path.join(result_dir,'mask_'+ str(index)),maskimg)

    #     # feature descriptor
    #     print('description {} ...'.format(index))
    #     description,description_index = feature_descriptor.MSOP_descriptor_vector(img,maskimg,keypoint_count)
    #     np.save(os.path.join(result_dir,'feature_'+ str(index)),description)
    #     np.save(os.path.join(result_dir,'feature_index'+ str(index)),description_index)
    #     index = index + 1

    stitched_img = []
    for i in range(0,len(img_list)-1):
        if i == 0:
            img0 = Cylindrical_Projector.CylindricalProjection(img_list[0],focal_length)
            img1 = Cylindrical_Projector.CylindricalProjection(img_list[1],focal_length)
        else:
            img0 = Cylindrical_Projector.CylindricalProjection(stitched_img[i-1],focal_length)
            img1 = Cylindrical_Projector.CylindricalProjection(img_list[i+1],focal_length)
        cv2.imshow('img0',img0)
        cv2.imshow('img1',img1)
        cv2.waitKey(0)
        print('\n --- 找{}個特徵點, 圖 {} --- '.format(keypoint_count,i))
        _ , mask0, corner0 = feature_detector.Corner_detection(img0,keypoint_count)
        description0,description_index0 = feature_descriptor.MSOP_descriptor_vector(img0,mask0,keypoint_count)

        print('\n --- 找{}個特徵點, 圖 {} --- '.format(keypoint_count,i+1))
        _ , mask1, corner1 = feature_detector.Corner_detection(img1,keypoint_count)
        description1,description_index1 = feature_descriptor.MSOP_descriptor_vector(img1,mask1,keypoint_count)

        print('\n --- 找Match點 --- ')
        correspond = feature_matcher.Matcher(img0,img1,description0,description1,description_index0,description_index1)

        print('\n --- Align圖片 --- ')
        result = Alignment.MultiBandBlending(img0, img1, correspond)
        print('\n --- Align 結束 ---')
        stitched_img.append(result)

        cv2.imwrite('./result_final/panorama_' + str(i) + '.jpg',stitched_img[len(stitched_img) - 1])

    end = time.clock()
    print("total time: ",end - start)