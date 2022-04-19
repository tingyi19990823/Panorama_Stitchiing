from unittest import result
import feature_detector
import feature_descriptor
import cv2
import os
import time
import numpy as np

# img_dir = "./grail"
img_dir = "./pic"
img_list = []

# result_dir = "./result_grail"
result_dir = "./result_pic"

keypoint_count = 250

if __name__ == '__main__':
    start = time.clock()
    # 讀圖片
    for filename in os.listdir(r"./" + img_dir):
        img_list.append(cv2.imread(os.path.join(img_dir,filename)))
    
    index = 0
    for img in img_list:
        # 找 corner
        grayimg, maskimg, cornerimg = feature_detector.Corner_detection(img,keypoint_count)
        # cv2.imwrite(os.path.join(result_dir,'gray' + str(index) + '.jpg'),grayimg)
        cv2.imwrite(os.path.join(result_dir,'mask' + str(index) + '.jpg'),maskimg)
        cv2.imwrite(os.path.join(result_dir,'corner' + str(index) + '.jpg'),cornerimg)
        np.save(os.path.join(result_dir,'mask_'+ str(index)),maskimg)

        # feature descriptor
        description = feature_descriptor.MSOP_descriptor_vector(img,maskimg,keypoint_count)
        np.save(os.path.join(result_dir,'feature_'+ str(index)),description)
        index = index + 1
    end = time.clock()
    print("total time: ",end - start)