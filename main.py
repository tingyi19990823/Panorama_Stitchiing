from unittest import result
import feature_detector
import cv2
import os
import time

img_dir = "./grail"
img_list = []

result_dir = "./result_grail"

if __name__ == '__main__':
    start = time.clock()
    # 讀圖片
    for filename in os.listdir(r"./" + img_dir):
        img_list.append(cv2.imread(os.path.join(img_dir,filename)))
    # 找 corner
    index = 0
    for img in img_list:
        grayimg, maskimg, cornerimg = feature_detector.Corner_detection(img)
        # cv2.imwrite(os.path.join(result_dir,'gray' + str(index) + '.jpg'),grayimg)
        cv2.imwrite(os.path.join(result_dir,'mask' + str(index) + '.jpg'),maskimg)
        cv2.imwrite(os.path.join(result_dir,'corner' + str(index) + '.jpg'),cornerimg)
        index = index + 1
    end = time.clock()
    print("total time: ",end - start)