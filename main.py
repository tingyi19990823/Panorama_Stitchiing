import feature_detector
import cv2
import os

img_dir = "./pic"
img_list = []



if __name__ == '__main__':
    # 讀圖片
    for filename in os.listdir(r"./" + img_dir):
        img_list.append(cv2.imread(os.path.join(img_dir,filename)))

    for img in img_list:
        feature_detector.CalHLMatrix(img)