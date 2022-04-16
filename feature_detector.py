from cv2 import blur
import numpy as np
import cv2
import os

def CalHLMatrix(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    blur_img = cv2.GaussianBlur(img,(3,3),1.0)
    Ix = cv2.Sobel(blur_img,cv2.CV_64F,1,0,ksize = 3)
    Iy = cv2.Sobel(blur_img,cv2.CV_64F,0,1,ksize = 3)
    Ix2 = Ix * Ix
    
    Iy2 = Iy * Iy
    


# if __name__ == '__main__':
#     # 讀圖片
#     for filename in os.listdir(r"./" + img_dir):
#         img_list.append(cv2.imread(os.path.join(img_dir,filename)))


#     blur_img = cv2.GaussianBlur(img_list[0],(3,3),1.0)
#     Ix = cv2.Sobel(blur_img,cv2.CV_64F,1,0,ksize = 3)
#     Iy = cv2.Sobel(blur_img,cv2.CV_64F,0,1,ksize = 3)