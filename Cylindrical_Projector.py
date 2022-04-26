from pickletools import uint8
import numpy as np
import cv2
import math
import os

focal_length = 705  # 704.916
cylinder_radius = focal_length

def CylindricalProjection(img,cylinder_radius):
    height , width = img.shape[:2]

    projectedImg = np.zeros((height,width,3),dtype = np.uint8)

    for i in range(height):
        for j in range(width):
            width_project = int((cylinder_radius * math.atan((j-width/2)/cylinder_radius)) + width/2)
            height_project =  int((cylinder_radius * ((i-height/2)/math.sqrt(math.pow(j-width/2,2) + math.pow(cylinder_radius,2))))+height/2)
            projectedImg[height_project,width_project] = img[i,j]

    _, thresh = cv2.threshold(cv2.cvtColor(projectedImg, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)

    return projectedImg[y:y+h,x:x+w]


if __name__ == '__main__':
    # for filename in os.listdir(r"./parrington"):
    #     print('projecting {} ... \n'.format(filename))
    #     img = cv2.imread('./parrington/' + filename)
    #     projectedImg = CylindricalProjection(img,cylinder_radius)
    #     cv2.imwrite('./result_project/project_' + filename,projectedImg)
    # print(img.dtype)
    # cv2.imshow('test1',img)
    # cv2.waitKey(0)

    img = cv2.imread('./parrington/prtn00.jpg')
    for i in range(300,focal_length,20):
        projectedImg = CylindricalProjection(img,i)
        cv2.imwrite('./result_project/project_new_'+str(i)+'.jpg',projectedImg)
        print(i)