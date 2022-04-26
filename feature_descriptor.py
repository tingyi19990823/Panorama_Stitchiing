from itertools import count
from stat import FILE_ATTRIBUTE_READONLY
from cv2 import norm, rotate
import numpy as np
import cv2
import math

# input: 40*40的feature,resize成8*8的大小，並做標準化
def Feature_Upsample(feature):
    output = np.zeros((8,8),dtype=float)
    feature = feature / 25
    for i in range(0,8):
        for j in range(0,8):
            output[i,j] = np.sum(feature[i*5:i*5+5,j*5:j*5+5])
    #if np.std(output) == 0.0:
        #print('feature out of bound')
    output = np.divide((output - np.mean(output)),np.std(output),out = np.zeros_like(output), where=np.std(output) != 0)
    
    return output

# input: 彩色圖片
# output: feature description
def MSOP_descriptor_vector(img, mask,feature_count):
    feature = np.zeros((8,8,feature_count))
    feature_index = np.zeros((feature_count,2),dtype = int)
    # feature = np.zeros((8,8,feature_count),dtype=float)
    index = 0

    height , width = img.shape[:2]
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg,(3,3),4.5)
    # cv2.imshow('test',blurImg)
    # cv2.imshow('test2',grayImg)
    # cv2.waitKey(0)
    Ix = cv2.Sobel(blurImg,cv2.CV_64F,1,0)
    Iy = cv2.Sobel(blurImg,cv2.CV_64F,0,1)
    # 找 gradient 的大小和旋轉角度
    _ , angle = cv2.cartToPolar(Ix,Iy,angleInDegrees = True)

    lineImg = np.copy(img)
    count = 0
    for i in range(0, height):
        for j in range(0, width):
            if mask[i,j] == 255:
                # 畫 gradient 的線
                # desx = int(i - 20 * math.cos(math.radians(angle[i,j])))
                # desy = int(j - 20 * math.sin(math.radians(angle[i,j])))
                # lineImg = cv2.line(lineImg,(j,i),(desy,desx),(0,0,255),2)
                # count = count + 1
                rotate_mat = cv2.getRotationMatrix2D((j,i),-1*angle[i,j],1) # (旋轉中心),旋轉角度,縮放比例
                # rotate_mat = cv2.getRotationMatrix2D((width/2+200,height/2),45,1) # (旋轉中心),旋轉角度,縮放比例
                #print(rotate_mat.shape)
                img_rotate = cv2.warpAffine(blurImg,rotate_mat,(width,height), flags = cv2.INTER_NEAREST)
                height_offset_down = i + 20
                height_offset_up = i - 20
                width_offset_right = j + 20
                width_offset_left = j - 20
                rotate_feature = img_rotate[height_offset_up:height_offset_down, width_offset_left:width_offset_right]
                # cv2.imshow('test',rotate_feature)
                # cv2.waitKey(0)
                # print(rotate_feature.shape)
                # print(rotate_feature)
                # cv2.imwrite(os.path.join('feature_check','40_'+str(index)+'.jpg'),rotate_feature)

                '''
                debug 用
                if rotate_feature.shape[0] != 40 or rotate_feature.shape[1] != 40:
                    print('not enough size, need 40*40')
                    test = np.copy(img_rotate)
                    test[height_offset_up:height_offset_down, width_offset_left:width_offset_right,:] = 255
                    print(angle[i,j])
                    print(i)
                    print(j)
                    print(width_offset_left)
                    print(width_offset_right)
                    print(rotate_feature.shape)
                    # print(rotate_feature)
                    cv2.imshow('test',test)
                    cv2.waitKey(0)
                    print(rotate_feature)
                '''

                feature_resize = Feature_Upsample(rotate_feature)
                
                # if index == 0:
                #     print(i,j)
                #     print('40*40: ',rotate_feature)
                #     print('height_offset_up :',height_offset_up)
                #     print('height_offset_down: ',height_offset_down)
                #     print('width_offset_left: ', width_offset_left)
                #     print('width_offset_left: ',width_offset_right)
                #     print('shape: ',rotate_feature.shape)
                #     print('8*8: ',feature_resize)
                    # cv2.imshow('test',rotate_feature)
                    # cv2.waitKey(0)
                    
                feature[:,:,index] = feature_resize
                feature_index[index,0] = i
                feature_index[index,1] = j
                index = index + 1
    
    
    # cv2.imwrite(os.path.join('./result_pic','line.jpg'),lineImg)
    # print(count)
    # cv2.imshow('test',lineImg)
    # cv2.waitKey(0)

    return feature, feature_index


if __name__ == '__main__':
    img1 = cv2.imread('./lin_test/prtn00.jpg')
    #mask_npy1 = np.load('./lin_test/mask_0_0.npy')
    mask_npy1 = np.load('./lin_test/my_array0.npy').astype(int)
    img2 = cv2.imread('./lin_test/prtn01.jpg')
    #mask_npy2 = np.load('./lin_test/mask_1_0.npy')
    mask_npy2 = np.load('./lin_test/my_array1.npy').astype(int)
    #print(img1.shape)
    print(mask_npy1.dtype)

    count1 = (mask_npy1 == 255).sum()
    count2 = (mask_npy2 == 255).sum()
    print(print(mask_npy1[mask_npy1 != 0]))
    print(count1)
    print(count2)

    feature1,index1 = MSOP_descriptor_vector(img1,mask_npy1,count1)
    feature2,index2 = MSOP_descriptor_vector(img2,mask_npy2,count2)

    np.save('./lin_test/feature_0.npy',feature1)
    np.save('./lin_test/feature_index0.npy',index1)
    np.save('./lin_test/feature_1.npy',feature2)
    np.save('./lin_test/feature_index1.npy',index2)