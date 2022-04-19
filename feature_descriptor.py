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
    output = (output - np.mean(output)) / np.std(output)
    return output

# input: 彩色圖片
# output: feature description
def MSOP_descriptor_vector(img, mask,feature_count):
    feature = np.zeros((8,8,feature_count),dtype=float)
    index = 0

    height , width = img.shape[:2]
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg,(3,3),4.5)
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
                feature[:,:,index] = feature_resize
                index = index + 1
    
    
    # cv2.imwrite(os.path.join('./result_pic','line.jpg'),lineImg)
    # print(count)
    # cv2.imshow('test',lineImg)
    # cv2.waitKey(0)

    return feature


if __name__ == '__main__':
    img = cv2.imread('./pic/test.jpg')
    mask_npy = np.load('./result_pic/mask_0.npy')
    # img = cv2.imread('./pic/test2.jpg')
    # mask_npy = np.load('./result_pic/mask_1.npy')
    MSOP_descriptor_vector(img,mask_npy,250)