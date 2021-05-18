import cv2
import numpy as np
import pandas as pd
import os
import PIL as pl
from matplotlib import pyplot as plt

#x,y为输入的起始点坐标 img为图像

def lineDect(img,x,y):
    size_x=img.shape[1]    #像素的列
    size_y=img.shape[0]    #像素的行
    #rflag = np.zeros([size_y, size_x])  # 矩阵标志位，表明当前像素是否已被检测
    #print(np.sum(rec_flag==1))
    img_b = img[:, :, 0]
    boundary = 20 #边界值
    small_col=x
    big_col=x
    small_row = y
    big_row = y

    #向上找最小行索引
    for in_row in range(y,boundary, -1):
        if (img_b[in_row, x]>127):
            small_row = in_row
            #rflag[in_row,x]=1
        else:
            break

    #向下找最大的行索引
    for in_row in range(y, size_y-boundary, 1):
        if(img_b[in_row, x]>127 ):
            big_row=in_row
            #rflag[in_row, x] = 1
        else:
            break

    #向右找最大的列索引
    for in_col in range(x,size_x-boundary,1):
      #if (np.all(rflag[small_row:big_row, in_col])== 0):
            rec = img_b[small_row:big_row, in_col]
            #print(rec.shape)
            h=big_row-small_row
            all_sum = np.sum(rec>127)
            #print(h,all_sum)
            if all_sum > (h * 0.85):       #判断线段上点是否有90 %
                big_col = in_col
                #rflag[small_row:big_row, in_col]=1
            else:
                break

    # 向左找最大的列索引
    for in_col in range(x,boundary,-1):
        #if (np.all(rflag[small_row:big_row, in_col]) == 0):
            rec = img_b[small_row:big_row, in_col]
            h = big_row - small_row
            all_sum = np.sum(rec>127)
            if all_sum > (h * 0.85):  # 判断线段上点是否有90 %
                small_col = in_col
                #rflag[small_row:big_row, in_col]=1
            else:
                break

    point = [small_row,big_row,small_col,big_col]
    #print(point)
    return point

def GlobalDect(img):
    print('开始检测：')
    # 彩色通道使用#
    img_b = img[:, :, 0]
    #单通道使用#
    #img_b = img
    # print(np.sum(img_b>0))
    #print(img_b.shape)
    size_x = img_b.shape[1]  #像数列数
    size_y = img_b.shape[0]  #像数行数
    print(size_x,size_y)
    boundary=10
    #df=pd.DataFrame(columns=['point_1','point_2','point_3','point_4','point_1'])
    #CoordinateSet=pd.DataFrame(columns=['small_col','small_row','big_col','big_row'])
    count=0
    CreateTxtFileHead()
    rec_flag=np.zeros([size_y,size_x])  #矩阵标志位，标志当前像素点是否被之前的矩阵圈住
    for row_index in range(boundary,size_y-boundary,1):
        for col_index in range(boundary,size_x-boundary,1):
            if(rec_flag[row_index,col_index]):
                continue
            else:
                #point=lineDect(img,col_index,row_index)
                #rec_flag[point[0]:point[1],point[2]:point[3]]=1
                col_end=col_index
                for index in range(col_index,size_x-boundary,1):
                    if(img_b[row_index,index]<127):
                        col_end=index
                        break
                rec_flag[row_index,col_index:col_end]=1
                if (col_index==col_end):
                    continue
                else:
                    #cv2.rectangle(img, (row_index, col_index), (row_index, col_end), (0, 0, 255), 1)
                    #保存矩形四个顶点的像素坐标,注意横纵坐标  这里保存的是(y,x)形式，即是先读行后读列
                    #df.loc[df.shape[0]] = {"point_1": (point[0], point[2]),"point_2": (point[0], point[3]),"point_3": (point[1], point[3]),"point_4": (point[1], point[2]),"point_1": (point[0], point[2])}
                    #CoordinateSet.loc[CoordinateSet.shape[0]]={"small_col": point[2],"small_row": point[0],"big_col": point[3],"big_row": point[1]}
                    CreateTxtFile((-1*row_index),col_index,col_end)
                    count=count+1
    CreateTxtFileTail()
    print(count)
    #print(np.sum(rec_flag==1),np.sum(rec_flag==0),np.sum(img_b>0))
    #print("检测占比:",'%.2f%%' % (np.sum(rec_flag == 1)/np.sum(img_b > 0)*100))
    #df.to_csv('data/data_point.csv')
    #CoordinateSet.to_csv('data/CoordinateSet.csv')

def image_binarization(path,Threshold):
    #获取文件名
    print(Threshold)
    img=cv2.imread(path)
    content, tempfilename = os.path.split(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #指定阈值 灰度二值化
    retval, dst = cv2.threshold(gray, Threshold, 255, cv2.THRESH_BINARY)
    # 最大类间方差法(大津算法)，thresh会被忽略，自动计算一个阈值
    #retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #中值去噪，也称为椒盐去噪
    blur = cv2.medianBlur(dst, 5)
    #ImgShow(blur)
    #拼接文件名
    filename, extension = os.path.splitext(tempfilename)
    filename=filename+str('_binary')+extension
    filepath = os.path.join("../images/ImgSave",filename)
    filepath=filepath.replace('\\','/')
    print(filepath)
    cv2.imwrite(filepath, blur)
    #img=cv2.imread(filepath)
    return blur,filepath

def CreateTxtFile(row_index,col_index,col_end):
    #按照(x,y)格式保存四个顶点 左上角-右上角-右下角-左下角-左上角，一个闭形循环
    vertex1 = str(col_index)+':'+str(row_index)
    vertex2 = str(col_end) + ':' +str(row_index)
    with open('data/data_format.txt', 'a') as file_handle:
        #file_handle.write('BOUNDARY')  # 开始写入数据
        file_handle.write('\n')     # 自动换行
        file_handle.write('PATH\n')
        file_handle.write('LAYER 50\n')
        file_handle.write('DATATYPE 0\n')
        #给甲方的时候 WIDTH=10 WIDTH=1代表0.002um 一个像素点对应0.02um
        file_handle.write('WIDTH 1\n')
        file_handle.write('XY'+' '+vertex1)    #左上角顶点
        file_handle.write('\n')
        file_handle.write(vertex2)  #右上角顶点
        file_handle.write('\n')
        file_handle.write('ENDEL')
        file_handle.write('\n')
    return 0

def CreateTxtFileHead():
    with open('data/data_format.txt', 'w') as file_handle:
        file_handle.write('HEADER 600 ')  # 开始写入数据
        file_handle.write('\n')     # 自动换行
        file_handle.write('BGNLIB 3/10/2021 17:35:23 3/10/2021 17:35:23 ')
        file_handle.write('\n')
        file_handle.write('LIBNAME DEFAULT')
        file_handle.write('\n')
        file_handle.write('UNITS 0.001 1e-009')
        file_handle.write('\n')
        file_handle.write('         ')
        file_handle.write('\n')
        file_handle.write('BGNSTR 3/10/2021 17:35:23 3/10/2021 17:35:23 ')
        file_handle.write('\n')
        file_handle.write('STRNAME VIA1')
        file_handle.write('\n')
        file_handle.write('         ')
        file_handle.write('\n')
    return 0

def CreateTxtFileTail():
    with open('data/data_format.txt', 'a') as file_handle:
        file_handle.write('ENDSTR')  # 开始写入数据
        file_handle.write('\n')     # 自动换行
        file_handle.write('ENDLIB')
        file_handle.write('\n')
    return 0

def EvaluationIndex(img):
    i=img
    img_b = img[:, :, 0]
    CSet=pd.read_csv('CoordinateSet.csv')
    print(CSet.loc[0,'small_col'],CSet.shape[0])
    t=0 #像素点重叠超过90%的个数
    for i in range(0,CSet.shape[0]):
        small_col=CSet.loc[i,'small_col']
        small_row=CSet.loc[i,'small_row']
        big_col=CSet.loc[i,'big_col']
        big_row=CSet.loc[i,'big_row']
        #print(small_col,small_row,big_col,big_row)
        count=0
        for c in range(small_col,big_col+1):
            for r in range(small_row,big_row+1):
                if(img_b[r,c]>127):
                    count=count+1
                else:
                    continue
        ConIndex= count/((big_row-small_row+1)*(big_col-small_col+1))
        #print(ConIndex,count)
        if(ConIndex>0.96):
            t=t+1
            cv2.rectangle(img, (small_col, small_row), (big_col, big_row), (0, 0, 255), 1)
        else:
            cv2.rectangle(img, (small_col, small_row), (big_col, big_row), (0, 255, 0), 1)

    acc=t/CSet.shape[0]
    cv2.putText(img, 'ACC:'+str('%.2f%%' % (acc*100)), (75, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, 'RedLine:' + 'True', (300, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, 'GreenLine:' + 'False', (500, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    print("检测的正确率：",'%.2f%%' % (acc*100))
    print("检测的错误率：", '%.2f%%' % ((1-acc) * 100))
    cv2.namedWindow('image', 0)
    cv2.resizeWindow('image', 900, 700)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

def ImgShow(img):
    cv2.namedWindow('image', 0)
    cv2.resizeWindow('image', 900, 700)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ChannelExtraction(path):
    I = cv2.imread(path)
    i = I[:, :, 2]
    content, tempfilename = os.path.split(path)
    filename, extension = os.path.splitext(tempfilename)
    binary_filename =filename+str('_channel2_binary')+extension
    filename=filename+str('_channel2')+extension
    filepath = os.path.join("images/ImgSave", filename)
    binary_filepath = os.path.join("images/ImgSave", binary_filename)
    filepath = filepath.replace('\\', '/')
    binary_filepath=binary_filepath.replace('\\', '/')
    #print(filepath,filename)
    cv2.imwrite(filepath,i)
    retval, dst = cv2.threshold(i, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(binary_filepath, dst)
    print(retval)
    #dst = cv2.blur(dst, (5,5))
    #ImgShow(dst)
    return dst

def img_seg(path):
    img = cv2.imread(path, -1)
    src=img[:,:,2]
    cnt = 1
    num = 1
    sub_images = []
    sub_image_num = 4
    src_height, src_width = src.shape[0], src.shape[1]
    sub_height = src_height // sub_image_num
    sub_width = src_width // sub_image_num
    for j in range(sub_image_num):
        for i in range(sub_image_num):
            if j < sub_image_num - 1 and i < sub_image_num - 1:
                image_roi = src[j * sub_height: (j + 1) * sub_height, i * sub_width: (i + 1) * sub_width]
            elif j < sub_image_num - 1:
                image_roi = src[j * sub_height: (j + 1) * sub_height, i * sub_width:]
            elif i < sub_image_num - 1:
                image_roi = src[j * sub_height:, i * sub_width: (i + 1) * sub_width]
            else:
                image_roi = src[j * sub_height:, i * sub_width:]
            sub_images.append(image_roi)
    for i, img in enumerate(sub_images):
        cv2.imwrite('images/ImgSave/'+'sub_img_' + str(i) + '.png', img)
        #retval, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        retval, dst = cv2.threshold(img, 105, 255, cv2.THRESH_BINARY)
        cv2.imwrite('images/ImgSave/' + 'sub_img_binary' + str(i) + '.png', dst)
        print(retval)
    return 0

#定义均值滤波器函数
def meanKernel(center,matrix,k):  #center：要替换点的坐标，matrix：目标所在kernel矩阵，k：近邻数
    matrix = matrix.astype('int')
    list1 = [[abs(i-center),i] for i in matrix.ravel()]   #对目标所在的矩阵平铺展开，然后相减，然后排序，前k个对应的
    list1.sort()
    return round(np.array(list1)[:k,1].mean())   #  round() 方法返回浮点数x的四舍五入值。

def KNN(img,kernel,k):
    result_img = img.copy()
    for i in range(result_img.shape[0]-kernel+1):  #（0,3）
        for j in range(result_img.shape[0]-kernel+1):
            #中心点（1,1）= K近邻(KNNF)均值滤波器
            result_img[i+int(kernel/2)][j+int(kernel/2)] = meanKernel(result_img[i+int(kernel/2)][j+int(kernel/2)],img[i:i+kernel,j:j+kernel],5)
    return result_img

def HoleMatch(tem_path,img_path):
   #读取图片时，若加上‘0’则表示读入灰度图，无需cv2.cvtColor函数
   template=cv2.imread(tem_path,0)
   img=cv2.imread(img_path,0)
   h,w=template.shape[:2]  # rows->h, cols->w

   # 相关系数匹配方法: cv2.TM_CCOEFF
   #匹配函数返回的是一幅灰度图，最白的地方表示最大的匹配。
   #使用cv2.minMaxLoc()函数可以得到最大匹配值的坐标，以这个点为左上角角点，模板的宽和高,画矩形
   res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
   left_top = max_loc  # 左上角
   right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
   cv2.rectangle(img, left_top, right_bottom, 255, 10)  # 画出矩形位置
   print(right_bottom)
   # plt.subplot(121), plt.imshow(res, cmap='gray')
   # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
   #
   # plt.subplot(122), plt.imshow(img, cmap='gray')
   # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
   # plt.show()
   ImgShow(img)
   return 0

def mulHoleMatch(tem_path,img_path):
    # 1. 读入原图和模板
    template = cv2.imread(tem_path, 0)
    img = cv2.imread(img_path)
    h, w = template.shape[:2]  # rows->h, cols->w
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 归一化平方差匹配
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.55

    # 这段代码后面会有解释
    loc = np.where(res >= threshold)  # 匹配程度大于80%的坐标y，x
    for pt in zip(*loc[::-1]):  # *号表示可选参数
        right_bottom = (pt[0] + w, pt[1] + h)
        print(right_bottom)
        cv2.rectangle(img, pt, right_bottom, (0, 0, 255), 5)
    ImgShow(img)
    return 0

def EqualizationBinary(path):
    img = cv2.imread(path)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gap=max(img[i,j,:])-min(img[i,j,:])
            if (gap < 30 and max(img[i, j, :]) < 150 and min(img[i,j,:]>50)):
                continue
            else:
                img[i,j,:]=0
    return img

def denoise(path):
    img = cv2.imread(path)
    B, G, R = cv2.split(img)
    for i in range(4042, 8734):
        for j in range(2999, 11010):
            if ((6163 < i < 6288) and (3786 < j < 8640)):
                if (R[i, j] < 63):
                    img[i, j, :] = 0
            else:
                if (R[i, j] < 70):
                    img[i, j, :] = 0
    cv2.imwrite("images/ImgSave/M1_E_R72.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return img

def edgedetection(path):
    img = cv2.imread(path,0)
    edges=cv2.Canny(img,85,200)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    return edges

if __name__ == "__main__":
    path = 'images/OrImg/M1.png'
    #tem_path = 'images/ImgSave/M1_tem.png'
    #img = edgedetection(path)
    img = cv2.imread(path)
    template=img[6580:6754,6864:7670,:]
    cv2.imwrite("images/ImgSave/M1_tem2.png", template,[cv2.IMWRITE_PNG_COMPRESSION, 0])
    #mulHoleMatch(tem_path, path)
    #img=EqualizationBinary(path)


    #通道分离
    # B, G, R = cv2.split(img)
    # plt.subplot(311), plt.hist(R.ravel(), 256, [0, 256]), plt.title('R')
    # plt.subplot(312), plt.hist(G.ravel(), 256, [0, 256]), plt.title('G')
    # plt.subplot(313), plt.hist(B.ravel(), 256, [0, 256]), plt.title('B')
    # plt.show()


    #ImgShow(template)

    #img_seg(path)
    #img=ChannelExtraction(path)
    #result=KNN(img,7,5)

    #i = cv2.imread(path)
    #dst = cv2.blur(i, (7, 7))
    #cv2.imwrite("images/ImgSave/M1_channel2_binary_blur.jpg", dst)

    #GlobalDect(i)
    #img=ChannelExtraction(path)
    #ImgShow(img)
    #img=cv2.imread(path)
    #i = image_binarization(img, path)

    #i=image_binarization(img,path)
    #ImgShow(i)
    #EvaluationIndex(i)



