
from PyQt_UI.MainWindow  import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
import NewLine
from PyQt_UI.regionSplit import Ui_Dialog
import cv2
import sys
import os
import numpy as np
from PyQt5.QtCore import QObject , pyqtSignal

class showMainWindow(QMainWindow,Ui_MainWindow):
    # 多继承方式，继承界面父类,分离逻辑代码和界面代码
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        #加载应用程序图标
        icon = QIcon()
        icon.addPixmap(QtGui.QPixmap('../icon/mainWindow.ico'))
        self.setWindowIcon(icon)

        # 成员变量
        self.Threshold = 127
        self.OrImgPath = ''
        self.BinaryImgPath = ''
        self.MergeImgPath=''
        self.diffCh=30
        self.bigCh=0     #R通道
        self.smallCh=1   #G通道

        #初始化部分界面（单阈值图标）
        self.slider_BinaryThre.setMinimum(0)
        self.slider_BinaryThre.setMaximum(255)
        self.slider_BinaryThre.setValue(127)
        self.slider_BinaryThre.setSingleStep(1)

        # 实例化子界面,用于显示子界面
        self.regionSplitWindow = regionSplitWindow()

        # 绑定槽函数
        self.openImg.triggered.connect(self.readImg)
        self.slider_BinaryThre.valueChanged.connect(self.threshold_change)
        self.button_ThreCon.clicked.connect(self.binaryImg)  #确定二值化按钮
        self.saveImg.triggered.connect(self.saveBinaryImg)
        self.channel_Diff.triggered.connect(self.channel_Difference)
        self.channel_Diff_2.triggered.connect(self.channel_Difference)
        self.regionSplitWindow.button_cancel.clicked.connect(self.closeWindow)
        self.layer_Overlay.triggered.connect(self.layerOverlay)
        self.saveAsTXT.triggered.connect(self.BinaryImgToTxt)

    def readImg(self):
        # 从指定目录打开图片（*.jpg *.gif *.png *.jpeg），返回路径
        image_file, _ = QFileDialog.getOpenFileName(self, '打开图片', 'E:\\Projects\\Items\\ChipTest\\images', 'Image files (*.jpg *.gif *.png *.jpeg)')
        # 缩放图片 设置标签的图片
        jpg = QtGui.QPixmap(image_file).scaled(self.label_OrImg.width(), self.label_OrImg.height())
        if(image_file==''):
            return 0
        self.OrImgPath = image_file
        #print(self.OrImgPath)
        #print(self.label_OrImg.width(),self.label_OrImg.height())
        self.label_OrImg.setPixmap(jpg)

    def saveBinaryImg(self):
        #保存（*.jpg *.gif *.png *.jpeg）图片，返回路径
        image_file,_=QFileDialog.getSaveFileName(self,'保存二值化图片','C:\\','Image files (*.jpg *.gif *.png *.jpeg)')
        #设置标签的图片
        print(image_file)
        #img=self.label_BinaryImg.pixmap().toImage()
        if(image_file==''):
            return 0
        img=cv2.imread(self.BinaryImgPath)
        cv2.imwrite(image_file,img)

    def binaryImg(self):
        if(self.OrImgPath==''):
            return 0
        img,binaryImgPath = NewLine.image_binarization(self.OrImgPath, self.Threshold)
        img = QtGui.QPixmap(binaryImgPath).scaled(self.label_BinaryImg.width(), self.label_BinaryImg.height())
        self.BinaryImgPath = binaryImgPath
        self.label_BinaryImg.setPixmap(img)

    def getThreshold(self):
        return self.Threshold

    def threshold_change(self):
        self.Threshold=self.slider_BinaryThre.value()
        self.label_ThresholdValue.setText(str(self.Threshold))
        print(self.Threshold)

    #子菜单点击槽函数
    def channel_Difference(self):
        # 将子界面的信号与接受数据的函数连接
        self.regionSplitWindow.signal_diffCh.connect(self.getChDiffData)
        self.regionSplitWindow.exec()
        #单个对话框可使用以下代码，无需通过复杂的信号槽传递参数，使用更为简洁方便
        # num, ok = QInputDialog.getInt(self, 'Integer input dialog', '输入数字')
        # num1, ok1 = QInputDialog.getInt(self, 'Integer input dialog', '输入')
        # if ok and num:
        #     self.GetIntlineEdit.setText(str(num))

    #子界面信号对应的槽函数
    def getChDiffData(self,diffCh):
        print('diffch的值为：',diffCh)
        d=int(diffCh)
        img = cv2.imread(self.OrImgPath)

        #按照第三维取最大数组和最小数组，用于计算标志位
        max_RGB= np.amax(img, axis=2)
        min_RGB = np.amin(img, axis=2)
        #分别获取三个标志位矩阵，将其相乘
        diff_RGB=((max_RGB-min_RGB)<d)
        max_RGB=(max_RGB<150)
        min_RGB=(min_RGB>50)
        rec_RGB=(diff_RGB*max_RGB*min_RGB)
        #不能用标志矩阵直接和img三维数组相乘，与numpy数组存在一定差异
        #img=(img*diff_RGB)
        img[:,:,0]=img[:,:,0]*rec_RGB
        img[:, :, 1] = img[:, :, 1] * rec_RGB
        img[:, :, 2] = img[:, :, 2] * rec_RGB
        img[img>0]=255

        #遍历方法太慢，最好不要使用
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):

        #默认将二值化的img图片写入原始图片的目录下
        content, tempfilename = os.path.split(self.OrImgPath)
        filename, extension = os.path.splitext(tempfilename)
        filename = filename + str('_ChDiffBinary') + extension
        filepath = os.path.join(content, filename)
        filepath = filepath.replace('\\', '/')
        cv2.imwrite(filepath,img)

        #二值化图片路径存入成员变量
        self.BinaryImgPath=filepath

        # 将图片转成BGR模式; img_rgb.shape[1] * img_rgb.shape[2]必须添加  不然照片是斜着的
        img=cv2.resize(img,(self.label_BinaryImg.height(), self.label_BinaryImg.width()))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        QtImg = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0],img_rgb.shape[1] * img_rgb.shape[2],
                                  QtGui.QImage.Format_RGB888)

        # 显示图片到label中
        self.label_BinaryImg.setPixmap(QtGui.QPixmap.fromImage(QtImg))

    def closeWindow(self):
        self.regionSplitWindow.close()

    #OpenCV图片数据格式转为QImage数据格式
    def cvimg_to_qtimg(self,cvimg):
        height, width, depth = cvimg.shape
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        cvimg = QtGui.QImage(cvimg.data, width, height,width*depth, QtGui.QImage.Format_RGB888)
        return cvimg


    def layerOverlay(self):
        # self.messageBox = QMessageBox()
        # self.messageBox.setWindowTitle('图层叠加')
        # icon = QIcon()
        # icon.addPixmap(QtGui.QPixmap('../icon/mainWindow.ico'))
        # self.messageBox.setWindowIcon(icon)
        # self.messageBox.setText('是否将处理后的图层叠加到原图层上？')
        # self.messageBox.addButton(QPushButton('确定'), QMessageBox.YesRole)
        # self.messageBox.addButton(QPushButton('取消'), QMessageBox.NoRole)
        # self.messageBox.exec_()
        # print(self.messageBox.result())
        self.box=QMessageBox(QMessageBox.Question,"图层叠加", "是否将处理后的图层叠加到原图层上？")
        icon = QIcon()
        icon.addPixmap(QtGui.QPixmap('../icon/mainWindow.ico'))
        self.box.setWindowIcon(icon)
        qyes = self.box.addButton(self.tr("确定"), QMessageBox.YesRole)
        qno = self.box.addButton(self.tr("取消"), QMessageBox.NoRole)
        self.box.exec_()
        if self.box.clickedButton() == qyes:
            if(self.OrImgPath=='' or self.BinaryImgPath=='' ):
                QMessageBox.warning(self, "提示", "未读取图层照片或未处理照片", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                return 0
            orImg = cv2.imread(self.OrImgPath)
            binbaryImg = cv2.imread(self.BinaryImgPath)
            #将二值化图片改成黄色
            np.where(binbaryImg[:, :, 1] > 0, binbaryImg[:, :, 1], 255)
            np.where(binbaryImg[:, :, 2] > 0, binbaryImg[:, :, 2], 255)
            binbaryImg[:, :, 0] = 0

            #保存图像半透明式覆盖原图层照片及其路径
            img = cv2.addWeighted(orImg, 0.95, binbaryImg, 0.05, 0)
            content, tempfilename = os.path.split(self.OrImgPath)
            filename, extension = os.path.splitext(tempfilename)
            filename = filename + str('_MergeImg') + extension
            filepath = os.path.join(content, filename)
            filepath = filepath.replace('\\', '/')
            self.MergeImgPath=filepath
            cv2.imwrite(filepath ,img)
            #缩放尺寸，标签显示图片
            img = cv2.resize(img, (self.label_BinaryImg.height(), self.label_BinaryImg.width()))
            self.label_BinaryImg.setPixmap(QtGui.QPixmap.fromImage(self.cvimg_to_qtimg(img)))
        else:
            return 0

    def BinaryImgToTxt(self):
        if(self.BinaryImgPath==''):
            return 0
        imgPath, _ = QFileDialog.getSaveFileName(self, '将二值化图片另存为TXT', 'C:\\', 'TXT File (*.txt )')
        if(imgPath==''):
            return 0
        print('开始检测：')
        img=cv2.imread(self.BinaryImgPath)
        # 彩色通道使用#,这里可以适当修改
        img_b = img[:, :, 0]
        # 单通道使用#
        # img_b = img
        size_x = img_b.shape[1]  # 像数列数
        size_y = img_b.shape[0]  # 像数行数
        boundary = 10
        count = 0
        self.CreateTxtFileHead(imgPath)
        rec_flag = np.zeros([size_y, size_x])  # 矩阵标志位，标志当前像素点是否被之前的矩阵圈住
        for row_index in range(boundary, size_y - boundary, 1):
            for col_index in range(boundary, size_x - boundary, 1):
                if (rec_flag[row_index, col_index]):
                    continue
                else:
                    col_end = col_index
                    for index in range(col_index, size_x - boundary, 1):
                        if (img_b[row_index, index] < 127):
                            col_end = index
                            break
                    rec_flag[row_index, col_index:col_end] = 1
                    if (col_index == col_end):
                        continue
                    else:
                        self.CreateTxtFile(imgPath,(-1 * row_index), col_index, col_end)
                        count = count + 1
        self.CreateTxtFileTail(imgPath)
        QMessageBox.information(self, "提示", "已成功导出TXT版图数据",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        return 0

    def CreateTxtFile(self,imgPath,row_index, col_index, col_end):

        vertex1 = str(col_index) + ':' + str(row_index)
        vertex2 = str(col_end) + ':' + str(row_index)
        with open(imgPath, 'a') as file_handle:
            # file_handle.write('BOUNDARY')  # 开始写入数据
            file_handle.write('\n')  # 自动换行
            file_handle.write('PATH\n')
            file_handle.write('LAYER 50\n')
            file_handle.write('DATATYPE 0\n')
            # 给甲方的时候 WIDTH=10 ,WIDTH=1代表0.002um 一个像素点对应0.02um
            file_handle.write('WIDTH 1\n')
            file_handle.write('XY' + ' ' + vertex1)  # 左上角顶点
            file_handle.write('\n')
            file_handle.write(vertex2)  # 右上角顶点
            file_handle.write('\n')
            file_handle.write('ENDEL')
            file_handle.write('\n')
        return 0

    def CreateTxtFileHead(self,imgPath):
        with open(imgPath, 'w') as file_handle:
            file_handle.write('HEADER 600 ')  # 开始写入数据
            file_handle.write('\n')  # 自动换行
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

    def CreateTxtFileTail(self,imgPath):
        with open(imgPath, 'a') as file_handle:
            file_handle.write('ENDSTR')  # 开始写入数据
            file_handle.write('\n')  # 自动换行
            file_handle.write('ENDLIB')
            file_handle.write('\n')
        return 0


#子界面类，用于显示父窗口中的子窗口，解决QInputDialog一次只能输入一个弹窗的困扰
class regionSplitWindow(QDialog,Ui_Dialog):
    #创建信号，用于界面传参
    signal_diffCh=pyqtSignal(str)    #通道差值的信号
    #传递多个参数 用下列代码
    #signal_diffCh = pyqtSignal(str,str)

    # 多继承方式，继承界面父类,分离逻辑代码和界面代码
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint)
        self.setWindowTitle("RGB通道的差值二值化")
        self.button_confirm.clicked.connect(self.dataTrans)

    def dataTrans(self):
        #print(self.comboBox_bigCh.currentIndex(),self.comboBox_smallCh.currentIndex(),self.lineEdit_diffCh.text())
        #传递多个参数，用下列代码
        #self.signal_diffCh.emit(str(self.comboBox_bigCh.currentIndex()),
        #                      str(self.comboBox_smallCh.currentIndex()),self.lineEdit_diffCh.text())
        self.signal_diffCh.emit(self.lineEdit_diffCh.text())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    showMainWindow = showMainWindow()
    showMainWindow.show()
    app.exec_()