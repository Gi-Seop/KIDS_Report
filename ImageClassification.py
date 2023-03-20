import os.path
import shutil
import sys

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QSpinBox, QLabel
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torchvision import transforms
from torchvision import models
import numpy as np
import pickle
import PIL
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
import warnings
Image.MAX_IMAGE_PIXELS = None
import cv2
from PIL import Image, ImageOps
import csv



def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path("ImageClassifier.ui")
form_class = uic.loadUiType(form)[0]

original_path = os.getcwd()
path_nowVector = os.path.join(os.getcwd(),'nowVector.pkl')


inputDim = (224, 224)
transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])
error_files = []

def ImageConvertAll(file_path: str):
    inputDir = file_path
    inputDirCNN = os.path.join(inputDir, "inputfigCNN")
    file_list = os.listdir(inputDir)
    os.makedirs(inputDirCNN, exist_ok=True)
    vectors = os.path.join(file_path, "vectors")
    os.makedirs(vectors, exist_ok=True)

    for ii in tqdm(range(len(file_list)), desc="Convert Image"):
        try:
            I = Image.open(os.path.join(inputDir, file_list[ii]))
            I = I.convert('RGB')
        except:
            try:
                I.close()
                shutil.move(inputDir + '/' + file_list[ii], file_path + '/' + 'error_read')
                continue
            except Exception as e:
                print(e)
                continue
        try:
            newI = transformationForCNNInput(I)  # CNN에 맞게 inputDim 크기로 변환
        except:
            I.close()
            shutil.move(inputDir + '/' + file_list[ii], file_path + '/' + 'error_read')
            continue
        try:
            exif = I.info['exif']
            newI.save(os.path.join(inputDirCNN, file_list[ii]), exif=exif)
        except:
            newI.save(os.path.join(inputDirCNN, file_list[ii]))
        newI.close()
        I.close()

def ImageConvert(file_path: str):
    file_name = file_path.split('/')[-1]
    print(file_name + " Converting...")
    image_path = os.path.dirname(file_path)
    nowDirCNN = os.path.join(original_path, "nowfigCNN")
    os.makedirs(nowDirCNN, exist_ok=True)
    try:
        I = Image.open(file_path)
    except:
        I.close()
        shutil.move(file_path, image_path + '/' + 'error_read')
        return
    try:
        newI = transformationForCNNInput(I)  # CNN에 맞게 inputDim 크기로 변환
    except:
        I.close()
        shutil.move(file_path, image_path + '/' + 'error_read')
        return
    try:
        # copy the rotation information metadata from original image and save, else your transformed images may be rotated
        exif = I.info['exif']
        newI.save(os.path.join(nowDirCNN, file_name), exif=exif)
    except:
        newI.save(os.path.join(nowDirCNN, file_name))
    newI.close()
    I.close()

class Img2VecResnet18:
    def __init__(self):
        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()

        # normalize the resized images as expected by resnet18
        # [0.485, 0.456, 0.406] --> normalized mean value of ImageNet, [0.229, 0.224, 0.225] std of ImageNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def getVec(self, img):
        a = self.toTensor(img)
        b = self.normalize(a)
        c = b.unsqueeze(0)
        d = c.to(self.device)
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)
        def copyData(m, i, o): embedding.copy_(o.data)
        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()
        return embedding.numpy()[0, :, 0, 0]
    def getFeatureLayer(self):
        cnnModel = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512
        return cnnModel, layer

img2vec = Img2VecResnet18()

def SaveVectorPkl(file_path: str):
    file_name = file_path.split('/')[-1]
    image_path = os.path.dirname(file_path)
    print(file_name)
    nowVectors = {}
    I = Image.open(os.path.join(original_path, "nowfigCNN", file_name))
    vec = img2vec.getVec(I)
    nowVectors[file_name] = vec
    I.close()
    path_nowVector = original_path + '/nowVector.pkl'
    nowV = open(path_nowVector, 'wb')
    pickle.dump(nowVectors, nowV)
    nowV.close()
    return path_nowVector

def SaveAllVectorPkl():
    file = open('dirname.txt', 'r')
    dirname = file.read()
    inputfigCNN = os.path.join(dirname, "inputfigCNN")
    file_list = os.listdir(inputfigCNN)
    path_allVectors = os.path.join(dirname, "vectors/allVectors.pkl")
    allVectors = {}
    print("Converting images to feature vectors:")
    for ii in tqdm(range(len(file_list)), desc="Convert Vector"):
        try:
            I = Image.open(os.path.join(inputfigCNN, file_list[ii]))
            vec = img2vec.getVec(I)
            allVectors[file_list[ii]] = vec
            I.close()
        except Exception as e:
            print(e)
    print("Saving allVectors into allVectors.pkl:")
    allV = open(path_allVectors, 'wb')
    pickle.dump(allVectors, allV)
    allV.close()

def getSimilarityMatrix(vectorList, vertor):
    ## vectors= allVectors
    v1 = np.array(list(vectorList.values())).T
    v2 = np.array(list(vertor.values())).T
    sim = np.inner(v1.T, v2.T) / (
            (np.linalg.norm(v1, axis=0).reshape(-1, 1)) * (np.linalg.norm(v2, axis=0).reshape(-1, 1)).T)
    keys = list(vectorList.keys())
    key = list(vertor.keys())
    matrix = pd.DataFrame(sim, columns=key, index=keys)
    return matrix

def getSimilarityImageList(path_nowVector, path_allVectors):
    nowV = open(path_nowVector, "rb")
    nowVector = pickle.load(nowV)
    allV = open(path_allVectors, "rb")
    allVectors = pickle.load(allV)
    similarityMatrix = getSimilarityMatrix(allVectors, nowVector)
    kSimilar = similarityMatrix.iloc[:, 0].sort_values(ascending=False).head(5)
    result = []
    for ii in range(len(kSimilar.index)):
        result.append(kSimilar.index[ii])
    return result

# 프로그램 메인을 담당하는 Class 선언
class MainClass(QMainWindow, form_class):
    def __init__(self):
        QMainWindow.__init__(self)
        # 연결한 Ui를 준비한다.
        self.setupUi(self)
        self.setAcceptDrops(True)
        self.file_open_button.clicked.connect(self.fileopen)
        # 화면을 보여준다.
        self.show()

    def fileopen(self):         # 디렉토리 선택을 위한 파일다이얼로그 띄우기
        dirname = QFileDialog.getExistingDirectory(self)
        if dirname:
            print(dirname)
            file = open('dirname.txt','w')
            file.write(dirname)
            file.close()
            self.textEdit_content.setText(dirname)
            ImageConvertAll(dirname)
            SaveAllVectorPkl()
        else:
            return

    def dragEnterEvent(self, event):    # 드래그 진입할때 호출되는 함수
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):     # 드래그하고 이동할때 호출되는 함수
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event, original_path=None):                 # 드래그를 드랍했을때 호출되는 함수
        files = [u.toLocalFile() for u in event.mimeData().urls()]  # 드랍한 파일들의 이름을 저장하는 변수
        for f in files:
            file_name = f.split('/')[-1]        # 파일 이름 따로 변수로 선언
            file = open('imgname.txt', 'w')
            file.write(os.path.splitext(file_name)[0])
            self.MainTitle.setText(file_name)   # 메인 파일 제목을 드랍한 파일 이름으로 바꾸기
            pixmap = QPixmap()                  # GUI에 띄울 300*300 이미지 생성을 위한 라이브러리 불러오기
            pixmap.load(f)                      # 그 라이브러리에 드랍한 파일 올려놓기
            pixsize = pixmap.size()

            if pixmap.height()>pixmap.width():  # 너비나 높이중 높은 값을 가진 부분을 300으로 fix하여 원본 파일 비율대로 축소하는 함수
                pixmap = pixmap.scaledToHeight(300)
            else:
                pixmap = pixmap.scaledToWidth(300)
            self.MainView.setPixmap(pixmap)     # 300*300 보다 작거나 같아진 이미지 파일을 메인 사진으로 띄움

            ImageConvert(f)
            path_nowVector = SaveVectorPkl(f)
            file = open('dirname.txt', 'r')
            dirname = file.read()
            path_allVectors = os.path.join(dirname, "vectors/allVectors.pkl")
            ImageList = getSimilarityImageList(path_nowVector,path_allVectors)
            print(ImageList)
            Image_path = dirname

            self.SubTitle.setText(ImageList[0])    # 유사한 이미지의 이름을 서브 제목들로 설정함
            self.SubTitle2.setText(ImageList[1])
            self.SubTitle3.setText(ImageList[2])
            self.SubTitle4.setText(ImageList[3])
            self.SubTitle5.setText(ImageList[4])

            path = Image_path + "/" + ImageList[0]   # 유사한 이미지들을 300*300 보다 작거나 같게 수정하는 과정
            if os.path.isfile(path):
                pixmap.load(path)
                if pixmap.height() > pixmap.width():
                    pixmap = pixmap.scaledToHeight(300)
                else:
                    pixmap = pixmap.scaledToWidth(300)
                self.SubView.setPixmap(pixmap)

            path = Image_path + "/" + ImageList[1]
            if os.path.isfile(path):
                pixmap.load(path)
                if pixmap.height() > pixmap.width():
                    pixmap = pixmap.scaledToHeight(300)
                else:
                    pixmap = pixmap.scaledToWidth(300)
                self.SubView2.setPixmap(pixmap)

            path = Image_path + "/" + ImageList[2]
            if os.path.isfile(path):
                pixmap.load(path)
                if pixmap.height() > pixmap.width():
                    pixmap = pixmap.scaledToHeight(300)
                else:
                    pixmap = pixmap.scaledToWidth(300)
                self.SubView3.setPixmap(pixmap)

            path = Image_path + "/" + ImageList[3]
            if os.path.isfile(path):
                pixmap.load(path)
                if pixmap.height() > pixmap.width():
                    pixmap = pixmap.scaledToHeight(300)
                else:
                    pixmap = pixmap.scaledToWidth(300)
                self.SubView4.setPixmap(pixmap)

            path = Image_path + "/" + ImageList[4]
            if os.path.isfile(path):
                pixmap.load(path)
                if pixmap.height() > pixmap.width():
                    pixmap = pixmap.scaledToHeight(300)
                else:
                    pixmap = pixmap.scaledToWidth(300)
                self.SubView5.setPixmap(pixmap)

    def closeEvent(self, QCloseEvent):
        ans = QMessageBox.question(self, "종료 확인", "종료하시겠습니까?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans == QMessageBox.Yes:
            QCloseEvent.accept()
            file = open('dirname.txt', 'r')
            dirname = file.read()
            shutil.rmtree(dirname + '/inputfigCNN')
            shutil.rmtree(dirname + '/vectors')
            path1 = os.path.join(os.getcwd(),'dirname.txt')
            path2 = os.path.join(os.getcwd(), 'imgname.txt')
            os.remove(path1)
            os.remove(path2)

        else:
            QCloseEvent.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)    # pyqt 어플리케이션 실행하는 함수들
    window = MainClass()
    app.exec_()