# encoding=utf-8
import numpy as np
import cv2
import os


class EigenFace(object):
    def __init__(self, threshold, dimNum, dsize):
        self.threshold = threshold
        self.dimNum = dimNum
        self.dsize = dsize

    def loadImg(self, fileName, dsize):
        img = cv2.imread(fileName)
        retImg = cv2.resize(img, dsize)
        retImg = cv2.cvtColor(retImg, cv2.COLOR_RGB2GRAY)
        retImg = cv2.equalizeHist(retImg)
        return retImg

    def createImgMat(self, dirName):
        dataMat = np.zeros((10, 1))
        label = []
        for parent, dirnames, filenames in os.walk(dirName):
            index = 0
            for filename in filenames:
                img = self.loadImg(parent + '/' + filename, self.dsize)
                tempImg = np.reshape(img, (-1, 1))
                if index == 0:
                    dataMat = tempImg
                else:
                    dataMat = np.column_stack((dataMat, tempImg))
                label.append(filename.split('.')[0])
                index += 1
        return dataMat, label
    def PCA(self, dataMat, dimNum):

        meanMat = np.mat(np.mean(dataMat, 1)).T
        diffMat = dataMat - meanMat
        covMat = (diffMat.T * diffMat) / float(diffMat.shape[1])  # 归一化
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))
        eigVects = diffMat * eigVects
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[::-1]
        eigValInd = eigValInd[:dimNum]  # 取出指定个数的前n大的特征值
        eigVects = eigVects / np.linalg.norm(eigVects, axis=0)  # 归一化特征向量
        redEigVects = eigVects[:, eigValInd]
        lowMat = redEigVects.T * diffMat
        return lowMat, redEigVects

    def compare(self, dataMat, testImg, label):
        testImg = cv2.resize(testImg, self.dsize)
        testImg = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)
        testImg = np.reshape(testImg, (-1, 1))
        lowMat, redVects = self.PCA(dataMat, self.dimNum)
        testImg = redVects.T * testImg
        disList = []
        testVec = np.reshape(testImg, (1, -1))
        for sample in lowMat.T:
            disList.append(np.linalg.norm(testVec - sample))
        sortIndex = np.argsort(disList)
        return label[sortIndex[0]]

    def predict(self, dirName, testFileName):
        testImg = cv2.imread(testFileName)
        dataMat, label = self.createImgMat(dirName)
        ans = self.compare(dataMat, testImg, label)
        return ans


if __name__ == '__main__':
    eigenface = EigenFace(20, 50, (64, 64))
    fh = open('181250004.txt', 'w', encoding='utf-8')
    count=0
    for parent, dirnames, filenames in os.walk('test'):
        for filename in filenames:
            count=count+1
            b=eigenface.predict('gallery','test/'+filename)
            fh.write(filename+" "+b+"\n")
            print("第%d张图片"%count)
            print(b)
    fh.close()
