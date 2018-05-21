#@Time     :2018/5/2 15:58
import numpy as np
from networks import ProcessData
import os
import SimpleITK as sitk
import random
class mark():
    def __init__(self, npImage):
        self.npImage = npImage
        return

    def GetLocation(self):
        npImage = np.zeros([256,256])
        loc = []
        shape = npImage.shape
        for i in range(0, shape[0]):
            for j in range(0,shape[1]):
                # how to get hte location of the marks
                if self.npImage[i,j]==1:
                    loc.append([i,j])
                    self.EraseOtherPixel(i,j)
        return loc
    def EraseOtherPixel(self, x,y):
        if self.npImage[x,y] !=1:
            return
        else:
            self.npImage[x,y] = 0
        if self.npImage[x,y-1] ==1:
            self.EraseOtherPixel(x,y-1)
        if self.npImage[x,y+1] ==1:
            self.EraseOtherPixel(x,y+1)
        if self.npImage[x+1,y] ==1:
            self.EraseOtherPixel(x+1,y)
        if self.npImage[x-1,y] == 1:
            self.EraseOtherPixel(x-1,y)
        return



    pass

def GetAllXY():
    npImages = []
    for index in range(0, 10000):
        sitkImage = sitk.ReadImage(os.path.join(os.path.abspath("."), "RawDataFullConnect","RawData","20-30", "NrrdData", "%d.nrrd" % index))
        npImage = sitk.GetArrayFromImage(sitkImage)
        npImage = npImage.astype(float)
        npImage = npImage.transpose((2, 1, 0))
        npImage = npImage[:, :, 0]
        npImages.append(npImage)
    for index in range(0, 10000):
        sitkImage = sitk.ReadImage(
            os.path.join(os.path.abspath("."), "RawDataFullConnect", "RawData", "_-30_10", "NrrdData", "%d.nrrd" % index))
        npImage = sitk.GetArrayFromImage(sitkImage)
        npImage = npImage.astype(float)
        npImage = npImage.transpose((2, 1, 0))
        npImage = npImage[:, :, 0]
        npImages.append(npImage)

    ImageLocs = []
    for i in range(0, len(npImages)):
        mymark = mark(npImages[i])
        locs = mymark.GetLocation()
        ImageLocs.append(locs)
    dataY = np.load(os.path.join(os.path.abspath("."),  "RawDataFullConnect","RawData","20-30", "transformArray", "transform.npy"))
    dataY2 = np.load(
        os.path.join(os.path.abspath("."), "RawDataFullConnect", "RawData", "_-30_10", "transformArray", "transform.npy"))
    dataY = np.append(dataY,dataY2, axis = 0)
    np.save(os.path.join(os.path.abspath("."), "RawDataFullConnect","AllData", "Locs.npy"), ImageLocs)
    np.save(os.path.join(os.path.abspath("."), "RawDataFullConnect","AllData", "trasform.npy"), dataY)
    return ImageLocs, dataY

def GetBatchAndSave():
    batchSize = 100
    DataX = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "AllData", "Locs.npy"))
    DataY = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "AllData", "trasform.npy"))
    zipped = list(zip(DataX, DataY))
    zippedout = [x for x in zipped if len(x[0]) == 7]
    random.shuffle(zippedout)
    dataX, dataY = zip(*zippedout)
    #use the order of dataX
    trainX = dataX[0:int(len(dataX) * 0.8)]
    trainY = dataY[0:int(len(dataY) * 0.8)]
    meanTrainX = np.mean(dataX,axis = 0)
    meanTrainY = np.mean(trainY, axis=0)
    stdTrainY = np.std(trainY, axis=0)
    trainY = (trainY - meanTrainY) / stdTrainY
    validX = dataX[int(0.8 * len(dataX)):int(0.9 * len(dataX))]
    validY = dataY[int(0.8 * len(dataY)):int(0.9 * len(dataY))]
    validY = (validY - meanTrainY) / stdTrainY
    testX = dataX[int(0.9 * len(dataX)):len(dataX) - 1]
    testY = dataY[int(0.9 * len(dataY)):len(dataY) - 1]
    testY = (testY - meanTrainY) / stdTrainY
    trainXBatches = []
    trainYBatches = []

    NumBatch = len(trainX) // batchSize
    for i in range(0, NumBatch - 1):
        trainXBatches.append(trainX[i * batchSize:(i + 1) * batchSize])
        trainYBatches.append(trainY[i * batchSize:(i + 1) * batchSize])
    trainXBatches.append(trainX[(NumBatch - 1) * batchSize:len(trainX)])
    trainYBatches.append(trainY[(NumBatch - 1) * batchSize:len(trainY)])
    np.save(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "trainX.npy"), trainXBatches)
    np.save(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "trainY.npy"), trainYBatches)
    np.save(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "validX.npy"), validX)
    np.save(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "validY.npy"), validY)
    np.save(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "testX.npy"), testX)
    np.save(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "testY.npy"), testY)
    ##save the mean and std of the code
    np.save(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "meanTrainY.npy"),
            meanTrainY)
    np.save(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "stdTrainY.npy"), stdTrainY)

def ReadBatchData():
    trainx = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "trainX.npy"))
    trainy = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "trainY.npy"))
    testx = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "testX.npy"))
    testy = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "testY.npy"))
    validx = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "validX.npy"))
    validy = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "validY.npy"))
    mean = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "meanTrainY.npy"))
    std = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "stdTrainY.npy"))
    tempy = validy * std + mean
    return trainx, trainy, testx, testy, validx, validy, mean, std

def GetTestDataAndSave():
    dataX = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "testData_15_25","AllData", "Locs.npy"))
    dataY = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "testData_15_25","AllData", "trasform.npy"))
    mean = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "meanTrainY.npy"))
    std = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "stdTrainY.npy"))
    zipped = list(zip(dataX, dataY))
    zippedout = [x for x in zipped if len(x[0]) == 7]
    random.shuffle(zippedout)
    dataX, dataY = zip(*zippedout)
    dataX = np.divide(dataX,255)
    dataY =(dataY-mean)/std
    np.save(os.path.join(os.path.abspath("."),  "RawDataFullConnect", "testData_15_25","testXY_rotationZ_20_25", "validX.npy"), dataX)
    np.save(os.path.join(os.path.abspath("."), "RawDataFullConnect", "testData_15_25","testXY_rotationZ_20_25", "validY.npy"), dataY)
    return

def ReadTestData():
    dataX =  np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "testData_15_25","testXY_rotationZ_20_25", "validX.npy"))
    dataY = np.load(os.path.join(os.path.abspath("."), "RawDataFullConnect", "testData_15_25", "testXY_rotationZ_20_25", "validY.npy"))
    mean = np.load(
        os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "meanTrainY.npy"))
    std = np.load(
        os.path.join(os.path.abspath("."), "RawDataFullConnect", "trainBatchData100000_normX_0507", "stdTrainY.npy"))
    return dataX, dataY,mean, std

# def main():
#
#      dataX,dataY = GetAllXY()
#      GetBatchAndSave()
#      return
#
# main()

#GetTestDataAndSave()