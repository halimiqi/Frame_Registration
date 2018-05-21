#@Time     :2018/4/24 16:43
import pandas as df
import numpy as np
from tqdm import tqdm
import os
import SimpleITK as sitk
import random

def ReadNrrdToNdarray():
    dataX = []
    for index in range(0,10000):
        sitkImage = sitk.ReadImage(os.path.join(os.path.abspath(".."),"RawDataFullConnect", "RawData","20-30","NrrdData","%d.nrrd"%index))
        npImage = sitk.GetArrayFromImage(sitkImage)
        npImage=npImage.astype(float)
        npImage = npImage.transpose((2,1,0))
        Index = np.where(npImage==1)
        for j in range(len(Index[0])):
            x =Index[0][j]
            y = Index[1][j]

            for i in range(-3,4):
                for k in range(-3,4):
                    if npImage[x+i,y+k,0] ==0.:
                        npImage[x+i,y+k,0] = 0.7
        dataX.append(npImage)
    for index in range(0,10000):
        sitkImage = sitk.ReadImage(os.path.join(os.path.abspath(".."),"RawDataFullConnect","RawData","_-30_10", "NrrdData","%d.nrrd"%index))
        npImage = sitk.GetArrayFromImage(sitkImage)
        npImage=npImage.astype(float)
        npImage = npImage.transpose((2,1,0))
        Index = np.where(npImage==1)
        for j in range(len(Index[0])):
            x =Index[0][j]
            y = Index[1][j]

            for i in range(-3,4):
                for k in range(-3,4):
                    if npImage[x+i,y+k,0] ==0.:
                        npImage[x+i,y+k,0] = 0.7
        dataX.append(npImage)


    dataY = np.load(os.path.join(os.path.abspath(".."),"RawDataFullConnect", "RawData","20-30","transformArray", "transform.npy"))
    dataY2 = np.load(os.path.join(os.path.abspath(".."),"RawDataFullConnect","RawData","_-30_10","transformArray", "transform.npy"))
    dataY = np.append(dataY, dataY2, axis = 0)
    #np.save(os.path.join(os.path.abspath(".."), "RawData2","ImageArray", "Image.npy"), dataX)
    return dataX , dataY
def GetDataXY():
    dataX = np.load(os.path.join(os.path.abspath(".."),"RawData","ImageArray", "Image.npy"))
    dataY = np.load(os.path.join(os.path.abspath(".."), "RawData", "transformArray", "transform.npy"))
    return dataX, dataY
def GetDataSet(dataX,dataY):
    zipped = list(zip(dataX,dataY))
    random.shuffle(zipped)
    dataX,dataY = zip(*zipped)
    trainX = dataX[0:int(len(dataX)*0.8)]
    trainY = dataY[0:int(len(dataY)*0.8)]
    meanTrainY = np.mean(trainY,axis = 0)
    stdTrainY = np.std(trainY,axis = 0)
    trainY = (trainY-meanTrainY)/stdTrainY

    validX = dataX[int(0.8*len(dataX)):int(0.9*len(dataX))]
    validY = dataY[int(0.8*len(dataY)):int(0.9*len(dataY))]
    validY = (validY-meanTrainY)/stdTrainY
    testX = dataX[int(0.9*len(dataX)):len(dataX)-1]
    testY = dataY[int(0.9*len(dataY)):len(dataY)-1]
    testY = (testY-meanTrainY)/stdTrainY

    np.save(os.path.join(os.path.abspath(".."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "validX.npy"), validX)
    np.save(os.path.join(os.path.abspath(".."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "validY.npy"), validY)
    np.save(os.path.join(os.path.abspath(".."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "testX.npy"), testX)
    np.save(os.path.join(os.path.abspath(".."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "testY.npy"), testY)
    ##save the mean and std of the code
    np.save(os.path.join(os.path.abspath(".."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "meanTrainY.npy"), meanTrainY)
    np.save(os.path.join(os.path.abspath(".."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "stdTrainY.npy"), stdTrainY)
    return trainX,trainY,validX,validY, testX, testY, meanTrainY, stdTrainY

def GetBatch(trainX,trainY,batchSize):
    trainXBatches =[]
    trainYBatches = []
    NumBatch = len(trainX)//batchSize
    for i in range(0,NumBatch-1):
        trainXBatches.append(trainX[i*batchSize:(i+1)*batchSize])
        trainYBatches.append(trainY[i*batchSize:(i+1)*batchSize])
    trainXBatches.append(trainX[(NumBatch-1)*batchSize:len(trainX)])
    trainYBatches.append(trainY[(NumBatch-1)*batchSize:len(trainY)])
    np.save(os.path.join(os.path.abspath(".."), "RawDataFullConnect","trainBatchData_20000_-30_10__20_30", "trainX.npy"), trainXBatches)
    np.save(os.path.join(os.path.abspath(".."), "RawDataFullConnect", "trainBatchData_20000_-30_10__20_30", "trainY.npy"), trainYBatches)
    return trainXBatches,trainYBatches

def ReadBatchesData():
    trainx = np.load(os.path.join(os.path.abspath("."), "RawData2","trainBatchData", "trainX.npy"))
    trainy = np.load(os.path.join(os.path.abspath("."), "RawData2","trainBatchData", "trainY.npy"))
    testx = np.load(os.path.join(os.path.abspath("."), "RawData2","trainBatchData", "testX.npy"))
    testy = np.load(os.path.join(os.path.abspath("."), "RawData2", "trainBatchData", "testY.npy"))
    validx = np.load(os.path.join(os.path.abspath("."), "RawData2", "trainBatchData", "validX.npy"))
    validy = np.load(os.path.join(os.path.abspath("."), "RawData2", "trainBatchData", "validY.npy"))
    return trainx, trainy, testx, testy, validx, validy

def ReadBatchesDataWithMeanStd():
    trainx = np.load(os.path.join(os.path.abspath("."), "RawData3","trainBatchData", "trainX.npy"))
    trainy = np.load(os.path.join(os.path.abspath("."), "RawData3","trainBatchData", "trainY.npy"))
    testx = np.load(os.path.join(os.path.abspath("."), "RawData3","trainBatchData", "testX.npy"))
    testy = np.load(os.path.join(os.path.abspath("."), "RawData3", "trainBatchData", "testY.npy"))
    validx = np.load(os.path.join(os.path.abspath("."), "RawData3", "trainBatchData", "validX.npy"))
    validy = np.load(os.path.join(os.path.abspath("."), "RawData3", "trainBatchData", "validY.npy"))
    mean = np.load(os.path.join(os.path.abspath("."), "RawData3", "trainBatchData", "meanTrainY.npy"))
    std = np.load(os.path.join(os.path.abspath("."), "RawData3", "trainBatchData", "stdTrainY.npy"))
    return trainx, trainy, testx, testy, validx, validy, mean, std

# dataX, dataY = ReadNrrdToNdarray()
# trainx, trainy ,validx, validy , tesstx, testy, mean, std = GetDataSet(dataX, dataY)
# trainXbatch, trainYbatch = GetBatch(trainx, trainy,50)

# datax, datay = ReadNrrdToNdarray()
# trainX, trainY, validX, validY, testX, testY = GetDataSet(datax , datay)
# trainxbatch, trainybatch = GetBatch(trainX, trainY)
