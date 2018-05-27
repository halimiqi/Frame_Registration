#@Time     :2018/4/23 10:38
import os
import time
class BaseConfig():
    def __init__(self):

        self.IsTraining = False  # TURE for training the algorithm and FLASE for run a trained algorithm
        self.LoadFileName = "checkpoints_20180520_193516FULL_102420482048drop2_rotateX"  #this is the file name of the trained network parameters
        self.Height = 256  # the height of the input image
        self.Width = 256   # the width of the input image
        self.Channel = 1   # the channel of the input image, Channel = 1 is grayscale image, Channel = 3 is RGB image
        self.Batch = 100   # the batch size of the network
        self.DataFormat = "NHWC" # the dataformat of the input data, NHWC is [number_of_image, height,width, channel], NCWH is [number_of_image, channel, width, height]
        #self.LearningRate = 0.001
        self.LearningRateMinimum = 0.00001 # the minimum learning rate of the optimizer
        self.LearningRateMaximum = 0.00008  # the maximum learning rate of the optimizer
        self.dropoutKeepProb = 0.5  # the keep probability of dropout layer

        #0.005 到0.006 is a rational range
        self.LearningRateDecayStep = 1000 # how many step to decrease the learning rate
        self.LearningRateDecayRate = 0.99 # the keep probability of every time decrease of the learning rate
        self.MaxTrainStep = 3000000  # maximum training step
        self.TestStep = 2000  # how many steps to print the performance of the network and save it to tensorboard
        self.SaveStep = 10000  # hwo many steps to save the parameters
        self.TestNum = 1    # if TestNum >1 that means the test set is too big to run in the algorithm and the test set will be divided into several parts to test the network.
        self.GpuUse = True # TRUE is using the GPU and CPU, FALSE is only using CPU
        if self.GpuUse == False:
            self.DataFormat = "NHWC"
        else:
            self.DataFormat = "NCHW"


        self.ModelName = "FULL_102420482048drop2_Test"  # the name of this network, it will shows as the folder name at save_model/checkpoints/ModelName
        currentPath = os.path.dirname(os.path.realpath(__file__))
        self.TFBoardPathTrain = os.path.join(currentPath,"tfboard",time.strftime("%Y_%m_%d_%H_%M_%S",
                                                                            time.localtime(time.time())) + self.ModelName, "Train")  
        self.TFBoardPathValid = os.path.join(currentPath,"tfboard",time.strftime("%Y_%m_%d_%H_%M_%S",
                                                                            time.localtime(time.time())) + self.ModelName, "Valid")
        self.ModelDir = os.path.join(currentPath,"save_model")
        self.CheckpointDir = os.path.join(self.ModelDir,"checkpoints", "checkpoints_"+
                                          time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))+ self.ModelName)
        self.LoadModelDir = os.path.join(self.ModelDir,"checkpoints",self.loadFileName)

        self.PlayResultDir = os.path.join(currentPath, "play", "results_"+time.strftime("%Y_%m_%d_%H_%M_%S",
                                                                            time.localtime(time.time())) + self.ModelName)
        return
    pass
