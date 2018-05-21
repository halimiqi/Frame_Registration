#@Time     :2018/4/23 10:38
import os
import time
class BaseConfig():
    def __init__(self):

        self.IsTraining = False

        self.Height = 256
        self.Width = 256
        self.Channel = 1
        self.Batch = 100
        self.DataFormat = "NHWC"
        #self.LearningRate = 0.001
        self.LearningRateMinimum = 0.00001
        self.LearningRateMaximum = 0.00008
        self.dropoutKeepProb = 0.5
        # the performance of 0.0006 is not good enough

        #0.005 到0.006 is a rational range
        self.LearningRateDecayStep = 1000
        self.LearningRateDecayRate = 0.99
        self.MaxTrainStep = 3000000
        self.TestStep = 2000
        self.SaveStep = 10000
        self.TestNum = 1  # 这里是
        self.GpuUse = True
        if self.GpuUse == False:
            self.DataFormat = "NHWC"
        else:
            self.DataFormat = "NCHW"


        self.ModelName = "FULL_102420482048drop2_Test"
        currentPath = os.path.dirname(os.path.realpath(__file__))
        self.TFBoardPathTrain = os.path.join(currentPath,"tfboard",time.strftime("%Y_%m_%d_%H_%M_%S",
                                                                            time.localtime(time.time())) + self.ModelName, "Train")
        self.TFBoardPathValid = os.path.join(currentPath,"tfboard",time.strftime("%Y_%m_%d_%H_%M_%S",
                                                                            time.localtime(time.time())) + self.ModelName, "Valid")
        self.ModelDir = os.path.join(currentPath,"save_model")
        self.CheckpointDir = os.path.join(self.ModelDir,"checkpoints", "checkpoints_"+
                                          time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))+ self.ModelName)
        self.LoadModelDir = os.path.join(self.ModelDir,"checkpoints","checkpoints_20180520_193516FULL_102420482048drop2_rotateX")

        self.PlayResultDir = os.path.join(currentPath, "play", "results_"+time.strftime("%Y_%m_%d_%H_%M_%S",
                                                                            time.localtime(time.time())) + self.ModelName)
        return
    pass
