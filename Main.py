#@Time     :2018/4/20 13:37

import tensorflow as tf
import config
from networks import CNN, ProcessData
import logging
logging.basicConfig(level=logging.DEBUG)


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def main(_):
  #IsTraining = False
  gpuConfig = "1/1"
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(gpuConfig))

  with tf.Session() as sess:
    baseConfig = config.BaseConfig()


    if not baseConfig.GpuUse:
      #raise Exception("use_gpu flag is true when no GPUs are available")
      print("use_gpu flag is true when no GPUs are available")

    trainxBatch, trainyBatch, testx, testy ,validx, validy ,mean, std= ProcessData.ReadBatchesDataWithMeanStd()
    if baseConfig.DataFormat != "NHWC":        #the data form is NCHW
      logging.debug("the shape of train is %s"%str(trainxBatch.shape))

      trainxBatch = trainxBatch.transpose((0,1,4,2,3))
      testx = testx.transpose((0,3,1,2))
      validx = validx.transpose((0,3,1,2))
    cnn = CNN.CNN(sess,baseConfig,mean, std,len(trainxBatch)*baseConfig.Batch)
    if baseConfig.IsTraining:

      cnn.Train(trainxBatch,trainyBatch,validx,validy, mean, std)

    else:
      cnn.Play()

def calc_gpu_fraction(fraction_string):  # 这个是用来进行gpu分析的
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

if __name__ == '__main__':
  tf.app.run()
