#@Time     :2018/5/9 11:16
#@Time     :2018/5/3 0:39
import tensorflow as tf
import config_full
from networks import FullConnect_102420482048_drop2
from SummarizeAndPlot import GetMarkLocation_summary
import functools as fts
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def main(_):
  #IsTraining = False
  gpuConfig = "1/1"
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(gpuConfig))

  with tf.Session() as sess:
    baseConfig = config_full.BaseConfig()


    if not baseConfig.GpuUse:
      #raise Exception("use_gpu flag is true when no GPUs are available")
      print("use_gpu flag is true when no GPUs are available")

    # trainyBatch, testx, testy ,validx, validy ,mean, std= GetMarkLocation.ReadBatchData()
    dataX, dataY, mean, std = GetMarkLocation_summary.ReadTestData()
    net = FullConnect_102420482048_drop2.FullConnect(sess,baseConfig, mean ,std,dataX.shape[0])

    net.Play(dataX,dataY,mean,std)

def calc_gpu_fraction(fraction_string):  # 这个是用来进行gpu分析的
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

if __name__ == '__main__':
  tf.app.run()
