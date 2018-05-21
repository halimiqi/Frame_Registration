#@Time     :2018/5/3 0:39
import tensorflow as tf
import config_full
from networks import FullConnect_102420482048_drop2
import GetMarkLocation
import functools as fts
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def main(_):
  #IsTraining = False
  gpuConfig = "0.9 /1"
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(gpuConfig))

  with tf.Session() as sess:
    baseConfig = config_full.BaseConfig()


    if not baseConfig.GpuUse:
      #raise Exception("use_gpu flag is true when no GPUs are available")
      print("use_gpu flag is true when no GPUs are available")

    trainxBatch, trainyBatch, testx, testy ,validx, validy ,mean, std= GetMarkLocation.ReadBatchData()
    net = FullConnect_102420482048_drop2.FullConnect(sess,baseConfig, mean ,std,len(trainxBatch)* baseConfig.Batch)
    if baseConfig.IsTraining:
      net.Train(trainxBatch,trainyBatch,validx,validy, mean, std)
    else:
      net.Play(testx,testy,mean,std)

def calc_gpu_fraction(fraction_string):  # 这个是用来进行gpu分析的
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

if __name__ == '__main__':
  tf.app.run()

