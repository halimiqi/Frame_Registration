import numpy as np
import pandas as pd
import math
import random
import os
import matplotlib.pyplot as plt

def SortError(errorX, errorY):
    zipped = list(zip(errorX, errorY))
    zipped.sort(key=lambda x: x[0])
    #zipped = list(filter(lambda x: x[0]<30, zipped))
    return zipped

def PlotError(standard,error ):
    leftx  = -15*np.ones([20])
    lefty = np.linspace(0,12,20)
    rightx = 15* np.ones([20])
    righty= np.linspace(0,12,20)
    plt.figure()
    plt.title(r'$The\ Error\ Of\ Rotation\ On\ Z\ Axis$', fontsize='large',fontweight = 'bold')
    ax = plt.gca()
    ax.spines['bottom'].set_position(('data', 0))

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel("the transformation of the frame  (mm)")
    plt.ylabel("the diviation of the prediction  (mm)")
    plt.xlim((-40,40))
    plt.ylim((0, 15))
    new_ticks = np.linspace(-30,30,25)
    plt.xticks(new_ticks)
    new_ticks = np.linspace(0,15,31)
    plt.yticks(new_ticks)
    plt.text(0, 10, r'$Training\ Range$',
            fontdict={'size': 13},verticalalignment = "center", horizontalalignment = 'center')
    plt.annotate('', xy=(-15, 10), xytext=(-7.5, 10), arrowprops=dict(facecolor='black', shrink=0.1,width = 0.5,headwidth = 5.0))
    plt.annotate('', xy=(15, 10), xytext=(7.5, 10),
                 arrowprops=dict(facecolor='black', shrink=0.1, width=0.5, headwidth=5.0))

    plt.plot(leftx,lefty, color='red', linewidth=1.0, linestyle='--')
    plt.plot(rightx, righty,color='red', linewidth=1.0, linestyle='--')
    #plt.plot(standard, error)
    plt.scatter(standard,error,s = 4.0,marker = "x")
    plt.show()

def Main():
    pred = np.load(os.path.join(os.path.abspath(".."),"play","results_2018_05_21_11_25_07FULL_102420482048drop2_rotateY","pred.npy"))
    standard = np.load(os.path.join(os.path.abspath(".."),"play","results_2018_05_21_11_25_07FULL_102420482048drop2_rotateY","standard.npy"))
    error = np.abs(pred - standard)
    errorRotateX = error[:,4]
    zippedError = SortError(standard[:,4], errorRotateX)
    Standard, Error = zip(*zippedError)
    print(np.mean(Error,axis = 0))
    print(np.std(Error,axis = 0))
    PlotError(Standard,Error)

Main()