#@Time     :2018/5/9 11:45

import tensorflow as tf
import numpy as np
import os
from networks.operations import linear,IfOverfitting
from tqdm import tqdm
import time
import os
import logging
logging.basicConfig(level=logging.INFO)
import random

class  FullConnect(object):
    def __init__(self,sess,baseconfig, mean, std, trainSetNum):
        self.Config = baseconfig
        self.Sess = sess
        self.mean = mean
        self.std = std
        self.TrainSetNum = trainSetNum
        self.BuildFullConnect()
        return

    def BuildFullConnect(self):
        self.w = {}
        self.b = {}
        #initializer = tf.truncated_normal_initializerA(0, 0.2)
        initializer = tf.contrib.layers.xavier_initializer()
        activation_fn = tf.nn.leaky_relu
        print("!!!!!!BUILD START!!!!!!!!")
        self.step_op = tf.Variable(0, trainable=False, name='step')
        self.step_input = tf.placeholder('int32', None, name='step_input')
        self.step_assign_op = self.step_op.assign(self.step_input)
        with tf.variable_scope("network"):
            self.X = tf.placeholder("float32", [None,7,2], name="Input")
            self.X_flat = tf.reshape(self.X, [-1,14])
            self.Y = tf.placeholder("float32", [None, 6], name="Input")
            self.keepProb = tf.placeholder("float32", None, name = "keepProb")
            self.linear1, self.w["w1_1"], self.b["b1_1"] = linear(self.X_flat, 128, activation_fn=activation_fn,
                                                                  name='linear1')
            self.linear2, self.w["w2_1"], self.b["b2_1"] = linear(self.linear1, 256, activation_fn=activation_fn,
                                                                  name='linear2')
            self.linear3, self.w["w3_1"], self.b["b3_1"] = linear(self.linear2, 512, activation_fn=activation_fn,
                                                                  name='linear3')

            self.linear4, self.w["w4_1"], self.b["b4_1"] = linear(self.linear3, 1024, activation_fn=activation_fn,
                                                                  name='linear4')
            self.drop4 = tf.nn.dropout(self.linear4, self.keepProb)
            self.linear5, self.w["w5_1"], self.b["b5_1"] = linear(self.linear4, 2048, activation_fn=activation_fn,
                                                                  name='linear5')
            self.drop5 = tf.nn.dropout(self.linear5, self.keepProb)
            self.linear6, self.w["w6_1"], self.b["b6_1"] = linear(self.drop5, 2048, activation_fn=activation_fn,
                                                                  name='linear6')
            self.linear7, self.w["w7_1"], self.b["b7_1"] = linear(self.linear6, 6, activation_fn=activation_fn, name = 'linear7')
            ####change the loss
            self.pred  = tf.multiply(self.linear7,self.std) + self.mean
            self.y_out = tf.multiply(self.Y, self.std) + self.mean
            self.delta = tf.subtract(self.pred, self.y_out)
            self.small_delta = tf.subtract(self.linear7,self.Y)
            ##L2 normalization
            for (k, v) in self.w.items():
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, v)
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0000001)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)
            #get loss
            self.loss = tf.reduce_mean(tf.reduce_sum(self.l2_error(self.delta),1)) + reg_term  #the loss is just use the absulote deviation
            self.learning_rate_op = tf.maximum(self.Config.LearningRateMinimum,
                                               tf.train.exponential_decay(
                                                   self.Config.LearningRateMaximum,
                                                   self.step_input,
                                                   self.Config.LearningRateDecayStep,
                                                   self.Config.LearningRateDecayRate,
                                                   staircase=True))  # 进行学习速率的提取，其中一部分是最小速率，剩下的通过一个速率学习出的
            self.optim = tf.train.AdamOptimizer(
                self.learning_rate_op).minimize(self.loss)
            self.acc = tf.reduce_mean(
                 tf.abs(tf.divide(self.delta, self.y_out)))
            #self.acc = tf.reduce_mean(
                #tf.abs(tf.divide(tf.multiply(self.delta, self.std), tf.multiply(self.Y, self.std) + self.mean)))
            # summary_op
            summary_wb_op = []
            for (k, v) in self.w.items():
                summary_wb_op.append(tf.summary.histogram("%s" % k, v))
            for (k, v) in self.b.items():
                summary_wb_op.append((tf.summary.histogram("%s" % k, v)))
            # summar the acc and loss
            self.summary_acc_op = []
            self.summary_acc_op.append(tf.summary.scalar("loss", self.loss))
            self.summary_acc_op.append(tf.summary.scalar("accuracy", self.acc))
            self.merged_summary = tf.summary.merge_all()

            self.Sess.run(tf.global_variables_initializer())
            # build the saver for the model
            self.saver = tf.train.Saver(list(self.w.values()) + list(self.b.values()) + [self.step_op],
                                        max_to_keep=30)

            print("!!!!!FINISH BUILD!!!!!!!!")
            self.load_model()
        return

    def Train(self,trainX, trainY,validX, validY,mean, std):
        if not os.path.exists(self.Config.TFBoardPathTrain):
            os.makedirs(self.Config.TFBoardPathTrain)
        if not os.path.exists(self.Config.TFBoardPathValid):
            os.makedirs(self.Config.TFBoardPathValid)
        self.summaryWriter = tf.summary.FileWriter(self.Config.TFBoardPathTrain)
        self.validsummaryWriter = tf.summary.FileWriter(self.Config.TFBoardPathValid)
        self.summaryWriter.add_graph(self.Sess.graph)
        self.Sess.run(self.step_assign_op, feed_dict = {self.step_input:0})
        testacc = []
        testloss = []
        currentloss = 0
        minloss = 0
        validBatchSize = len(validX) // self.Config.TestNum
        for i in tqdm(range(self.step_op.eval(),self.Config.MaxTrainStep)):
            index = random.randint(0,len(trainX))
            self.Sess.run(self.optim,{self.X:trainX[index%len(trainX)],self.Y:trainY[index%len(trainX)], self.step_input:i,self.keepProb:self.Config.dropoutKeepProb})
            if i% self.Config.TestStep == self.Config.TestStep-1:
                for j in range(0,self.Config.TestNum-1):
                    logging.debug("validXsize:%d,%d,%d,%d"%(validX[validBatchSize*j:validBatchSize*(j+1)].shape[0], validX[validBatchSize*j:validBatchSize*(j+1)].shape[1],validX[validBatchSize*j:validBatchSize*(j+1)].shape[2],validX[0:100].shape[3]))
                    logging.debug("begin%d, is %d"%(j, j*validBatchSize))
                    logging.debug("begin%d, is %d"%(j+1, (j+1)*validBatchSize))
                    tempacc = self.acc.eval({self.X:validX[(validBatchSize*j):(validBatchSize*(j+1))], self.Y:validY[validBatchSize*j:validBatchSize*(j+1),...], self.keepProb:1})
                    testacc.extend([tempacc])
                    temploss = self.loss.eval({self.X: validX[(validBatchSize * j):(validBatchSize * (j + 1))],
                                             self.Y: validY[validBatchSize * j:validBatchSize * (j + 1), ...], self.keepProb:1})
                    testloss.extend([temploss])
                testacc.extend([self.acc.eval({self.X:validX[validBatchSize*(self.Config.TestNum-1):len(validX),...], self.Y:validY[validBatchSize*(self.Config.TestNum-1):len(validX),...], self.keepProb:1})])
                testloss.extend([self.loss.eval({self.X:validX[validBatchSize*(self.Config.TestNum-1):len(validX),...], self.Y:validY[validBatchSize*(self.Config.TestNum-1):len(validX),...], self.keepProb:1})])
                # testacc, testloss = self.Sess.run([self.acc,self.loss], {self.X:validX, self.Y: validY})
                trainacc, trainloss = self.Sess.run([self.acc, self.loss],{self.X:trainX[index%len(trainX)],self.Y:trainY[index%len(trainX)], self.step_input:i, self.keepProb:1})
                str = self.Sess.run(self.merged_summary, feed_dict = {self.X:trainX[index%len(trainX)], self.Y:trainY[index%len(trainY)], self.keepProb:1})
                self.summaryWriter.add_summary(str,global_step=i)
                strvalidlist = self.Sess.run(self.summary_acc_op,feed_dict ={self.X:validX[validBatchSize*(self.Config.TestNum-1):len(validX),...], self.Y:validY[validBatchSize*(self.Config.TestNum-1):len(validX),...], self.keepProb:1} )
                for strvalid in strvalidlist:
                    self.validsummaryWriter.add_summary(strvalid, global_step=i)

                meantestacc = np.array(testacc).mean()
                meantestloss = np.array(testloss).mean()
                # early stopping
                currentloss = meantestloss
                if i == self.Config.TestStep-1:
                    minloss = currentloss
                if i > self.Config.SaveStep:
                    flag, minloss=IfOverfitting(currentloss,minloss)
                    if flag == True:
                        print("!!!!!There is overfitting!!!!!")
                        return


                print(20 * "==")
                print("the accuracy is %f;  the loss is %f" %(meantestacc, meantestloss))
                print(20*"**")
                print("the TRAIN acc is %f;   the TRAIN loss is %f"%(trainacc, trainloss))
                print(20*"==")
                ## stop if it is overfitting



            if i% self.Config.SaveStep == self.Config.SaveStep-1:
                self.step_assign_op.eval({self.step_input:i})
                self.save_model(step=i)
        return


    def l2_error(self, x):
        return 0.5 * tf.square(x)

    def save_model(self, step=None):
        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__
        save_model_name = os.path.join(self.Config.CheckpointDir, model_name)
        if not os.path.exists(self.Config.CheckpointDir):
            os.makedirs(self.Config.CheckpointDir)
        self.saver.save(self.Sess, save_model_name, global_step=step)  # the step represent the postfix of the save path

    def load_model(self):
        print(" [*] Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.Config.LoadModelDir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.Config.LoadModelDir, ckpt_name)
            self.saver.restore(self.Sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.Config.LoadModelDir)
            return False

    def Play(self,x,y,mean,std):
        print("!!!!!start Play!!!!!")
        self.load_model()
        pred = self.linear7.eval({self.X:x, self.Y:y, self.keepProb:1})
        pred_out = pred * std + mean
        y_out = y*std + mean
        acc = self.acc.eval({self.X:x,self.Y:y, self.keepProb:1})

        if not os.path.exists(self.Config.PlayResultDir):
            os.makedirs(self.Config.PlayResultDir)
        np.save(os.path.join(self.Config.PlayResultDir, "pred.npy"), pred_out)
        np.save(os.path.join(self.Config.PlayResultDir, "standard.npy"), y_out)
        with open(os.path.join(self.Config.PlayResultDir, "results.txt"), 'a') as f:
            f.write(30*"#" + str(acc) + 30* "#" + '\n')
            for i in range(0,pred_out.shape[0]):
                f.write(30 * "=" + "\n")
                f.write(str(pred_out[i,...]) + "\n")
                f.write(str(y_out[i,...]) + "\n")
                f.write("\n" + 30 * "=" + "\n")
        print(np.mean(np.abs(pred_out-y_out), axis = 0))
        print(np.std(np.abs(pred_out-y_out), axis = 0))
        return

    pass