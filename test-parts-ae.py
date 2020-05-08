#!/usr/bin/env python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from functions import *
import math
import random
from tabulate import tabulate
import sys
import numpy as np
import scipy.io as sio
##import pandas
import scipy
import csv
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tempfile import TemporaryFile
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functions2 import *
from classification2 import *
from autoencoder import *

# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - -  --  -- -  - - -
# read and prepare the data and labels from files
# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - -  --  -- -  - - -

def select_from_classifier2(data,y):

    clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=100,max_depth=1,class_weight="balanced"),n_estimators=1000)
    model = SelectFromModel(clf).fit(data,y)	
    indices = model.get_support(indices=True)
    data_n = model.transform(data)
    new_data = scale(data_n)
    return new_data

# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - 
# get the parameters from terminal
# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('task', 'hot_or_not', 'data task') 
flags.DEFINE_string('norm', 'scaling', 'data normalization') 
flags.DEFINE_integer('mod', '1', 'data modality')

flags.DEFINE_boolean('noise', False, 'noise') 
flags.DEFINE_float('ratio', 0.2, 'noise ratio') 

flags.DEFINE_boolean('sparse', False, 'sparsity') 
flags.DEFINE_float('beta', 0.1, 'brta') 
flags.DEFINE_float('lamda', 0.2, 'lamda') 


flags.DEFINE_integer('h1', 1200, 'hidden layer size') # hidden layer size
flags.DEFINE_integer('h2', 1000, 'hidden layer size') # hidden layer size
flags.DEFINE_integer('h3', 728, 'hidden layer size') # hidden layer size
flags.DEFINE_integer('h4', 512, 'hidden layer size') # hidden layer size
flags.DEFINE_integer('h5', 256, 'hidden layer size') # hidden layer size



flags.DEFINE_integer('ep1', 100, 'epochs') # batch size
#flags.DEFINE_integer('ep2', 100, 'epochs') # batch size
flags.DEFINE_integer('ep', 1000, 'epochs') # batch size


flags.DEFINE_float('lr1', 0.001, 'learning_rate') # learning rate
#flags.DEFINE_float('lr2', 0.001, 'learning_rate') # learning rate

flags.DEFINE_float('momentum', 0.7, 'momentum') # momentum

flags.DEFINE_integer('bsize1', 30, 'batch size') # batch size
#flags.DEFINE_integer('bsize2', 30, 'batch size') # batch size
flags.DEFINE_integer('bsize', 30, 'batch size') # batch size


flags.DEFINE_string('act', 'sigmoid', 'activation function') #tanh, relu, selu,elu,linear
flags.DEFINE_string('we', 'xavier_uniform', 'weight init') #'xavier_uniform','xavier_normal','he_normal','he_uniform','caffe_uniform'
flags.DEFINE_string('loss', 'mse', 'loss function') #cross_entropy, mse
flags.DEFINE_string('opt', 'Rmsprop', 'optimizer') # momentum,ada_grad, gradient_descent

#data_name = FLAGS.data
task = FLAGS.task
norm = FLAGS.norm # for early fusion
mod = FLAGS.mod
noise = FLAGS.noise
ratio = FLAGS.ratio
sparse = FLAGS.sparse
beta = FLAGS.beta
lamda = FLAGS.lamda


dim1 = FLAGS.h1
dim2 = FLAGS.h2
dim3 = FLAGS.h3
dim4 = FLAGS.h4
dim5 = FLAGS.h5

ep1 = FLAGS.ep1
#ep2 = FLAGS.ep2
ep = FLAGS.ep

lr1 = FLAGS.lr1
#lr2 = FLAGS.lr2

batch1 = FLAGS.bsize1
#batch2 = FLAGS.bsize2
batch = FLAGS.bsize

batch3 = batch4 = batch5 = batch2 = batch1
lr3 = lr4 = lr5 = lr2 = lr1
ep3 = ep4 = ep5 = ep2 = ep1

weight_init = FLAGS.we   
loss_func = FLAGS.loss
opt = FLAGS.opt
act_func=FLAGS.act
momentum = FLAGS.momentum


print("get the parameters ")
print("-----------------------------------------")
print (" hid1 = ", dim1, ",hid2 = ",dim2, ",hid3= ", ",epochs = ",ep1)
print (" learning rate = ", lr1, "batch size = ", batch1, "optimizer = ", opt)
print (" noise: ", noise, "noise ratio = ", ratio, " sparsity = ",sparse, "beta = ",beta, "lambda = ",lamda)
    #, sparse, "L2 reg = ", sparse_reg)
print("-----------------------------------------")

runs = 1
task = "gender" #gender
name = "complexity2" #complexity2,race
des = "lbp"
clas = "svm"
typed=["RGB2","DEPTH2","RGBD"]
weight = ["accloss","accloss","acc","meanlog"] 
seed = 42
parts = ["cheeknose","chin+mouth+jaw","eyes","chin","nose","nosemouth","face"]

for y in range(len(typed)):

	for z in range(len(weight)):
		print("-----------------------------------------------------------------------------------------")
		print("task: ",task,"data type",typed[y],"weight",weight[z])
		print("-----------------------------------------------------------------------------------------")
		print (" read + prepare the data : ===== ...")

		for r in range(runs):
		    labels = np.genfromtxt('../files/labels/labels'+name+'.txt', dtype=int, delimiter=',', names=None)
		    i = 0

		    for x in range(len(parts)):		
				classifier = svm.SVC(gamma= 0.001,C =1000,class_weight="balanced",random_state=7)#changed
                print(parts[x])
 				if typed[y] =="RGBD":
					path1 = '../files/'+task+'/'+ name +'-'+parts[x]+'-'+des+'-'+'RGB2'+'.txt'
					path2 = '../files/'+task+'/'+ name +'-'+parts[x]+'-'+des+'-'+'DEPTH2'+'.txt'
					path = '../files/'+task+'/'+ name +'-'+parts[x]+'-'+des+'-'+typed[y]+'.txt'
					data1 = np.genfromtxt(path1, dtype=float, delimiter=',', names=None)
					data2 = np.genfromtxt(path2, dtype=float, delimiter=',', names=None)
					data = np.genfromtxt(path, dtype=float, delimiter=',', names=None)
					X1 = data1[:, :-1]
					X2 = data2[:, :-1]
					XX = conc_data(X1,X2)
					XXX = scale(XX)
					Xs,Y = shuffle(XXX,labels)
					X = select_from_classifier2(Xs,Y)

				else: 
					path = '../files/'+task+'/'+ name +'-'+parts[x]+'-'+des+'-'+typed[y]+'.txt'
					data = np.genfromtxt(path, dtype=float, delimiter=',', names=None)
					X1 = data[:, :-1]
					X,Y = shuffle(X1,labels)
				
				features,_,_ = run_session(X,noise,ratio,sparse,beta,lamda,vt,dim1,act_func,lr1,momentum,weight_init,
                                   loss_func,opt,batch1,ep1,"wvt1en","wvt1dec","bvt1en","bvt1dec")

				F = scale(features)


    			kf = StratifiedKFold(n_splits=n,shuffle=True)
        		scores = cross_val_score(classifier, F, y, cv=kf)
        		avg_score = numpy.mean(scores)
        		print(" cross validation acc == ",avg_score)