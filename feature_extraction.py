import tensorflow as tf
import numpy as np

from data_preprocessing import *
from autoencoder import *

# features learned by the autoencoder
def feature_extraction(data,noise,ratio,sparse,lamda,beta,dim,hid,act_func,lr,momentum,weight_init,
                                   loss_func,opt,batch,ep,wen,wdec,ben,bdec):
       features,w,b = run_session(data,noise,ratio,dim,hid,act_func,lr,momentum,weight_init,
                                   loss_func,opt,batch,ep,wen,wdec,ben,bdec)

       features = normalize_data(features,'scaling')
       return features

# for combining two lists
def combine_list(data1,data2):

       data = []
       for x in xrange(0,len(data1)):

           vec = data1[x] + data2[x] 
           data.append(vec)

       data = normalize_data(data,'scaling')
       return data
