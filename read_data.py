import time, os, sys
import struct
from os import listdir
from os.path import isfile, join
import ctypes, math
import csv, struct
import numpy, scipy
#import pandas as pd

def file_array(file): # read a file and write to array
  array=[]
  with open(file, 'r') as f: #open file
    csv_reader = csv.reader(f, delimiter=',')
    array=numpy.loadtxt(file,dtype=str,delimiter=',',skiprows=0,usecols=(0,))
    return array


def path_to_list(pathF):
  list=[]
  listarray=[]
  for filename in os.listdir(pathF): #filename             
    pathN=pathF+filename # absolute path for every file
    list = file_array(pathN) # return VEH vector
    listarray.append(list)
    return listarray

def get_label(name):
  if name is "OkayGoogle":
      label=0
  else: label=1


def read_data(path):
	
  file=path + "file.txt"
  list1 = ["flatTop","verticalTop"]
  list2 = ["FineThankYou","GoodMorning","HowAreYou","OkayGoogle"]
	
  with open(file) as f:

     list0 = f.readlines()
     list0 = [x.strip() for x in list0] # individuals
     listdata=[]
     data0=[]
     data1=[]
     data2=[]
     data3=[]
     matlabel=[]
     for i in range(0, len(list0)):
      path2=path+list0[i]+'/'
      #print "path2=",path2
      for j in range(0, len(list1)): #for the both directions
         path3=path2+list1[j]+"/" 
         #print "path3=",path3
         for k in range(0, len(list2)): #for all sentences
          path4=path3+list2[k]+"/"
          #print "path4=",path4
          for filename in os.listdir(path4): #filename
            list3=[] 
            path5=path4+filename # absolute path for every file
            #print "path5=",path5  
            list3 = file_array(path5) # return VEH vector (temporal domain)
            list3=map(int,list3)
            listdata.append(list3)
            if list2[k] is "OkayGoogle":
               label=0
               data0.append(list3)
            else: 
               label=1
               data1.append(list3)
            #elif list2[k] is "FineThankYou":
            #   label=1
            #   data1.append(list3)
            #elif list2[k] is "HowAreYou":
            #   label=2
            #   data2.append(list3)
            #elif list2[k] is "GoodMorning":
            #   label=3
            #   data3.append(list3)
            matlabel.append(label)
            #print "user :",list0[i]," dir :",list1[j]," sentence: ",list2[k], "label: ", label
  return listdata,matlabel