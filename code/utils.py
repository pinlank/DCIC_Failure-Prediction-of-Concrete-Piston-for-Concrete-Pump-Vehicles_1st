import time
import sys
import os
import re
import gc
import datetime
import itertools
import pickle
import random
import numpy as np 
import pandas as pd 
from utils import *
from scipy import stats
import warnings
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold,train_test_split,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, f1_score, log_loss,roc_auc_score

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename
 
def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

def static_fe(data1,data2,df,column,values,cc,c):
    addn = df[[column,values]].copy()
    addn = addn.groupby(column)[values].agg(cc).reset_index()
    addn.columns = [column] + [c+values+'_'+i for i in cc]
    data1 = data1.merge(addn,on=column,how='left')
    data2 = data2.merge(addn,on=column,how='left')
    return data1,data2

def cons(x):
    num_times = [(k, len(list(v))) for k, v in itertools.groupby(list(x))]
    num_times = pd.DataFrame(num_times)
    num_times = num_times[num_times[0] == 1][1]
    return num_times.max()

def cons_fe(data,df,column,values):
    kk = df.groupby(column)[values].apply(cons)
    kk = kk.fillna(0).astype('int32').reset_index()
    kk.columns = [column,'cons_' + values]
    data = data.merge(kk, on=column, how='left')
    return data

def get_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

def auc(y,pred):
#     fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    return roc_auc_score(y, pred)

def f1(y,pred):
#     fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    return f1_score(y, pred,average='macro')