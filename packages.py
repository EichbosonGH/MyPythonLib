import time
from time import ctime
import os
import sys
import subprocess
import glob
import string
import re
import copy
import json
#import requests
import calendar
import pickle

# import IPython
import numpy as np
import pandas as pd
import scipy
import scipy.stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

import holidays
feierdays = holidays.Germany()

import sklearn
import sklearn.metrics # confusion_matrix,roc_curve,auc
import sklearn.model_selection # ShuffleSplit,StratifiedShuffleSplit,KFold,StratifiedKFold,train_test_split
import sklearn.preprocessing # MinMaxScaler,RobustScaler,StandardScaler,LabelEncoder
import sklearn.linear_model # LogisticRegression
import sklearn.neural_network # MLPRegressor
import sklearn.manifold # MDS, tSNE etc.

plt.rc('axes', axisbelow=True)
plt.rcParams['figure.max_open_warning']=100
plt.rcParams["figure.dpi"]=150
plt.rcParams["figure.autolayout"]=False # true for tight layout, no adjacent subplots possible then
plt.rcParams['axes.grid']=True
pd.set_option('display.width',200)
pd.set_option('display.max_columns',60)
pd.set_option('display.max_rows',60)
pd.set_option('display.max_colwidth',100)

import xgboost as xgb
import catboost as ctb

print(f'--- {"Datum:":<15} {ctime()}')
print(f'--- {"Holidays:":<15} {holidays.__version__}')
print(f'--- {"PID:":<15} {os.getpid()}')
print(f'--- {"Python:":<15} {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ({sys.version_info.releaselevel})')
print(f'--- {"Numpy:":<15} {np.__version__}')
print(f'--- {"Pandas:":<15} {pd.__version__}')
print(f'--- {"Matplotlib:":<15} {mpl.__version__}')
print(f'--- {"Scipy:":<15} {scipy.__version__}')
print(f'--- {"Scikit-learn:":<15} {sklearn.__version__}')
print(f'--- {"Statsmodels:":<15} {sm.__version__}')
print(f'--- {"XGBoost:":<15} {xgb.__version__}')
