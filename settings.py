import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import re
import datetime
import requests
from bs4 import BeautifulSoup
from workalendar.asia import China
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
from keras.layers import GRU, Dropout, BatchNormalization
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from tqdm.notebook import tqdm
import copy
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
import torch
import torch.nn as nn
import random
import os
import time
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

# 表名控制
holiday_table = "t_holiday_info"
patient_table = "t_patient_info"
weather_table = "t_weather_info"
predict_table = "20211208_tmp_yhy"

# 控制模型输出的可重复性
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONHASHSEED"] = str(1)
tf.random.set_seed(1)
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

# 超参数调整
# 数据处理部分
history_day = 400
predict_day = 6

# 模型参数
lag = 5
batch_size = 64
layers = 2
hidden_dim = 32
lr = 0.01
epochs = 300
model_path = "./model.pkl"
# mysql连接信息
MYSQL_CONFIG = {
    'host': '192.168.9.157',
    'user': 'xwtech',
    'passwd': 'hwfx1234',
    'db': 'datasquare',
    'port': 3306,
    'charset': 'utf8'
}