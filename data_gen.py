
import numpy as np
from collections import Counter
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm as tqdm_plain

from keras import backend as K
import time
import numpy as np_utils

np.random.seed(2017)
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, DepthwiseConv2D, Conv2D, SeparableConv2D, \
    MaxPooling1D
from keras.layers import Input, concatenate
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Nadam, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l2
from keras_contrib.callbacks import CyclicLR
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from data_science_utils.vision.keras import *
from time import time
import pandas as pd
import numpy as np
from keras.layers.core import SpatialDropout2D,SpatialDropout1D
import missingno as msno
import re
from joblib import Parallel, delayed
from data_science_utils import dataframe as df_utils
from data_science_utils import models as model_utils
from data_science_utils import plots as plot_utils
from data_science_utils.dataframe import column as column_utils
from data_science_utils import misc as misc
from data_science_utils import preprocessing as pp_utils
from data_science_utils import nlp as nlp_utils

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from data_science_utils.dataframe import get_specific_cols
from nltk import word_tokenize
import itertools

import more_itertools
from more_itertools import flatten
import ast
from sklearn.preprocessing import LabelEncoder
from params import *
from lib import *

def batch_flow_data(df,extract_fn,batch_size,n_batch_groups=10):
    # print("Flowing %s samples in %s batch sizes"%(len(df),self.batch_size))
    
    for k,g in df.groupby(np.arange(len(df))//(batch_size*n_batch_groups)):
        # Processing
        flowed_data = extract_fn(g)
        if len(flowed_data)==2:
            X,y = flowed_data
            y = [y]
            for Xs,ys in list(zip(array_split(n_batch_groups,*X),array_split(n_batch_groups,*y))):
                yield Xs,ys
        elif len(flowed_data)==1:
            X = flowed_data[0]
            for Xs in list(array_split(n_batch_groups,*X)):
                yield Xs
        else:
            raise ValueError()


class MakeIter(object):
    def __init__(self,df,extract_fn,batch_size):
        self.batch_generator_func = batch_flow_data
        self.batch_size = batch_size
        self.length = int(np.ceil(len(df)/batch_size))
        self.df = df
        self.extract_fn = extract_fn
        self.iter = self.batch_generator_func(self.df.sample(frac=1),self.extract_fn,self.batch_size)
        
        
    def __iter__(self):
        while True:
            try:
                yield next(self.iter)
            except StopIteration:
                self.iter = self.batch_generator_func(self.df.sample(frac=1),self.extract_fn,self.batch_size)
            
    
        
    def __len__(self): 
        return self.length
    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = self.batch_generator_func(self.df.sample(frac=1),self.extract_fn,self.batch_size)
            return next(self.iter)
            
        
    def next(self):
        return next(self.iter)
    

