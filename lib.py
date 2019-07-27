import numpy as np
from collections import Counter
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm as tqdm_plain
from keras.regularizers import l2
from keras.regularizers import L1L2

from keras import backend as K
import time
import numpy as np_utils


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
from sklearn.metrics import f1_score

preprocess_string = lambda x:re.sub('[^ a-zA-Z0-9%@_]',' ',nlp_utils.clean_text(x)) if x is not None and type(x)==str else x

def build_dict(data, vocab_size=100000, min_count=1):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""

    word_count = Counter()  # A dict storing the words that appear in the reviews along with how often they occur
    for sentence in tqdm_plain(data):
        word_count.update(sentence)

    print("Total Words before Min frequency filtering", len(word_count))
    sorted_words = [word for word, freq in word_count.most_common() if freq >= min_count]
    print("Total Words after Min frequency filtering", len(sorted_words))
    word_dict = {}  # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]):  # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2  # 'infrequent' labels

    return word_dict


def get_text_le(vocab_size=100000, min_count=1):
    le = {}
    INFREQ = 1
    NOWORD = 0
    UNKNOWN_TOKEN = '<unknown>'

    def le_train(texts):
        le['wd'] = build_dict(texts, vocab_size=vocab_size, min_count=min_count)
        return le['wd']

    def word2label(word):
        word_dict = le['wd']
        if word in word_dict:
            return word_dict[word]
        else:
            return INFREQ

    def wordarray2labels(wordarray):
        return list(map(word2label, wordarray))

    def le_transform(texts):
        word_list = [wordarray2labels(x) for x in tqdm_plain(texts)]
        return word_list

    return le_train, le_transform, le



def preprocess_for_word_cnn(df, text_column='text_raw', output_column="text", word_length_filter=2, jobs=16):
    """
    Preprocess and convert all text columns to one column named text
    """
    pp = lambda text: nlp_utils.combined_text_processing(text, word_length_filter=word_length_filter)
    text = Parallel(n_jobs=jobs, backend="loky")(delayed(pp)(x) for x in tqdm_plain(df[text_column].fillna(' ').values))
    df[output_column] = text
    return df



def conv_layer(inputs, n_kernels=32, kernel_size=3, dropout=0.2,spatial_dropout=0.2,
               dilation_rate=1, padding='valid', strides=1):
    out = Conv1D(n_kernels,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding=padding,
                 kernel_regularizer=l2(1e-6),
                 dilation_rate=dilation_rate)(inputs)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = SpatialDropout1D(spatial_dropout)(out) if spatial_dropout > 1e-8 else out
    if dropout > 1e-8:
        out = Dropout(dropout)(out)
    return out

def fc_layer(inputs, neurons=32, dropout=0.2, bn=True):
    out = Dense(neurons, kernel_regularizer=L1L2(1e-6,1e-6))(inputs)
    out = BatchNormalization()(out) if bn else out
    out = Activation('relu')(out)
    if dropout > 1e-8:
        out = Dropout(dropout)(out)
    return out


def transition_layer(inputs, n_kernels=32, dropout=0):
    out = conv_layer(inputs, n_kernels, kernel_size=1, dropout=dropout, padding='same')
    return out


def pre_dense_layer(inputs):
    out1 = GlobalAveragePooling1D()(inputs)
    out2 = GlobalMaxPooling1D()(inputs)
    out = concatenate([out1, out2])
    return out

def pad_text_sequences(sequences,maxlen,empty='',jobs=2):
    def pad(seq):
        ls = len(seq)
        len_empty = maxlen - ls
        if len_empty<=0:
            return seq[:maxlen] if ls>maxlen else seq
        return [empty]*len_empty + list(seq)
    sequences = Parallel(n_jobs=jobs, backend="loky")(delayed(pad)(x) for x in sequences)
    return sequences

class PreTrainedEmbeddingsTransformer:
    def __init__(self, model="fasttext-wiki-news-subwords-300", size=300,
                 normalize_word_vectors=True):
        self.normalize_word_vectors = normalize_word_vectors
        self.model = model
        self.size = size
        self.token2vec_dict = {}

    def fit(self, X=None, y='ignored'):
        if type(self.model) == str:
            self.model = api.load(self.model)

    def partial_fit(self, X=None, y=None):
        self.fit(X, y='ignored')

    def transform(self, X, y='ignored'):
        uniq_tokens = set(more_itertools.flatten(X))
        uniq_tokens = uniq_tokens - self.token2vec_dict.keys()
        empty = np.full(self.size, 0)
        token2vec = {k: self.model.wv[k][:self.size] if k in self.model.wv else empty for k in uniq_tokens}
        # token2vec = {k: np.nan_to_num(v / np.linalg.norm(v)) for k, v in token2vec.items()}
        token2vec = {k: np.nan_to_num(v) for k, v in token2vec.items()}
        self.token2vec_dict.update(token2vec)
        token2vec = self.token2vec_dict
        uniq_tokens = set(token2vec.keys())
        def tokens2vec(token_array):
            empty = np.full(self.size, 0)
            if len(token_array) == 0:
                return [empty]
            return [token2vec[token] if token in uniq_tokens else empty for token in token_array]

        # ft_vecs = list(map(tokens2vec, X))
        ft_vecs = [tokens2vec(t) for t in X]
        ft_vecs = np.array(ft_vecs)
        return ft_vecs

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y='ignored'):
        self.fit(X)
        return self.transform(X)
    
    
from numpy import dot
from numpy.linalg import norm
def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def vec_norm(v):
    return np.nan_to_num(v / np.linalg.norm(v))

def show_results(y_true,y_pred):
    acc = accuracy_score(y_true,y_pred)
    f1_mac = f1_score(y_true, y_pred, average='macro')
    print("Accuracy = %.2f, Macro F1 = %.2f"%(acc,f1_mac))
    return (acc,f1_mac)


def find_occurence_count(df):
    vals = list(zip(df.text.values,df.drug.values))
    def locater(x):
        return [m.start() for m in re.finditer(x[1], x[0])]
    occ = list(map(locater,vals))
    df["occurences"] = occ
    df["occurences_count"] = df.occurences.apply(len)
    return df

def extract_surround(text,index,n_words):
    i = index
    j = n_words+1
    ctr = 0
    wahead = []
    idx = i
    while ctr<j and idx<len(text):
        wahead.append(text[idx])
        idx = idx+1
        if idx>=len(text):
            break
        if text[idx]==" ":
            ctr = ctr+1

    wahead = "".join(wahead)
    
    wback = []
    text = "".join(list(reversed(text)))
    j = n_words
    i = len(text)-i
    wback = []
    idx = i
    ctr = 0
    
    while ctr<j and idx<len(text):
        
        wback.append(text[idx])
        idx = idx+1
        if idx>=len(text):
            break
        if text[idx]==" ":
            ctr = ctr+1

    wback = "".join(list(reversed(wback)))
    return wback+wahead

def get_surrounding_text(df,surround):
    vals = zip(df.text.values,df.occurences.values,df.drug.values)
    def extract(x):
        text = x[0]
        occs = x[1]
        drug = x[2]
        sc = ""
        for i in occs:
            s = extract_surround(text,i,surround)
            sc = sc + s

        return sc
    context = list(map(extract,vals))
    
    return context


import math
def normpdf(mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    def pdf_calculator(x):
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom
    return pdf_calculator

def make_mask_single(arr,value,mask_spread):
    if type(value)!=list:
        if type(value)==str:
            value = word_tokenize(value)
            value = set(value)
        else:
            value = set([value])
    else:
        value = set(value)
    lv = len(value)
    
    padding = [None] * (mask_spread - 1)
    vals = list(more_itertools.windowed(itertools.chain(padding, arr), mask_spread))
    
    mask = []
    gau_mask = []
    for val in vals:
        
        v = set(val)
        il = len(v.intersection(value))
        if il>0:
            gau_mask.append(1)
            mask.append(il/lv)
        else:
            gau_mask.append(0)
            mask.append(0)
    guas = [0]*len(gau_mask)  
    pdf_fn = normpdf(0,mask_std)
    for i,g in enumerate(gau_mask):
        
        if gau_mask[i]>0:
            for j in range(max(i-(mask_spread*3),0),min(i+(mask_spread*3),len(gau_mask))):
                guas_value = j-i
                guas[j] = guas[j] + pdf_fn(guas_value)
    assert len(mask)==len(arr)   
    return mask,guas


def make_mask(df):
    vals = list(zip(df.full_txt.values,df.drug.values))
    msfn = lambda x: make_mask_single(x[0],x[1],mask_spread)
    masks = list(map(msfn,vals))
    
    
    masks = [list(t) for t in zip(*masks)]
    
    df['full_txt_mask'] = masks[0]
    df['full_txt_mask_gaussian'] = masks[1]
    return df

def read_csv(filename):
    df = pd.read_csv(filename)
    
    df['occurences'] = Parallel(n_jobs=jobs, backend="loky")(delayed(ast.literal_eval)(x) for x in df['occurences'].values)
    df['context_txt'] = Parallel(n_jobs=jobs, backend="loky")(delayed(ast.literal_eval)(x) for x in df['context_txt'].values)
    df['full_txt'] = Parallel(n_jobs=jobs, backend="loky")(delayed(ast.literal_eval)(x) for x in df['full_txt'].values)
    if 'ohe_labels' in df.columns:
        df['ohe_labels'] = Parallel(n_jobs=jobs, backend="loky")(delayed(ast.literal_eval)(x) for x in df['ohe_labels'].values)
    df['full_txt_mask_gaussian'] = Parallel(n_jobs=jobs, backend="loky")(delayed(ast.literal_eval)(x) for x in df['full_txt_mask_gaussian'].values)
    df['full_txt_mask'] = Parallel(n_jobs=jobs, backend="loky")(delayed(ast.literal_eval)(x) for x in df['full_txt_mask'].values)
    return df

def shuffle_copy(*args):
    rng_state = np.random.get_state()
    results = []
    for arg in args:
        res = np.random.shuffle(np.copy(arg))
        results.append(res)
        np.random.set_state(rng_state)
    return results[0] if len(args)==1 else results

def array_split(splits,*arrays):
    data = [np.array_split(arr,splits) for arr in arrays]
    results = [[row[i] for row in data] for i in range(len(data[0]))]
    return results


class Metrics(keras.callbacks.Callback):
    def __init__(self,train,val):
        self.train = train
        self.val = val
    def on_epoch_end(self, batch, logs={}):
        
        predict = np.asarray(self.model.predict(self.train[0]))
        targ = self.train[1]
        targ = np.argmax(targ, axis=1)
        predict = np.argmax(predict, axis=1)
        f1s_train=f1_score(targ, predict, average="macro")
        
        predict = np.asarray(self.model.predict(self.val[0]))
        targ = self.val[1]
        targ = np.argmax(targ, axis=1)
        predict = np.argmax(predict, axis=1)
        f1s=f1_score(targ, predict, average="macro")
        print("Validation F1 Score = %.4f, Train F1 Score = %.4f",(f1s,f1s_train))
        return
    
# https://github.com/keras-team/keras/issues/5794
# https://github.com/keras-team/keras/issues/10472
class DataGenMetrics(keras.callbacks.Callback):
    def __init__(self,train,val):
        self.train = train
        self.val = val
        self.train_batches = len(train)
        self.val_batches = len(val)
    
    def exec_iter(self,iterator,length):
        
        y_preds = []
        y_true = []
        for i in range(length):
            x,y = next(iterator)
            y_pred = self.model.predict(x)
            y_pred = np.asarray(y_pred)
            y = np.asarray(y)
            y = y.reshape(y_pred.shape)
            
            y = list(np.argmax(y, axis=1))
            y_pred = list(np.argmax(y_pred, axis=1))
            y_true = y_true + y
            y_preds = y_preds + y_pred
            
        f1s=f1_score(y_true, y_preds, average="macro")
        return f1s
        
    def on_epoch_end(self, batch, logs={}):
        
        f1s_train = self.exec_iter(self.train,self.train_batches)
        f1s_val = self.exec_iter(self.val,self.val_batches)
        print("Validation F1 Score = %.4f, Train F1 Score = %.4f"%(f1s_val,f1s_train))
        return
    
    
def batch_cutout(*args,**options):
    """
    Works only on padded equal length sequences and only on numpy array
    """
    p = options.pop('p', 0.5)
    min_words = options.pop('min_words', 2)
    max_words = options.pop('max_words', 5)
    p_1 = np.random.rand()
    if p_1 > p:
        return args
    mx = np.random.randint(min_words, max_words + 1)
    start = np.random.randint(0, len(args[0][0]))
    end = min(start+mx,len(args[0][0]))
    for arg in args:
        # print(start,end,"Shape = ",arg[0].shape,)
        arg[:,start:end] = 0
    
    return args

def cutout(*args,**options):
    p = options.pop('p', 0.5)
    min_words = options.pop('min_words', 2)
    max_words = options.pop('max_words', 5)
    mx = np.random.randint(min_words, max_words + 1)
    for i in range(len(args[0])):
        p_1 = np.random.rand()
        if p_1 > p:
            continue
        
        start = np.random.randint(0, len(args[0][i]))
        end = min(start+mx,len(args[0][i]))
        for arg in args:
            arg[i] = np.array(arg[i])
            # print(start,end,"Shape = ",arg[i].shape,)
            arg[i][start:end] = 0
    
    return args




class PreTrainedDocEmbeddingsTransformer:
    def __init__(self, model="fasttext-wiki-news-subwords-300", start=0,end=300,
                 normalize_word_vectors=True):
        self.normalize_word_vectors = normalize_word_vectors
        self.model = model
        self.start = start
        self.end = end
        self.size = end - start
        assert end > start
        self.token2vec_dict = {}

    def fit(self, X=None, y='ignored'):
        if type(self.model) == str:
            self.model = api.load(self.model)

    def partial_fit(self, X=None, y=None):
        self.fit(X, y='ignored')

    def transform(self, X, y='ignored'):
        uniq_tokens = set(more_itertools.flatten(X))
        uniq_tokens = uniq_tokens - self.token2vec_dict.keys()
        empty = np.full(self.size, 0)
        token2vec = {k: self.model.wv[k][self.start:self.end] if k in self.model.wv else empty for k in uniq_tokens}
        # token2vec = {k: np.nan_to_num(v / np.linalg.norm(v)) for k, v in token2vec.items()}
        token2vec = {k: np.nan_to_num(v) for k, v in token2vec.items()}
        self.token2vec_dict.update(token2vec)
        token2vec = self.token2vec_dict
        uniq_tokens = set(token2vec.keys())
        def tokens2vec(token_array):
            empty = np.full(self.size, 0)
            if len(token_array) == 0:
                return empty
            vec = np.asarray([token2vec[token] if token in uniq_tokens else empty for token in token_array])
            vec = np.mean(vec,axis=0)
            return vec

        # ft_vecs = list(map(tokens2vec, X))
        ft_vecs = [tokens2vec(t) for t in X]
        ft_vecs = np.array(ft_vecs)
        return ft_vecs

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y='ignored'):
        self.fit(X)
        return self.transform(X)





