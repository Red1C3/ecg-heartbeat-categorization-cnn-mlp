import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import tensorflow.keras.layers as layers
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.regularizers import l2,l1,l1_l2
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.interpolate import interp1d

import math

CATEGORIES_COUNT = 5


# Import the dataset, it can take a subset of it, and shuffle it
def importTrainingSet(oversampling=False,binary_set=False):
    if binary_set==False :
      dataframe = pd.read_csv('./mitbih_train.csv', header=None)
    else:
      normal=pd.read_csv('./ptbdb_normal.csv',header=None)
      abnormal=pd.read_csv('./ptbdb_abnormal.csv',header=None)
      dataframe=pd.concat([normal,abnormal],axis=0)

    if oversampling:
        #oversampler = SMOTE()
        #x, y = oversampler.fit_resample(x, y)

        max_cat_size=max(len(dataframe[dataframe[187]==0]),
                         len(dataframe[dataframe[187]==1]),
                         len(dataframe[dataframe[187]==2]),
                         len(dataframe[dataframe[187]==3]),
                         len(dataframe[dataframe[187]==4]))
        j=2
        def resample_signal(row):
          resampledRow=signal.resample(row[:-1],187*j)
          f=interp1d(np.linspace(0,187,num=187*j,endpoint=False),resampledRow)
          row[:-1]=f(np.linspace(0.5,187,num=187,endpoint=False))
          return row
        
        def oversample(group):
            samples_ratio=(max_cat_size//len(group))-1
            group_copy=group.copy()
            for i in range(samples_ratio):
                global j;j=(i+2)
                newGroup=group_copy.copy()
                newGroup=newGroup.apply(resample_signal,axis=1).reset_index(drop=True)
                group=pd.concat([group,newGroup],axis=0)
            return group
        
        dataframe=dataframe.groupby(187).apply(oversample).reset_index(drop=True)
       

    dataframe=dataframe.sample(frac=1)
    y = dataframe[dataframe.columns[-1:]]
    x = dataframe[dataframe.columns[:-1]]
    y = y.to_numpy()
    x = x.to_numpy()

    # Add data dimension, doesn't actually change the data but that's how the input is expected
    # like it could have multiple values per timestamp
    x = x.reshape([x.shape[0], 187, 1])

    return x, y


def predict(model: tf.keras.Model):
    dataframe_test = pd.read_csv('./mitbih_test.csv', header=None)
    y_test = dataframe_test[dataframe_test.columns[-1:]]
    x_test = dataframe_test[dataframe_test.columns[:-1]]
    y_test = y_test.to_numpy()

    x_test = x_test.to_numpy()

    x_test = x_test.reshape([x_test.shape[0], 187, 1])

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    return classification_report(y_test, y_pred)


def evaluate(model: tf.keras.Model, batch_size=500, verbose=True):
    dataframe_test = pd.read_csv('./mitbih_test.csv', header=None)
    y_test = dataframe_test[dataframe_test.columns[-1:]]
    x_test = dataframe_test[dataframe_test.columns[:-1]]
    y_test = y_test.to_numpy()

    x_test = x_test.to_numpy()

    x_test = x_test.reshape([x_test.shape[0], 187, 1])

    return model.evaluate(x_test, y_test, batch_size, verbose)


def archi1(model):
    model.add(layers.Conv1D(7, 7, activation='relu', input_shape=(187, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(5, 5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(3, 3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))


# arch1 did better ig on weights initilizing
def archi1_diff_init(model):
    model.add(layers.Conv1D(7, 7, activation='relu', input_shape=(187, 1), kernel_initializer='he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(5, 5, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(3, 3, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(5, activation='softmax', kernel_initializer='he_uniform'))


def archi1_dropout1(model):
    model.add(layers.Conv1D(7, 7, activation='relu', input_shape=(187, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(5, 5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(3, 3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))


def archi2(model):
    model.add(layers.Conv1D(64, 7, activation='relu', input_shape=(187, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(5))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(5, activation='softmax'))


def archi2_3layered(model):
    model.add(layers.Conv1D(32, 7, activation='relu', input_shape=(187, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(5))
    model.add(layers.Conv1D(16, 5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(16, 5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(5, activation='softmax'))

#97 on training / 96 on testing, best one so far ig
def archi3(model):
    model.add(layers.Conv1D(32, 7, activation='relu', input_shape=(187, 1),padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(16, 5, activation='relu',padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(7))
    model.add(layers.Conv1D(16, 5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(5))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(5, activation='softmax'))

def archi4(model):
    model.add(layers.Conv1D(64,7,activation='relu',input_shape=(187,1),name="Conv1D-64/7"))
    model.add(layers.SpatialDropout1D(0.4,name="SD1D-0.4"))
    model.add(layers.BatchNormalization(name="BN-1"))
    model.add(layers.MaxPool1D(7,name="MP1D-7"))
    model.add(layers.Conv1D(32,5,activation='relu',name="Conv1D-32/5"))
    model.add(layers.BatchNormalization(name="BN-2"))
    model.add(layers.MaxPool1D(5,name="MP1D-5"))
    model.add(layers.GlobalMaxPool1D(name="GMP1D"))
    model.add(layers.Dense(5,activation='softmax',kernel_regularizer=l1_l2(0.0001),name="Output"))

def archi4_bin(model):
    model = tf.keras.models.Sequential(name="CNN Binary")
    model.add(layers.Conv1D(64,7,activation='relu',input_shape=(187,1),name="Conv1D-64/7"))
    model.add(layers.SpatialDropout1D(0.4,name="SD1D-0.4"))
    model.add(layers.BatchNormalization(name="BN-1"))
    model.add(layers.MaxPool1D(7,name="MP1D-7"))
    model.add(layers.Conv1D(32,5,activation='relu',name="Conv1D-32/5"))
    model.add(layers.BatchNormalization(name="BN-2"))
    model.add(layers.MaxPool1D(5,name="MP1D-5"))
    model.add(layers.GlobalMaxPool1D(name="GMP1D"))
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=l1_l2(0.0001),name="Output"))

#93 on training / 92 on testing
def archi1_nn(model):
    model.add(layers.Dense(10, activation='relu', input_shape=(187,)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

def archi2_nn(model):
    model.add(layers.Dense(64, activation='relu', input_shape=(187,)))
    model.add(layers.Dense(50,activation='relu'))
    model.add(layers.Dense(42,activation='relu'))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

def archi3_nn(model):
    model.add(layers.Dense(64, activation='relu', input_shape=(187,)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

# for 10 epochs it did 95 / 95
def archi4_nn(model):
    model.add(layers.Dense(64, activation='relu', input_shape=(187,)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(16,activation='relu',kernel_regularizer=l2(0.0001)))
    model.add(layers.Dense(5, activation='softmax',kernel_regularizer=l2(0.0001)))
#getting 96/96 
def archi5_nn(model):
    model.add(layers.Dense(64, activation='relu', input_shape=(187,),name="Dense-64"))
    model.add(layers.Dropout(0.25,name="Dropout-0.25"))
    model.add(layers.Dense(32,activation='relu',name="Dense-32"))
    model.add(layers.Dense(16,activation='relu',kernel_regularizer=l1_l2(0.0001),name="Dense-16"))
    model.add(layers.Dense(5, activation='softmax',kernel_regularizer=l1_l2(0.0001),name="Output"))

def archi5_nn_bin(model):
    model.add(layers.Dense(64, activation='relu', input_shape=(187,),name="Dense-64"))
    model.add(layers.Dropout(0.25,name="Dropout-0.25"))
    model.add(layers.Dense(32,activation='relu',name="Dense-32"))
    model.add(layers.Dense(16,activation='relu',kernel_regularizer=l1_l2(0.0001),name="Dense-16"))
    model.add(layers.Dense(1, activation='sigmoid',kernel_regularizer=l1_l2(0.0001),name="Output"))

def archi6_nn(model):
    model.add(layers.Dense(64, activation='relu', input_shape=(187,)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(20,activation='relu',kernel_regularizer=l1_l2(0.0001)))
    model.add(layers.Dense(5, activation='softmax',kernel_regularizer=l1_l2(0.0001)))

def archi1_lstm(model):
    model.add(layers.LSTM(16, input_shape=(187,1)))
    model.add(layers.Dense(8,activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

def predict_nn(model: tf.keras.Model):
    dataframe_test = pd.read_csv('./mitbih_test.csv', header=None)
    y_test = dataframe_test[dataframe_test.columns[-1:]]
    x_test = dataframe_test[dataframe_test.columns[:-1]]
    y_test = y_test.to_numpy()

    x_test = x_test.to_numpy()

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    return classification_report(y_test, y_pred)


def evaluate_nn(model: tf.keras.Model, batch_size=500, verbose=True):
    dataframe_test = pd.read_csv('./mitbih_test.csv', header=None)
    y_test = dataframe_test[dataframe_test.columns[-1:]]
    x_test = dataframe_test[dataframe_test.columns[:-1]]
    y_test = y_test.to_numpy()

    x_test = x_test.to_numpy()

    return model.evaluate(x_test, y_test, batch_size, verbose)


# Import the dataset, it can take a subset of it, and shuffle it
def importTrainingSet_nn(oversampling=False,binary_set=False):
    if binary_set==False :
      dataframe = pd.read_csv('./mitbih_train.csv', header=None)
    else:
      normal=pd.read_csv('./ptbdb_normal.csv',header=None)
      abnormal=pd.read_csv('./ptbdb_abnormal.csv',header=None)
      dataframe=pd.concat([normal,abnormal],axis=0)

    if oversampling:
        #oversampler = SMOTE()
        #x, y = oversampler.fit_resample(x, y)
        
        max_cat_size=max(len(dataframe[dataframe[187]==0]),
                         len(dataframe[dataframe[187]==1]),
                         len(dataframe[dataframe[187]==2]),
                         len(dataframe[dataframe[187]==3]),
                         len(dataframe[dataframe[187]==4]))
        j=2
        def resample_signal(row):
          resampledRow=signal.resample(row[:-1],187*j)
          f=interp1d(np.linspace(0,187,num=187*j,endpoint=False),resampledRow)
          row[:-1]=f(np.linspace(0.5,187,num=187,endpoint=False))
          return row
        
        def oversample(group):
            samples_ratio=(max_cat_size//len(group))-1
            group_copy=group.copy()
            for i in range(samples_ratio):
                global j;j=(i+2)
                newGroup=group_copy.copy()
                newGroup=newGroup.apply(resample_signal,axis=1).reset_index(drop=True)
                group=pd.concat([group,newGroup],axis=0)
            return group
        dataframe=dataframe.groupby(187).apply(oversample).reset_index(drop=True)

    dataframe=dataframe.sample(frac=1)
    y = dataframe[dataframe.columns[-1:]]
    x = dataframe[dataframe.columns[:-1]]
    y = y.to_numpy()
    x = x.to_numpy()

    return x, y

def import_set(oversampling,binary_set):
    if binary_set==False :
      dataframe = pd.read_csv('./mitbih_train.csv', header=None)
      dataframe_test = pd.read_csv('./mitbih_test.csv', header=None)
      y_test = dataframe_test[dataframe_test.columns[-1:]]
      x_test = dataframe_test[dataframe_test.columns[:-1]]
      y_test = y_test.to_numpy()
      x_test = x_test.to_numpy()
    else:
      normal=pd.read_csv('./ptbdb_normal.csv',header=None)
      abnormal=pd.read_csv('./ptbdb_abnormal.csv',header=None)
      dataframe=pd.concat([normal,abnormal],axis=0)
      dataframe=dataframe.sample(frac=1)
      x,x_test,y,y_test=train_test_split(x,y,test_size=0.2)
      dataframe=pd.concat([x,y],axis=1)
      x_test=x_test.to_numpy()
      y_test=y_test.to_numpy()


    if oversampling:        
        max_cat_size=max(len(dataframe[dataframe[187]==0]),
                         len(dataframe[dataframe[187]==1]),
                         len(dataframe[dataframe[187]==2]),
                         len(dataframe[dataframe[187]==3]),
                         len(dataframe[dataframe[187]==4]))
        j=2
        def resample_signal(row):
          resampledRow=signal.resample(row[:-1],187*j)
          f=interp1d(np.linspace(0,187,num=187*j,endpoint=False),resampledRow)
          row[:-1]=f(np.linspace(0.5,187,num=187,endpoint=False))
          return row
        
        def oversample(group):
            samples_ratio=(max_cat_size//len(group))-1
            group_copy=group.copy()
            for i in range(samples_ratio):
                global j;j=(i+2)
                newGroup=group_copy.copy()
                newGroup=newGroup.apply(resample_signal,axis=1).reset_index(drop=True)
                group=pd.concat([group,newGroup],axis=0)
            return group
        dataframe=dataframe.groupby(187).apply(oversample).reset_index(drop=True)

    dataframe=dataframe.sample(frac=1)
    y = dataframe[dataframe.columns[-1:]]
    x = dataframe[dataframe.columns[:-1]]
    y = y.to_numpy()
    x = x.to_numpy()

    return x,x_test,y,y_test
