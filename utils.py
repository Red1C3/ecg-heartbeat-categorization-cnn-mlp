import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import tensorflow.keras.layers as layers
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.regularizers import l2,l1,l1_l2

import math

CATEGORIES_COUNT = 5


# Import the dataset, it can take a subset of it, and shuffle it
def importTrainingSet(shuffle: bool, oversampling=False, samples: int = None):
    dataframe = pd.read_csv('./mitbih_train.csv', header=None)

    y = dataframe[dataframe.columns[-1:]]
    x = dataframe[dataframe.columns[:-1]]

    if oversampling:
        oversampler = SMOTE()
        x, y = oversampler.fit_resample(x, y)

        # amplitude oversampling
        # max_cat_size=max(len(dataframe[dataframe[187]==0]),
        #                  len(dataframe[dataframe[187]==1]),
        #                  len(dataframe[dataframe[187]==2]),
        #                  len(dataframe[dataframe[187]==3]),
        #                  len(dataframe[dataframe[187]==4]))
        #
        # def oversample(group):
        #     samples_ratio=(max_cat_size//len(group))-1
        #     group_copy=group.copy()
        #     for i in range(samples_ratio):
        #         newGroup=group_copy.copy()
        #         newGroup[newGroup.columns[:-1]]+=i*0.00001
        #         group=pd.concat([group,newGroup],axis=0)
        #     return group
        # dataframe=dataframe.groupby(187).apply(oversample).reset_index(drop=True)
        # y = dataframe[dataframe.columns[-1:]]
        # x = dataframe[dataframe.columns[:-1]]

    if samples is not None:
        category_samples_count = math.ceil(samples / CATEGORIES_COUNT)

        def sampling_k_elements(group):
            if len(group) < category_samples_count:
                return group
            return group.sample(category_samples_count)

        df = pd.concat([x, y], axis=1)
        balanced = df.groupby(187).apply(sampling_k_elements).reset_index(drop=True)
        if shuffle:
            balanced = balanced.sample(frac=1).reset_index(drop=True)
        y = balanced[balanced.columns[-1:]]
        x = balanced[balanced.columns[:-1]]
    elif shuffle:
        dataframe = pd.concat([x, y], axis=1)
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
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
    model.add(layers.Dense(64, activation='relu', input_shape=(187,)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(16,activation='relu',kernel_regularizer=l1_l2(0.0001)))
    model.add(layers.Dense(5, activation='softmax',kernel_regularizer=l1_l2(0.0001)))

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
def importTrainingSet_nn(shuffle: bool, oversampling=False, samples: int = None):
    dataframe = pd.read_csv('./mitbih_train.csv', header=None)

    y = dataframe[dataframe.columns[-1:]]
    x = dataframe[dataframe.columns[:-1]]

    if oversampling:
        oversampler = SMOTE()
        x, y = oversampler.fit_resample(x, y)

        # amplitude oversampling
        # max_cat_size=max(len(dataframe[dataframe[187]==0]),
        #                  len(dataframe[dataframe[187]==1]),
        #                  len(dataframe[dataframe[187]==2]),
        #                  len(dataframe[dataframe[187]==3]),
        #                  len(dataframe[dataframe[187]==4]))
        #
        # def oversample(group):
        #     samples_ratio=(max_cat_size//len(group))-1
        #     group_copy=group.copy()
        #     for i in range(samples_ratio):
        #         newGroup=group_copy.copy()
        #         newGroup[newGroup.columns[:-1]]+=i*0.00001
        #         group=pd.concat([group,newGroup],axis=0)
        #     return group
        # dataframe=dataframe.groupby(187).apply(oversample).reset_index(drop=True)
        # y = dataframe[dataframe.columns[-1:]]
        # x = dataframe[dataframe.columns[:-1]]

    if samples is not None:
        category_samples_count = math.ceil(samples / CATEGORIES_COUNT)

        def sampling_k_elements(group):
            if len(group) < category_samples_count:
                return group
            return group.sample(category_samples_count)

        df = pd.concat([x, y], axis=1)
        balanced = df.groupby(187).apply(sampling_k_elements).reset_index(drop=True)
        if shuffle:
            balanced = balanced.sample(frac=1).reset_index(drop=True)
        y = balanced[balanced.columns[-1:]]
        x = balanced[balanced.columns[:-1]]
    elif shuffle:
        dataframe = pd.concat([x, y], axis=1)
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        y = dataframe[dataframe.columns[-1:]]
        x = dataframe[dataframe.columns[:-1]]

    y = y.to_numpy()
    x = x.to_numpy()

    return x, y