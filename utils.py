import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import tensorflow.keras.layers as layers

import math

CATEGORIES_COUNT=5

#Import the dataset, it can take a subset of it, and shuffle it
def importTrainingSet(shuffle:bool,samples:int=None):
    dataframe=pd.read_csv('./mitbih_train.csv',header=None)

    #Last column is the category column, a scaler value from 0 to 4
    y=dataframe[dataframe.columns[-1:]]
    x=dataframe[dataframe.columns[:-1]]

    # Balances the dataset by oversampling it (it seems like it uses interpolation)
    # https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    oversampler=SMOTE()
    x,y=oversampler.fit_resample(x,y)

    if samples!=None:
        category_samples_count=math.ceil(samples/CATEGORIES_COUNT)
        def sampling_k_elements(group):
            if len(group) < category_samples_count:
                return group
            return group.sample(category_samples_count)

        df=pd.concat([x,y],axis=1)
        balanced = df.groupby(187).apply(sampling_k_elements).reset_index(drop=True)
        if shuffle:
            balanced=balanced.sample(frac=1).reset_index(drop=True)
        y=balanced[balanced.columns[-1:]]
        x=balanced[balanced.columns[:-1]]
    elif shuffle:
        dataframe=pd.concat([x,y],axis=1)
        dataframe=dataframe.sample(frac=1).reset_index(drop=True)
        y=dataframe[dataframe.columns[-1:]]
        x=dataframe[dataframe.columns[:-1]]

    y=y.to_numpy()
    x=x.to_numpy()

    #Add data dimension, doesn't actually change the data but that's how the input is expected
    #like it could have multiple values per timestamp
    x=x.reshape([x.shape[0],187,1])

    return x,y

def evaluete(model:tf.keras.Model,batch_size=500,verbose=True):
    dataframe_test=pd.read_csv('./mitbih_test.csv', header=None)
    y_test = dataframe_test[dataframe_test.columns[-1:]]
    x_test = dataframe_test[dataframe_test.columns[:-1]]
    y_test = y_test.to_numpy()

    x_test = x_test.to_numpy()

    x_test = x_test.reshape([x_test.shape[0], 187, 1])

    return model.evaluate(x_test,y_test,batch_size,verbose)

def archi1(model):
    model.add(layers.Conv1D(7, 7, activation='relu', input_shape=(187,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(5,5,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(3,3,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(5,activation='softmax'))