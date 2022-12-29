import pandas as pd
from imblearn.over_sampling import SMOTE
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
        dataframe=df.sample(frac=1).reset_index(drop=True)
        y=dataframe[dataframe.columns[-1:]]
        x=dataframe[dataframe.columns[:-1]]

    y=y.to_numpy()
    x=x.to_numpy()

    #Add data dimension, doesn't actually change the data but that's how the input is expected
    #like it could have multiple values per timestamp
    x=x.reshape([x.shape[0],187,1])

    return x,y