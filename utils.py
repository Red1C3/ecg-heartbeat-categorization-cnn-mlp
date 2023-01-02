import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.regularizers import l2,l1,l1_l2
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.interpolate import interp1d
import seaborn as sns

import math

def archi_cnn(model,binary):
    model.add(layers.Conv1D(64,7,activation='relu',input_shape=(187,1),name="Conv1D-64/7"))
    model.add(layers.SpatialDropout1D(0.4,name="SD1D-0.4"))
    model.add(layers.BatchNormalization(name="BN-1"))
    model.add(layers.MaxPool1D(7,name="MP1D-7"))
    model.add(layers.Conv1D(32,5,activation='relu',name="Conv1D-32/5"))
    model.add(layers.BatchNormalization(name="BN-2"))
    model.add(layers.MaxPool1D(5,name="MP1D-5"))
    model.add(layers.GlobalMaxPool1D(name="GMP1D"))
    if binary:
      model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=l1_l2(0.0001),name="Output"))
    else:
      model.add(layers.Dense(5,activation='softmax',kernel_regularizer=l1_l2(0.0001),name="Output"))

def archi_mlp(model,binary):
    model.add(layers.Dense(64, activation='relu', input_shape=(187,),name="Dense-64"))
    model.add(layers.Dropout(0.25,name="Dropout-0.25"))
    model.add(layers.Dense(32,activation='relu',name="Dense-32"))
    model.add(layers.Dense(16,activation='relu',kernel_regularizer=l1_l2(0.0001),name="Dense-16"))
    if binary:
      model.add(layers.Dense(1, activation='sigmoid',kernel_regularizer=l1_l2(0.0001),name="Output"))
    else:
      model.add(layers.Dense(5, activation='softmax',kernel_regularizer=l1_l2(0.0001),name="Output"))

def import_set(oversampling,binary_set):
    if binary_set==False :
      dataframe = pd.read_csv('./mitbih_train.csv', header=None)
      dataframe_test = pd.read_csv('./mitbih_test.csv', header=None)

	  # Plot the original training and testing sets histograms
      fig, ax = plt.subplots(2,sharex=False)
      sns.countplot(x=187,data=dataframe,ax=ax[0])
      sns.countplot(x=187,data=dataframe_test,ax=ax[1])
      ax[0].set_title('Original Multi-Classification Training Set')
      ax[0].set_xlabel('categories')
      ax[1].set_title('Original Multi-Classification Testing Set')
      ax[1].set_xlabel('categories')
      fig.tight_layout()
      
      # Split and transform the testing set to numpy array
      y_test = dataframe_test[dataframe_test.columns[-1:]]
      x_test = dataframe_test[dataframe_test.columns[:-1]]
      y_test = y_test.to_numpy()
      x_test = x_test.to_numpy()
    else:
      normal=pd.read_csv('./ptbdb_normal.csv',header=None)
      abnormal=pd.read_csv('./ptbdb_abnormal.csv',header=None)
      
      #Stack normal and abnormal sets and shuffle them
      dataframe=pd.concat([normal,abnormal],axis=0)
      dataframe=dataframe.sample(frac=1)
      
      y = dataframe[dataframe.columns[-1:]]
      x = dataframe[dataframe.columns[:-1]]
      
      #Split the stacked set into training and testing
      x,x_test,y,y_test=train_test_split(x,y,test_size=0.2)
      
      #Merge the data with labels to apply oversampling if on
      dataframe=pd.concat([x,y],axis=1)

	  # Plot the original training and testing sets histograms
      fig, ax = plt.subplots(2,sharex=False)
      sns.countplot(x=187,data=dataframe,ax=ax[0])
      sns.countplot(x=187,data=pd.concat([x_test,y_test],axis=1),ax=ax[1])
      ax[0].set_title('Original Binary-Classification Training Set')
      ax[0].set_xlabel('normal/abnormal')
      ax[1].set_title('Original Binary-Classification Testing Set')
      ax[1].set_xlabel('normal/abnormal')
      fig.tight_layout()
      
      # Split and transform the testing set to numpy array
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
                    
          # Generate a new signal with more samples
          resampledRow=signal.resample(row[:-1],187*j)
          
          # Connect the newly generated signal linearly
          f=interp1d(np.linspace(0,187,num=187*j,endpoint=False),resampledRow)
          
          # Sample 187 samples out the generated signal after shifting it by 0.5
          row[:-1]=f(np.linspace(0.5,187,num=187,endpoint=False))
          return row
        
        def oversample(group):
        
        	# Calculate how many copies of this category you need to generate
            samples_ratio=(max_cat_size//len(group))-1
            
            group_copy=group.copy()
            for i in range(samples_ratio):
            	
            	# Increase the sampling samples so it generates different signals with each newly generated group
                global j;j=(i+2)
                
                newGroup=group_copy.copy()
                newGroup=newGroup.apply(resample_signal,axis=1).reset_index(drop=True)
                
                # Add the newly generated group
                group=pd.concat([group,newGroup],axis=0)
            return group
        dataframe=dataframe.groupby(187).apply(oversample).reset_index(drop=True)

	# Shuffle the data
    dataframe=dataframe.sample(frac=1)
    
    y = dataframe[dataframe.columns[-1:]]
    x = dataframe[dataframe.columns[:-1]]
    y = y.to_numpy()
    x = x.to_numpy()

    return x,x_test,y,y_test


def evaluate(name,results,x_test,y_test,is_multi,verbose):

  # Plots definations
  fig, ax = plt.subplots(2,sharex=True)
  fig.suptitle(name)
  ax[0].set_title("Accuracy")
  ax[0].plot(results.epoch, results.history['accuracy'],label='Train');
  ax[0].plot(results.epoch, results.history['val_accuracy'],label='Validation');
  ax[0].legend()
  ax[1].set_title("Loss")
  ax[1].plot(results.epoch, results.history['loss'],label='Train');
  ax[1].plot(results.epoch, results.history['val_loss'],label='Validation');
  ax[1].legend()
  
  # Predict test set results using the model
  y_pred=results.model.predict(x_test,verbose=verbose)
  
  if is_multi:
  	# Change data format from hot-shot encoding to the resultant category index (cuz softmax)
    y_pred=np.argmax(y_pred,axis=1)
  else:
  	# Round the results (cuz sigmoid)
    y_pred=np.round(y_pred)

  print(classification_report(y_test,y_pred))
  print(name+' evaluation:')
  evaluation=results.model.evaluate(x_test,y_test,verbose=verbose)
  print('loss = '+str(evaluation[0]))
  print('accuarcy = '+str(evaluation[1]))
