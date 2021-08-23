import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import utils
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

import pandas as pd
import numpy as np

from sklearn import metrics
import matplotlib.pyplot as plt

#LSTM class

class _LSTM:

    """
    A custom class for different LSTM model variants

    """

    def __init__(self,data,timestep=10,type=None):
        self.timestep=timestep
        self.type=type
        self.X1,self.X2,self.y1,self.y2=data


    def _fit(self,verbose=False,summary=False,epochs=100):

        x=layers.Input(shape=(self.X1.shape[1],self.timestep))

        if self.type==None or self.type=='vanilla lstm':
            y=layers.LSTM(50,activation='tanh',return_sequences=False,input_shape=(self.timestep,self.X1.shape[1]),name='VaLSTM')(x)
        elif self.type=='stacked lstm':
            y=layers.LSTM(50,activation='tanh',return_sequences=True,input_shape=(self.timestep,self.X1.shape[1]),name='LSTM1')(x)
            y=layers.LSTM(50,activation='tanh',name='LSTM2')(y)
        elif self.type=='bidirectional lstm':
            y=layers.Bidirectional(layers.LSTM(50,activation='tanh',input_shape=(self.timestep,self.X1.shape[1]),name='BiLSTM1'))(x)

        y=layers.Dense(200,activation='relu',name='Dense1')(y)
        y=layers.Dropout(0.2)(y)
        y=layers.BatchNormalization()(y)

        y=layers.Dense(100,activation='relu',name='Dense2')(y)
        y=layers.Dropout(0.2)(y)
        y=layers.BatchNormalization()(y)
        
        y=layers.Dense(1,activation='sigmoid',name='Output')(y)

        self.model=Model(inputs=x,outputs=y)
        
        self.model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(learning_rate=0.001),metrics=['AUC','accuracy'])
        
        if summary==True:self.model.summary()

        print('\n')
        print('Fitting model...')
        self.model.fit(self.X1,self.y1,epochs=epochs,verbose=0,shuffle=False,validation_split=0.25)

        K.clear_session()
        
    def _fit_regression(self,verbose=False,summary=False,forward=5,epochs=100):

        x=layers.Input(shape=(self.X1.shape[1],self.timestep))

        if self.type==None or self.type=='vanilla lstm':
            y=layers.LSTM(50,activation='tanh',return_sequences=False,input_shape=(self.timestep,self.X1.shape[1]),name='VaLSTM')(x)
        elif self.type=='stacked lstm':
            y=layers.LSTM(50,activation='tanh',return_sequences=True,input_shape=(self.timestep,self.X1.shape[1]),name='LSTM1')(x)
            y=layers.LSTM(50,activation='tanh',name='LSTM2')(y)
        elif self.type=='bidirectional lstm':
            y=layers.Bidirectional(layers.LSTM(50,activation='tanh',input_shape=(self.timestep,self.X1.shape[1]),name='BiLSTM1'))(x)

        y=layers.Dense(200,activation='relu',name='Dense1')(y)
        y=layers.Dropout(0.2)(y)
        y=layers.BatchNormalization()(y)

        y=layers.Dense(100,activation='relu',name='Dense2')(y)
        y=layers.Dropout(0.2)(y)
        y=layers.BatchNormalization()(y)
        
        y=layers.Dense(forward,name='Output')(y)

        self.model=Model(inputs=x,outputs=y)
        
        self.model.compile(loss='mse',optimizer=optimizers.Adam(learning_rate=0.001),metrics=['mean_squared_error'])
        
        if summary==True:self.model.summary()

        print('\n')
        print('Fitting model...')
        self.model.fit(self.X1,self.y1,epochs=epochs,verbose=0,shuffle=False,validation_split=0.25)

        K.clear_session()
        
    def _predict(self,X):
        preds=self.model.predict(X)
        return preds

    def _metrics(self,act,pred):

        self.fpr,self.tpr,self.th=metrics.roc_curve(act,pred)
        self.auc=metrics.auc(self.fpr,self.tpr)
        self.accuracy=metrics.accuracy_score(act,np.round(pred))

        return self.auc,self.accuracy
    
    def _plot_model(self):
        plot_model(self.model,show_shapes=True,show_layer_names=True)        

    def _plots(self,ticker,act,pred,figsize=(20,5)):
        fig,ax=plt.subplots(1,2,figsize=figsize)
        ax[0].plot(self.fpr,self.tpr)
        ax[0].set_xlabel('FPR')
        ax[0].set_ylabel('TPR')
        ax[0].set_title(ticker+': ROC Curve')
        auc,acc=self._metrics(act,pred)
        ax[0].annotate('AUC: '+ str(np.round(auc,4)),xy=(0.85,0.1))

        ax[0].plot(np.linspace(0,1),np.linspace(0,1))

        cm=metrics.confusion_matrix(act,np.round(pred))
        #print(metrics.confusion_matrix(act,np.round(pred)))
        im=ax[1].imshow(cm,cmap='viridis')
        ax[1].set_xlabel("Predicted labels")
        ax[1].set_ylabel("True labels")
        ax[1].set_xticks([],[])
        ax[1].set_yticks([],[])
        ax[1].set_title(ticker+': Confusion matrix')

        fig.colorbar(im,ax=ax[1],orientation='vertical')
        
        plt.show()
    
    def _reg_preds(self):
        
        #Train set
        t1=self.y1
        t2=self.model.predict(self.X1)
        
        #Vaid set
        t3=self.y2
        t4=self.model.predict(self.X2)
        
        act=list((t1)[0])
        pred=list(t2[0])
        
        act_=list((t3)[0])
        pred_=list(t4[0])

        for i in range(1,len(self.y1)):
            act.append(t1[i][-1])
            pred.append(t2[i][-1])
            
        for i in range(1,len(self.y2)):
            act_.append(t3[i][-1])
            pred_.append(t4[i][-1])
        
        fig,ax=plt.subplots(1,2,figsize=(25,6))
        
        ax[0].plot(act,label='Actual train set')
        ax[0].plot(pred,label='Predictions',color='darkgreen')
        ax[0].set_title('Train set (Normalized prices), '+'RMSE: '+str(np.round(np.sqrt(metrics.mean_squared_error(act,pred)),4)))
        ax[0].legend()
        
        ax[1].plot(act_,label='Actual valid set',color='orange')
        ax[1].plot(pred_,label='Predictions',color='darkgreen')
        ax[1].set_title('Valid set (Normalized prices), '+'RMSE: '+str(np.round(np.sqrt(metrics.mean_squared_error(act_,pred_)),4)))
        ax[1].legend()

        
#Function to calculate Metrics for classification
def check_metrics(ticker,data,lstm_model):
    #print('\nStock: ',ticker)
    act=data[3]
    pred=lstm_model._predict(X=data[1])
    auc,acc=lstm_model._metrics(act,pred)
    lstm_model._plots(ticker,act,pred)
