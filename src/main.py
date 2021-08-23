from features import stocks
from _LSTM import _LSTM,check_metrics
from _Dataset import _Dataset
import feature_selection
from sklearn import metrics
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import argparse

def run(ticker,timestep=10,forward=5,epochs=5,fs='SOM'):
    
    features=feature_selection.select_features(stocks[ticker],method=fs)
    
    print('\nFeatures selected:\n')
    print(features)
    print('\n')
    
    data=_Dataset(stocks[ticker][features],timestep=timestep,forward=forward).prepare(convert=True)
    
    lstm_model=_LSTM(data,timestep=timestep,type='vanilla lstm')
    lstm_model._fit(epochs=epochs)
    
    check_metrics(ticker,data,lstm_model)
    print('\nAUC for ',forward,'-Day Forward Return prediction:',np.round(metrics.auc(lstm_model.fpr,lstm_model.tpr),4))
    
    print('\nDone!')
    
# 5 tickers to choose from ['BATS.L','DTE.DE','RNO.PA','SIE.DE','TTE.PA']
if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--ticker",type=str)
    parser.add_argument("--timestep",type=int)
    parser.add_argument("--forward",type=int)
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--fs",type=str)
    
    args=parser.parse_args()
    
    run(
        ticker=args.ticker,
        timestep=args.timestep,
        forward=args.forward,
        epochs=args.epochs,
        fs=args.fs,
        )
    
    

