from sklearn.model_selection import train_test_split
from sklearn import impute,preprocessing
import pandas as pd
import numpy as np

class _Dataset:

    """
    Prepare Dataset for LSTM models
    """

    def __init__(self,df,timestep=10,forward=10):
        self.timestep=timestep
        self.forward=forward
        self.df=df.copy()

    def prepare(self,convert=True,prob='classification'):

        df_=self.df.copy()
        #df_=df_.dropna()

        Xtrain,Xtest=train_test_split(df_,test_size=0.3,shuffle=False)

        imputer=impute.SimpleImputer().fit(Xtrain)
        Xtrain=pd.DataFrame(imputer.transform(Xtrain),columns=df_.columns,index=Xtrain.index)
        Xtest=pd.DataFrame(imputer.transform(Xtest),columns=df_.columns,index=Xtest.index)

        scaler=preprocessing.StandardScaler().fit(Xtrain)
        Xtrain_scaled=pd.DataFrame(scaler.transform(Xtrain),columns=df_.columns,index=Xtrain.index)
        Xtest_scaled=pd.DataFrame(scaler.transform(Xtest),columns=df_.columns,index=Xtest.index)

        temp=Xtrain_scaled.append(Xtest_scaled).copy()
        final=pd.DataFrame(columns=temp.columns,index=temp.index)

        print('Preparing dataframes...')
        
        if prob=='classification':
            final['temp1']=''
            final['temp2']=pd.DataFrame(scaler.inverse_transform(temp),columns=temp.columns,index=temp.index).dret.values

            for col in (temp.columns):
                    for i in range(len(temp)):
                        final.loc[:,col][i]=(temp[col].values[i-self.timestep:i])
                        #final.loc[:,'temp1'][i]=(final['temp2'].values[i-self.forward:i])

            for i in range(len(temp)):
                final.loc[:,'temp1'][i]=(final['temp2'].values[i-self.forward:i])

            _max=max(self.forward,self.timestep)
            final=final.iloc[_max:]

            #Defining the target
            final['target']=final.temp1.shift(-self.forward)
            final=final.dropna()
            final['target']=final.target.apply(lambda x:1 if np.prod([i for i in map(lambda y:(1+y),x)])-1>0.025 else 0).values
            final=final.drop(['temp1','temp2'],axis=1)
        
        else:#Regression
            
            for col in (temp.columns):
                    for i in range(len(temp)):
                        final.loc[:,col][i]=(temp[col].values[i-self.timestep:i])

            _max=max(self.forward,self.timestep)
            final=final.iloc[_max:]

            #Defining the target
            final['target']=final.Close.shift(-self.forward)
            final=final.dropna()

        Xtrain,Xtest,ytrain,ytest=train_test_split(final.drop('target',axis=1),final.target,test_size=0.3,shuffle=False)

        if convert==False:
            return Xtrain,Xtest,ytrain,ytest

        else:
            print('\n')
            print('Converting dataframes to arrays...')
            
            if prob=='classification':
                X1=np.ndarray(shape=(Xtrain.shape[0],Xtrain.shape[1],self.timestep))
                X2=np.ndarray(shape=(Xtest.shape[0],Xtest.shape[1],self.timestep))

                y1=np.array(ytrain).reshape(-1,1)
                y2=np.array(ytest).reshape(-1,1)

                for i in (range(X1.shape[0])):
                    for j in range(X1.shape[1]):
                        for k in range(self.timestep):
                            X1[i,j,k]=Xtrain.iloc[i,j][k]

                for i in (range(X2.shape[0])):
                    for j in range(X2.shape[1]):
                        for k in range(self.timestep):
                            X2[i,j,k]=Xtest.iloc[i,j][k]
            
            else:#Regression
                
                X1=np.ndarray(shape=(Xtrain.shape[0],Xtrain.shape[1],self.timestep))
                X2=np.ndarray(shape=(Xtest.shape[0],Xtest.shape[1],self.timestep))

                y1=np.ndarray(shape=(ytrain.shape[0],self.forward))
                y2=np.ndarray(shape=(ytest.shape[0],self.forward))

                for i in (range(X1.shape[0])):
                    for j in range(X1.shape[1]):
                        for k in range(self.timestep):
                            X1[i,j,k]=Xtrain.iloc[i,j][k]
                            
                    for k in range(self.forward):
                        y1[i,k]=ytrain.iloc[i][k]

                for i in (range(X2.shape[0])):
                    for j in range(X2.shape[1]):
                        for k in range(self.timestep):
                            X2[i,j,k]=Xtest.iloc[i,j][k]
                            
                    for k in range(self.forward):
                        y2[i,k]=ytrain.iloc[i][k]
                                
            return X1,X2,y1,y2


