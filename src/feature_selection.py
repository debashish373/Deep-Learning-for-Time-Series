from _KMeans import _KMeans
from _SOM import _SOM

import numpy as np

def select_features(df,method='SOM'):
    
    if method=='SOM':
                    
        cols=df.columns.tolist()
        
        sm=_SOM(df.copy(),units=100)
    
        features1=sm._features(cols=cols[:7]) #Stock price/Volume
        features2=sm._features(cols=cols[7:63]) #Market data
        features3=sm._features(cols=cols[63:67]) #Bond specific data (spreads/yields)
        features4=sm._features(cols=cols[67:114]) #Fin ratios
        features5=sm._features(cols=cols[114:]) #Technical indicators from TALIB
    
        features_som=features1+features2+features3+features4+features5
    
        if 'dret' not in features_som:
            features_som=features_som+['dret']
            
        return features_som
    
    elif method=='KMC':
        def get_ks(threshold):
    
            t=[]
            
            cols=df.columns.tolist()
            
            f1=cols[:7] #Stock price/Volume
            f2=cols[7:63] #Market data
            f3=cols[63:67] #Bond specific data (spreads/yields)
            f4=cols[67:114] #Fin ratios
            f5=cols[114:] #Technical indicators from TALIB
            
            for i,f in enumerate([f1,f2,f3,f4,f5]):
                
                distortions=[]
                inertias=[]
                
                deltas=[]
                
                Ks=np.arange(1,len(f)) if len(f)<30 else np.arange(1,30)
        
                for k in (Ks):
        
                    km=_KMeans(df[f].copy(),k)
                    km._fit()
                    distortions.append(km._distortion)
                    inertias.append(km._inertia)
                
                deltas=list(-np.diff(distortions))
                #print(deltas)
                try:
                    t.append(deltas.index(max([x for x in deltas if x<threshold]))) #selecting the K for which the decrease is below the specified threshold 
                except:
                    t.append(Ks[-1])
            return t
            
        k=get_ks(0.01)#No of clusters to be chosen for each feature set
        
        cols=df.columns.tolist()
        
        features1=_KMeans(df[cols[:7]].copy(),k[0])._features #Stock price/Volume
        features2=_KMeans(df[cols[7:63]].copy(),k[1])._features #Market data
        features3=_KMeans(df[cols[63:67]].copy(),k[2])._features #Bond specific data (spreads/yields)
        features4=_KMeans(df[cols[67:114]].copy(),k[3])._features #Fin ratios
        features5=_KMeans(df[cols[114:]].copy(),k[4])._features #Technical indicators from TALIB
    
        features_km=features1+features2+features3+features4+features5
    
        if 'dret' not in features_km:
            features_km=features_km+['dret']
                
        return features_km
  