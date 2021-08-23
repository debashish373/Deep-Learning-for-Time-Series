import itertools
from sklearn import preprocessing,cluster
from minisom import MiniSom
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

#Custom KMeans class based on the sklearn library

class _KMeans:
    def __init__(self,df,k):
        self.df=df
        self.X=preprocessing.scale(df.T.fillna(999))
        self.k=k
        self.cols=df.columns.tolist()

    def _fit(self):
        self.kmeans=cluster.KMeans(n_clusters=self.k,init='random',max_iter=100,random_state=21)
        self.kmeans.fit(self.X)

    @property
    def _inertia(self):
        return self.kmeans.inertia_

    @property
    def _distortion(self):
        return sum(np.min(cdist(self.X,self.kmeans.cluster_centers_,metric='euclidean'),axis=1))/self.X.shape[0]

    def _elbow_plot(self,distortions,inertias,Ks,figsize=(25,3)):
        fig,ax=plt.subplots(figsize=figsize)
        ax.plot(Ks,distortions,color='brown',marker='X',label='distortions')
        ax_=ax.twinx()
        ax_.plot(Ks,inertias,color='k',marker='o',label='inertias')
        ax.set_xlabel('K')
        ax.legend(loc=(0.9,0.9))
        ax_.legend(loc=(0.9,0.8))
        plt.show()
        
        return fig

    @property
    def _features(self):
        
        """
        Features are selected based on the distance of a particular feature to the cluster centroid.
        The one with the least distance is chosen from a particular cluster.
        
        """
        self._fit()
        feat=pd.DataFrame({'feature':self.df.T.index.tolist()})
        feat['cluster']=self.kmeans.labels_
        feat['dist']=np.min(cdist(self.X,self.kmeans.cluster_centers_,metric='euclidean'),axis=1)
        feat=feat.sort_values(by='dist',ascending=False).drop_duplicates(subset='cluster',keep='last')
        return feat.feature.tolist()

    
