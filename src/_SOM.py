import itertools
from sklearn import preprocessing
from minisom import MiniSom
import numpy as np
import pandas as pd

#Custom SOM class based on the Minisom library 

class _SOM:
    def __init__(self,df,units):
        self.df=df.copy()
        self.X=preprocessing.scale(df.copy())
        self.units=units

        som=MiniSom(self.units,self.units,len(self.X[0]),neighborhood_function='gaussian',random_seed=2021)
        som.random_weights_init(self.X)
        som.train_random(self.X,100,verbose=True)
        
        self.W=som.get_weights()

    @property
    def _get_weights(self):
        return self.W
    
    
    def _features(self,cols,threshold=0.5):
        
        """
        Features are selected based on the similarity mapping of the weight vectors for each feature.
        The norm of the differences between two weight vectors is taken as a proxy for similarity.
        A threshold of 0.5 for the norm difference is taken as default, below which one feature out of the two is eliminated.
    
        """

        combs=[c for c in itertools.combinations(cols,2)]
        features=cols.copy()

        norms=np.array([np.linalg.norm(self.W[:,:,cols.index(c[0])]-self.W[:,:,cols.index(c[1])]) for c in combs])
        _max=max(norms)
        _min=min(norms)
        norms=list([(n-_min)/(_max-_min) for n in (norms)])

        for i,c in enumerate(combs):

            if norms[i]<threshold and c[1] in features:
                features.remove(c[1])

        return features
    
    