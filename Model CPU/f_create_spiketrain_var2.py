
from __future__ import division
import numpy as np


def  f_create_spiketrain_var2(Rates, timeStep):
    #induce spikes: 
    #   induce Poisson-distributed spikes and regular spikes
    
    train = np.zeros(np.shape(Rates))
    
    # avoid dividing by 0
    Rates[Rates==0] = 0.0000001
    
    ### Poisson-distributed spikes ###
    r = np.divide(np.random.rand(np.size(Rates,axis=0),np.size(Rates,axis=1))*(1000/timeStep),Rates)
    train[r<1] = 1
    
    return train
    
    
