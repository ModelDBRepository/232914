
from __future__ import division
import numpy as np


def  f_instWeights(Synapses, pre, post, loc ):
    # instantaneous weights
    
    if type(loc)== np.ndarray:
        pre,post,loc = np.ix_(pre,post,loc)
    else:
        pre,post = np.ix_(pre,post)
        
    a = Synapses[pre,post,loc,:]
    
    instWeight = np.sum(a)
    
    return instWeight
