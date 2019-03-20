
from __future__ import division
import numpy as np


def  f_instRate( spikes, length, spikes_memory, NrnPerFeat ):
    #Instantaneous firing rate
    
    spk = np.zeros(int(np.size(spikes,axis=0)/NrnPerFeat))
    for nn in np.arange(np.size(spk,axis=0)):
        spk[nn] = np.sum(spikes[nn*NrnPerFeat:(nn+1)*NrnPerFeat],axis=0)
   
    
    spikes_memory[:,0:-1] = spikes_memory[:,1:]
    spikes_memory[:,-1] = spk
    
    inst_rate = np.sum(spikes_memory,axis=1)/(0.001*length*NrnPerFeat)

    return inst_rate,spikes_memory
