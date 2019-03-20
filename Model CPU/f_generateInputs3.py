import numpy as np

import sys
sys.path.append('../')

from f_create_spiketrain_var2 import f_create_spiketrain_var2
from f_corr_assoc_var6 import f_corr_assoc_var6

def f_generateInputs3(itervr,buffertime,inp_time,rate,NrON,NrIn_perFeat,NrFeatures,timeStep,FeatProbabs):
    ispikes = []
    
    for ww in np.arange(itervr):
        
        s_rates,ftr = f_corr_assoc_var6(NrON,NrIn_perFeat,NrFeatures,FeatProbabs,inp_time,timeStep,buffertime,rate) 
        spiketrain = f_create_spiketrain_var2(s_rates, timeStep) 
        
        if len(ispikes)>0:
            ispikes = np.append(ispikes,spiketrain,axis=1)  
        else:
            ispikes = spiketrain 
    
    return ispikes,ftr