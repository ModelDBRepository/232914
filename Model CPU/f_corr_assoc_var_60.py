
from __future__ import division
import numpy as np


def  f_corr_assoc_var_60(NrON,NrIn_perFeat,NrFeat,prob_var,inp_time,timeStep,buffertime,StimFreq):
  
    

    nr_nrn = NrFeat*NrIn_perFeat
    itime = int(inp_time/timeStep) # nr of timesteps for active input
    btime = int(buffertime/timeStep) # nr of timesteps between two active inputs
    
    prob_v = np.random.rand(1)
    
    if prob_v<prob_var[0]:
        rvar = np.expand_dims(np.array([1,1,1,1,0,0]),1)
        rvar = np.kron(StimFreq*(rvar),np.ones((NrIn_perFeat,1)))
        r_matrix = np.append(np.tile(rvar,[1,itime]), np.zeros((nr_nrn,btime)),axis=1)     
        ftr=1
    elif prob_v<prob_var[0]+prob_var[1]:
        rvar = np.expand_dims(np.array([0,0,1,1,1,1]),1)
        rvar = np.kron(StimFreq*(rvar),np.ones((NrIn_perFeat,1)))
        r_matrix = np.append(np.tile(rvar,[1,itime]), np.zeros((nr_nrn,btime)),axis=1)  
        ftr=2
    else:
        rvar = np.expand_dims(np.array([0,0,0,0,0,0]),1)
        rvar = np.kron(StimFreq*(rvar),np.ones((NrIn_perFeat,1)))
        r_matrix = np.append(np.tile(rvar,[1,itime]), np.zeros((nr_nrn,btime)),axis=1) 
        ftr=3
    
       
    r_matrix = np.append(r_matrix , np.zeros((NrON,np.size(r_matrix,axis=1))), axis=0)
    
    return r_matrix,ftr
    
