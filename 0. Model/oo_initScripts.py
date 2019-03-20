# Functions to reset a biophysical neuron or synapses to the initial values


from brian2 import *

#import sys
#sys.path.append('../0. Model')

#from oo_Parameters import *


def set_init_nrn(neuron,Theta_low):
      
    neuron.v = Theta_low 
    neuron.v_ca = Theta_low 
    neuron.v_ltd = Theta_low 
    neuron.v_ltp = Theta_low 
    neuron.s_trace = 0. 
    neuron.gs = 0. 
    neuron.gNMDA = 0. 

    neuron.m = 0
    neuron.h = 0
    neuron.m2 = 0
    neuron.h2 = 0
    neuron.m3 = 0
    neuron.h3 = 0
    
            
    neuron.n = 0
    neuron.n2 = 0
    neuron.n3 = 0
    neuron.n4 = 0
    neuron.l = 1
    neuron.l2 = 1
    neuron.l3 = 1
    
    
def set_init_syn(syn,init_weight):
    nr_syn = len(syn.wampa[:])
    syn.wampa[:] = init_weight*ones(nr_syn)
    syn.wnmda[:] = init_weight*ones(nr_syn)
    syn.g = 0
    syn.g_nmda = 0