
from __future__ import division
import numpy as np


def  f_Inpt_x(Pre_spikes,input_x,Neuron_par,timeStep):
#Calculate evolution of spike trace (inputs)

    #######################
    # Parameters
    #######################
    x_tau = Neuron_par[8] #timeconstant for spiketrace
    x_reset = Neuron_par[9]/x_tau # reset value for spiketrace    
    
    ##############
    # x_trace
    ##############         
    
    ### x (spike trace) ###
    input_x = input_x - (timeStep/x_tau)*input_x + Pre_spikes*x_reset
    
    return input_x
    
    
