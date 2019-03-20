
from __future__ import division
import numpy as np


def  f_neurons( NrON , Neuron_par,timeStep):
# addNeurons: Initialize neuron matrix with NrON = nr of Neurons

    ''' 
    Each row is a new neuron

    %  S_potential variables are ordered in following columns:
    %                   0/ postsynaptic potential - soma
    
    %  D_potential variables ordered in following columns: 
    %                   0/ local postsynaptic potential - proximal
    %                   1/ local postsynaptic potential - distal
    %  3rd dimension = number of dendrites
    
    %  Nrn_u_delay
    %                   0/ local membrane potential at time (t-2) - proximal
    %                   1/ local membrane potential at time (t-2) - distal
    %  3rd dimension = number of dendrites
    
    %  Nrn_u_lowpass variables are ordered in following columns:
    %                   0/ v1 low-pass filtered local postsynaptic potential - proximal
    %                   1/ v1 low-pass filtered local postsynaptic potential - distal
    %                   2/ v2 low-pass filtered local postsynaptic potential - proximal
    %                   3/ v2 low-pass filtered local postsynaptic potential - distal   
    %  3rd dimension = number of dendrites
    
    %  Nrn_traces variables are ordered in following columns:
    %                   0/ homeostatic filtered version of the somatic potential
    %                   1/ x (spike trace)
    %                   2/ V_T   (threshold evolution)
    %                   3/ spikeYN (1or0 = threshold reached or not)
    %                   4/ I_ext (external current to soma)
    %                   5/ Noise Current
    
    
    %  Nrn_conductance variables are ordered in following columns:
    %                   0/ g_ampa (proximal ampa conductance)
    %                   1/ g_nmda (distal nmda conductance)
    %  3rd dimension = number of dendrites 
    '''
    
    Nr_D = Neuron_par[4] # number of dendrites
    E_L = Neuron_par[0] # resting potential 
    V_T_rest = Neuron_par[12]*np.ones((NrON,1)) # rest value V_T
    delay_time = Neuron_par[21]+Neuron_par[22] #spike latency+spike width [ms]
    
    #---------------------------
    # Initialize matrices
    #---------------------------
    
    S_potential = E_L*np.ones(NrON) # initial value for somatic potential
    D_potential = E_L*np.ones((NrON,2,Nr_D)) # initial value for dendritic potentials
    Nrn_u_delay = E_L*np.ones((NrON,2,Nr_D,int(np.round(delay_time/timeStep)))) # initial value for potentials
    Nrn_u_lowpass = E_L*np.ones((NrON,4,Nr_D)) # initial value for potentials
    
    Nrn_traces = np.append(np.append(np.zeros((NrON,2)) ,        # initial value for u_homeostatic, x
        V_T_rest, axis=1) ,              # initial value for V_T
        np.zeros((NrON,3)),axis=1)           # initial value for spike(0 or 1), ext current and noise
        
    
    Nrn_conductance = np.zeros((NrON,4,Nr_D))    #initial value for conductances/currents

    return S_potential, D_potential, Nrn_u_delay, Nrn_u_lowpass, Nrn_traces, Nrn_conductance

