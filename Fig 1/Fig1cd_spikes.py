######################################################
# Find the number of synapses needed for a somatic or dendritic spike, for each compartment ***
######################################################
from __future__ import division

from brian2 import *
import numpy as np
import json

import os, sys
mod_path = os.path.abspath(os.path.join('..','0. Model'))
sys.path.append(mod_path)

from oo_Parameters import *
from oo_equations_AMPAplast import *
from oo_initScripts import set_init_nrn, set_init_syn
from MakeNeuron_AMPAplast import *
from MorphologyData import *

start_scope()

######################################################
## Load Morpho
######################################################
#morph = '../0. Model/Branco2010_Morpho.swc'
#morph_data = BrancoData

morph = '../0. Model/Acker2008.swc'
morph_data = AckerData

######################################################
## Simulation Parameters
######################################################

Theta_low = morph_data['thetalow']*mV

nrIn = 25 #maximum number of synapses connected to a compartment

# select all dendritic compartments
#ii=31
compartments = morph_data['basal'][:] #range(9,487)
print(str(compartments[0])+' to '+str(compartments[-1]))

nrD = len(compartments)
dendList = range(nrD) 

# initialize matrix to store the number of synapses needed per compartment
saveNrSyn = np.zeros(nrD)

#####################################################
# AMPA - NMDA ratio
#####################################################

ampa_cond = 1500.*1e-12*siemens  # AMPA maximal conductance
nmda_cond = 1500.*1e-12*siemens  # NMDA maximal conductance

#####################################################
# Input Neurons
#####################################################
init_weight = 1.5
V_rest = -70.*mV
tau_in = 8.*ms
V_thresh = -45.*mV

# Equations input neuron
eqs_in = ''' 
dv/dt = (V_rest-v)/tau_in : volt
''' 
#####################################################
# Create neuron
#####################################################
N_input = NeuronGroup(len(dendList)*nrIn, eqs_in, threshold='v/mV>-40', reset='v=V_rest',method='linear')#

test_model = BRIANModel(morph)
neuron = test_model.makeNeuron_Ca(morph_data)
neuron.run_regularly('''Mgblock = 1./(1. +  exp(-0.062*vu2)/3.57)''', 
                    dt=defaultclock.dt)
print('Neurons created...')

#####################################################
# create Synapses
#####################################################

Syn_1 = Synapses(N_input,neuron,
                 model= eq_1_nonPlast,
                 on_pre = eq_2_nonPlast,
                method='linear'
                )

for ggg in range(len(dendList)):
    if dendList[ggg]<nrD:
        Syn_1.connect(i=ggg*nrIn+np.arange(nrIn),j=neuron[compartments[dendList[ggg]]:compartments[dendList[ggg]]+1])
    else:
        print('ERROR, wrong synaptic organisation')

#####################################################
# Initial Values
#####################################################
N_input.v = V_rest
set_init_nrn(neuron,Theta_low)
set_init_syn(Syn_1,init_weight)

print('Synapses created...')

#####################################################
# Monitors
#####################################################
M_post_soma = StateMonitor(neuron.main, ['v'], record=True)
M_post_dend = StateMonitor(neuron, ['v'], record=compartments)
      
######################################################
## Simulation
######################################################
print('Start '+morph[12:-4])
t_init = 0
times_mtx = zeros(len(dendList)+1)
times_mtx[-1] = -1
for ddd in range(len(dendList)):
    
    maxvar = -70    # the maximal depolarization will be stored in this variable
    variableIn =  2 # the initial number of synapses

        
    print('Running Compartment '+str(compartments[dendList[ddd]]))
    times_mtx[ddd] = int(t_init)
    
    # loop until the threshold theta_high is crossed
    while maxvar < Theta_high/mV:
        
        #####################################################
        # Run
        #####################################################

        
        deltat = 1.     # interspike interval between synaptic activations [ms]
        fin_time = 50   # time running the simulation after the synaptic activations [ms]    
            
        run(5.*ms) 
        for ss in range(variableIn): 
            run(deltat*ms)
            N_input.v[ddd*nrIn+ss] = 0*mV  # activate synapse
        run(fin_time*ms)
        
        maxvar = np.amax(M_post_dend.v[dendList[ddd]][t_init+400:-1]/mV) #calculate the maximum depolarisation in the dendrites
        somamax = np.amax(M_post_soma.v[0][t_init+400:-1]/mV) # calculate the maximum depolarisation in the soma
#        print('Max '+str(maxvar))
        variableIn += 1    # increment the number of synapses
        set_init_nrn(neuron,Theta_low)
        set_init_syn(Syn_1,init_weight) #reset synapses to initial values
        t_init = int(M_post_dend.t[-1]/defaultclock.dt) # change t_init to current time
#    print('--end '+str(ddd+1)+' of '+str(len(dendList))+', inp: '+str(variableIn-1))
    
    # store the number of synapses needed to reach the theta_high threshold
    # if due to a somatic spike (somamax>0), store the negative of this number
    if somamax < 0:
        saveNrSyn[ddd] = variableIn-1
    else:
        saveNrSyn[ddd] = -variableIn+1
        
        

    

#####################################################
# Plot
#####################################################

#titlestr = str(compartments[0])+'_'+str(compartments[-1])
#data1 = open('Data/AxonHFty_'+morph[12:-4]+'_nrSyn_'+titlestr+'.txt','w')
#json.dump(saveNrSyn.tolist(),data1)
#data1.close()

#figure(figsize=(10, 4))
#for dddd in range(len(dendList)):
#    plot(M_post_dend.t[int(times_mtx[dddd]):int(times_mtx[dddd+1])]/ms, M_post_dend.v[dendList[dddd]][int(times_mtx[dddd]):int(times_mtx[dddd+1])]/mV)
#plot(M_post_soma.t/ms, M_post_soma.v[0]/mV)
#legend(['dend','soma'])
#title('V')
#
#figure(figsize=(10, 4))
#plt.plot(dendList, saveNrSyn[:], '.', markersize=15)
#plt.title(titlestr)
#plt.xlim([np.amin(dendList)-.5,np.amax(dendList)+.5])
#plt.ylim([np.amin(saveNrSyn)-1.,np.amax(saveNrSyn)+1.])


print('****')
print('Data stored in saveNrSyn ')
print('End ')



