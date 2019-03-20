######################################################
# Evoke NMDA spike and store local and somatic voltage ***
######################################################

from __future__ import division

from brian2 import *
import numpy as np
import matplotlib.pylab as plt
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
morph = '../0. Model/Branco2010_Morpho.swc'
morph_data = BrancoData

distal_compartments = distal_compartments_Branco_eff
proximal_compartments = proximal_compartments_Branco

#morph = '../0. Model/Acker2008.swc'
#morph_data = AckerData
#
#distal_compartments = distal_compartments_Acker_eff
#proximal_compartments = proximal_compartments_Acker

dist_c = distal_compartments[1] # distal compartment

#####################################################
# Input Neuron (integr and fire)
#####################################################
V_rest = -70.*mV #resting potential
tau_in = 8.*ms # membrane timeconstant
V_thresh = -45.*mV # spiking threshold
C = 200.*pF # membrane capacitance

NrIn = 10  # number of input spikes
init_weight = 1 #initial weight
deltat = 0.1 #interspike interval [ms]

# Equations input neuron
eqs_in = ''' 
dv/dt = (V_rest-v)/tau_in + Idrive/C: volt
Idrive : amp
ds_trace/dt = -s_trace/taux :1
''' 

#####################################################
# Create neurons
#####################################################

#inputs
N_input = NeuronGroup(1, eqs_in, threshold='v/mV>-40', 
                      reset='v=V_rest;s_trace+=x_reset*(taux/ms)',method='linear')#

#morphological neuron
test_model = BRIANModel(morph)
neuron = test_model.makeNeuron_Ca(morph_data)
neuron.run_regularly('Mgblock = 1./(1.+ exp(-0.062*vu2)/3.57)',dt=defaultclock.dt)

print('Neurons created...')

#####################################################
# create Synapses
#####################################################

Syn_1 = Synapses(N_input,neuron,
                model= eq_1_nonPlast,
                on_pre = eq_2_nonPlast,
                method = 'heun'
                )

Syn_1.connect(i=0,j=neuron[dist_c:dist_c+1])

set_init_syn(Syn_1,init_weight)

print('Synapses created...')
 
#####################################################
# Initial Values
#####################################################

N_input.v = V_rest
N_input.Idrive = 0*pA

#####################################################
# Monitors
#####################################################
M_soma = StateMonitor(neuron.main, ['v'], record=True)
M_dend = StateMonitor(neuron[dist_c:dist_c+1], ['v'], record=True)

#####################################################
# Run
#####################################################
print('Start simulation...')

fin_time = 150 #final time of integration

run(50*ms) 

# loop over input spikes
for jj in range(NrIn):
    run(deltat*ms)
    N_input.v[0] = 0*mV
        
run(fin_time*ms) 

print('Simulation finished!')

#####################################################
# Plot
#####################################################
lw=4
fs1=25
figure(figsize=(10, 4))
plot(M_soma.t/ms, M_soma.v[0]/mV,color='k',linewidth=lw)
plot(M_dend.t/ms, M_dend.v[-1]/mV,color='r',linewidth=lw)
plot([0,M_dend.t[-1]/ms], [Theta_high/mV,Theta_high/mV],'-.',color='k',linewidth=lw)
xticks( fontsize = fs1)
yticks( fontsize = fs1)
xlim((0,200))
xlabel('Time [ms]',fontsize=fs1)
ylabel('Voltgae [mV]',fontsize=fs1)
text(125,-25,'LTP threshold',color='k',fontsize=fs1)
title('NMDA spike',fontsize=30)
#plt.savefig('NMDAsp'+'.eps', format='eps', bbox_inches='tight')


