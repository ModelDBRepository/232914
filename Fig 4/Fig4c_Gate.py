######################################################
# Distributed synapses are gated by a distal group
######################################################

from __future__ import division

from brian2 import *
import matplotlib.pylab as plt
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
morph = '../0. Model/Acker2008.swc'
morph_data = AckerData

distal_compartments = distal_compartments_Acker_eff
proximal_compartments = proximal_compartments_Acker

compt_var = 94

#####################################################
# Parameters
#####################################################
Theta_low = morph_data['thetalow']*mV

wa_max = .4             # maximal AMPA weight
wn_max = wa_max             # maximal NMDA weight
w_min = .3            # minimal weight


A_LTD = 0.3*A_LTD  # reduce LTD amplitude to compensate for the reduced minimum-to-maximum weight
A_LTP = 0.3*A_LTP  # reduce LTP amplitude for same reason

V_rest = 0.*mV  # resting potential input neuron
V_thresh = 0.5*mV  # threshold input neuron

NrIn = 5 # nr of input neurons (distally clustered)
Pool_size = 10 #size of distributed pools
NrPools = 3 #nr of distributed pools
Nr_distr = Pool_size*NrPools #nr of distributed neurons
init_weight = w_min # initial weight

if morph[12:13] == 'A':
    rrr = 40
else:
    rrr = 40

cluster_rate = 50*Hz #activation rate of the distal group
signal_rate = rrr*Hz # activation rate of distributed synapses

t_stim = 250*ms  # duration of one activation
buffertime = 200*ms # rest time between activations
reps = 200
t_max = reps*(t_stim+buffertime) # total simulation time


# Equations input neuron
eqs_in = ''' 
dv/dt = (V_rest-v)/ms: volt
v2 = rand()<rate_v*dt :1 (constant over dt)
rate_v :Hz
ds_trace/dt = -s_trace/taux :1
''' 


#####################################################
# Create neuron
#####################################################
N_input = NeuronGroup(NrIn+Nr_distr, eqs_in, threshold='v+v2*2*V_thresh>V_thresh', 
                      reset='v=V_rest;s_trace+=x_reset*(taux/ms)',method='linear')#


test_model = BRIANModel(morph)
neuron = test_model.makeNeuron_Ca(morph_data)
neuron.run_regularly('Mgblock = 1./(1.+ exp(-0.062*vu2)/3.57)',dt=defaultclock.dt)

print('Neurons created ...')


#####################################################



#####################################################
# create Synapses
#####################################################

Syn_1 = Synapses(N_input,neuron,
                model= eq_1_plastAMPA,
                on_pre = eq_2_plastAMPA,
                method= 'heun'
                )
# connect distal group                
Syn_1.connect(i=range(NrIn),j=neuron[compt_var:compt_var+1])#N_dend_12[7:8])

# connect distributed groups
basal_comps = len(morph_data['basal'])
other_comps = 1+len(morph_data['axon'])
rand_post_comp = np.floor(np.random.rand(Nr_distr)*(basal_comps-other_comps)+other_comps)
    
for pp in range(Nr_distr):
    Syn_1.connect(i=NrIn+pp,j=neuron[int(rand_post_comp[pp]):int(rand_post_comp[pp]+1)])#N_dend_12[7:8])

set_init_syn(Syn_1,init_weight)
Syn_1.wampa[0:NrIn] = np.multiply(np.ones(NrIn),wa_max)
Syn_1.wnmda[:] = np.multiply(np.ones(NrIn+Nr_distr),wn_max)

print('Synapses created ...')
 
#####################################################
# Initial Values
#####################################################
N_input.v = V_rest

#####################################################
# Monitors
#####################################################
M_soma = StateMonitor(neuron.main, ['v'], record=True, dt = 10*defaultclock.dt)
M_clust = StateMonitor(Syn_1,['wampa'],record=[0],  dt = 10*defaultclock.dt)
M_weights = StateMonitor(Syn_1,['wampa'],record=[NrIn + x*Pool_size+1 for x in range(NrPools)],  dt = 10*defaultclock.dt)

#####################################################
# Run
#####################################################

print('Simulating ...')
for tt in range(reps):
    if tt < int(0.5*reps):
        N_input.rate_v[0:NrIn] = cluster_rate*(mod(tt,NrPools) < 1)
    else:
        N_input.rate_v[0:NrIn] = cluster_rate*(mod(tt,NrPools) == 1)
        
        
    pvar = np.mod(tt,NrPools)*Pool_size
    pvar2 = (np.mod(tt,NrPools)+1)*Pool_size
    N_input.rate_v[NrIn+pvar:NrIn+pvar2] =  signal_rate
    
    run(t_stim)    
    N_input.rate_v = 0*Hz        
    run(buffertime)
    
    if np.mod(tt,25) ==0:
        print('> '+str(tt)+' of '+str(reps))

print('Simulation Finished!')

#####################################################
# Plot
#####################################################

data1 = open(morph[12:-4]+'AMPA_axonH_Gate_volt'+str(rrr)+'.txt','w')
data2 = open(morph[12:-4]+'AMPA_axonH_Gate_weight'+str(rrr)+'.txt','w')
json.dump((M_soma.v[0]/mV).tolist(),data1)
json.dump((M_weights.wampa[:]).tolist(),data2)
data1.close()
data2.close()

fs1 = 20
figure(figsize=(10, 4))
plot(M_soma.t/ms, M_soma.v[0]/mV,'k')
xticks( fontsize = fs1)
yticks( fontsize = fs1)
xlabel('Time [ms]',fontsize=fs1)
ylabel('Voltage [mV]',fontsize=fs1)
title(str(init_weight),fontsize = 30)
#plt.savefig('./Gate_volt.eps', format='eps', bbox_inches='tight')


plt.figure(figsize=(10, 2))
plt.plot(M_weights.t/ms,np.transpose(M_weights.wampa[:]),linewidth=4,linestyle='-')
#plt.plot(np.divide(range(len(wvar[0])),40.),np.transpose(wvar),linewidth=4,linestyle='-')
plt.legend(['Pool'+str(x+1) for x in range(NrPools)])
plt.xticks( fontsize = fs1)
plt.yticks( fontsize = fs1)
plt.ylim((w_min-0.01,wa_max+0.01))
plt.xlabel('Time [ms]',fontsize=fs1)
plt.ylabel('Synaptic Weight [a.u.]',fontsize=fs1)
plt.title(str(init_weight),fontsize = 30)
#plt.savefig('./Gate_weight.eps', format='eps', bbox_inches='tight')


plt.figure(figsize=(10, 2))
plt.plot(M_clust.t/ms,M_clust.wampa[0],linewidth=4,color='r',linestyle='-',label='cluster')
#plt.plot(np.divide(range(len(wvar[0])),40.),np.transpose(wvar),linewidth=4,linestyle='-')
plt.legend()
plt.xticks( fontsize = fs1)
plt.yticks( fontsize = fs1)
#plt.ylim((w_min,wa_max+0.1))
plt.xlabel('Time [ms]',fontsize=fs1)
plt.ylabel('Synaptic Weight [a.u.]',fontsize=fs1)
plt.title('distal group',fontsize = 30)
#

