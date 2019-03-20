######################################################
# coactive inhibition can switch excitatory plasticity
# from LTP to LTD
######################################################

from __future__ import division

from brian2 import *
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
inhibvar = 1 #1: inhibition, 0: no inhibition
######################################################


######################################################
## Load Morpho
######################################################
#morph = '../0. Model/Branco2010_Morpho.swc'
#morph_data = BrancoData
#
#dc = distal_compartments_Branco_eff #[distal_compartments_Branco[0], distal_compartments_Branco[3]]
#pc = proximal_compartments_Branco

morph = '../0. Model/Acker2008.swc'
morph_data = AckerData

dc = [distal_compartments_Acker_eff[1],distal_compartments_Acker_eff[3]-2] #0 14
pc = [proximal_compartments_Acker[4], proximal_compartments_Acker[5]] #12 14

Theta_low = morph_data['thetalow']*mV

#####################################################
# AMPA - NMDA ratio
#####################################################

ampa_cond = 1500.*1e-12*siemens  # AMPA maximal conductance
nmda_cond = 1500.*1e-12*siemens  # NMDA maximal conductance
gaba_cond = 1000.*1e-12*siemens  # NMDA maximal conductance

#####################################################
# Input Neuron
#####################################################
V_rest = 0.*mV
V_thresh = 0.5*mV

clustvar = 0 #0 = dist cluster, 1=prox cluster
randomDistr = 0 # 0= clustered, 1=distributed
meannoise = 100
deltaActivations = 5

if clustvar ==0:
    compv = dc
else:
    compv = pc


NrInhib = 5
NrPools = 1  #nr of input pools
NrNrns = 5   # nr of neurons per pool
init_weight = .5  # initial weight
signal_rate = 80.*Hz # activation rate of the pools
t_stim = 30*ms # length of activation
buffertime = 150*ms # rest time between two activations
#A_LTP = .1*A_LTP

nrPoolsAct = 1 #51 # total number of activations
t_max = nrPoolsAct*(t_stim+buffertime)

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
N_input = NeuronGroup(2*NrPools*NrNrns, eqs_in, threshold='v+v2*2*V_thresh>V_thresh', 
                      reset='v=V_rest;s_trace+=x_reset*(taux/ms)',method='linear')#
N_inh = NeuronGroup(NrInhib, eqs_in, threshold='v+v2*2*V_thresh>V_thresh', 
                      reset='v=V_rest;s_trace+=x_reset*(taux/ms)',method='linear')#


test_model = BRIANModel(morph)
neuron = test_model.makeNeuron_Ca(morph_data)
neuron.run_regularly('Mgblock = 1./(1.+ exp(-0.062*vu2)/3.57)',dt=defaultclock.dt)

print('Neurons created...')


#####################################################

bsl = list(morph_data['basal'])
apc = list(morph_data['apical'])
axn = list(morph_data['axon'])


#####################################################
# create Synapses
#####################################################

Syn_1 = Synapses(N_input,neuron,
                model= eq_1_plastAMPA,
                on_pre = eq_2_plastAMPA,
                method='heun'
                )

for rr in range(NrPools):
        Syn_1.connect(i=range(rr*NrNrns,(rr+1)*NrNrns),j=neuron[compv[rr]:compv[rr]+1])
        Syn_1.connect(i=range(NrPools*NrNrns+rr*NrNrns,NrPools*NrNrns+(rr+1)*NrNrns),j=neuron[compv[rr]:compv[rr]+1])

set_init_syn(Syn_1,init_weight)
nr_syn = NrPools*NrNrns
Syn_1.wampa[0:NrPools*NrNrns] = init_weight*ones(nr_syn)
Syn_1.wnmda[0:NrPools*NrNrns] = ones(nr_syn)
Syn_1.wampa[NrPools*NrNrns:] = zeros(nr_syn)
Syn_1.wnmda[NrPools*NrNrns:] = zeros(nr_syn)
Syn_1.wgaba[NrPools*NrNrns:] = ones(nr_syn)
    
set_init_nrn(neuron,Theta_low)

print('Synapses created...')
 
#####################################################
# Initial Values
#####################################################
N_input.v = V_rest

#####################################################
# Monitors
#####################################################
M_soma = StateMonitor(neuron.main, ['v'], record=True, dt = 25*defaultclock.dt)

M_dend = StateMonitor(neuron[compv[0]:compv[0]+1], ['v','Igaba'], record=True, dt = 25*defaultclock.dt)

M_weights = StateMonitor(Syn_1,['wampa'],record=range(NrPools*NrNrns), dt = 25*defaultclock.dt)

#####################################################
# Run
#####################################################

print('Simulating...')

#noisy current in soma
neuron.main.noise_sigma = 10*pA # initial value membrane voltage
neuron.main.noise_avg = meannoise*pA # initial value membrane voltage   

run(50*ms)
for tt in range(int(t_max/(t_stim+buffertime))):
    
    N_input.rate_v[0:NrPools*NrNrns] = signal_rate
    N_input.rate_v[NrPools*NrNrns:] = 0*Hz
    
    if inhibvar:
        N_input.rate_v = signal_rate  
        
    run(t_stim)
    N_input.rate_v = 0*Hz
    N_inh.rate_v = 0*Hz
    run(buffertime)

    # Display the progress
    if np.abs(tt-nrPoolsAct*0.2*np.round(5*tt/nrPoolsAct)) < 0.5:
        print('> '+str(np.round(100*tt/nrPoolsAct))+'%') 


#####################################################
# Plot
#####################################################

Mdend_max = np.amax(M_dend.v/mV,axis=0)
fs = 35

figure(figsize=(10, 4))
t_len = len(M_soma.t)
nrInp_plot = nrPoolsAct
plot(M_soma.t[t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/ms, M_soma.v[0][t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/mV,linewidth=4,color='k',label='soma')
plot(M_dend.t[t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/ms, M_dend.v[0][t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/mV,linewidth=4,color='r',label='distal')
ylim((-90,0))
title('V binding '+str(clustvar)+str(randomDistr)+' '+str(signal_rate/Hz)+str(meannoise))
legend(fontsize=fs)
xlabel('Time [ms]',fontsize=fs)
ylabel('Voltage [mV]',fontsize=fs)
xticks( fontsize = fs)
yticks( fontsize = fs)
#plt.savefig('Inhib_'+str(inhibvar)+'_v.eps', format='eps', dpi=1000)

#figure(figsize=(10, 4))
#plot(M_dend.t/ms, M_dend.Igaba[0]/mV)

if clustvar==0:
    cstr = 'r'
else:
    cstr = 'b'

figure(figsize=(10, 6))
for jjj in range(NrPools*NrNrns):
    plot(M_weights.t/ms,np.transpose(M_weights.wampa[jjj]),linewidth=4,color='k')
#plot(M_weights.t/ms,np.transpose(M_weights.wampa[1]),linewidth=2,color=cstr)
#plot(M_weights.t/ms,np.transpose(M_weights.wampa[NrIn:NrIn+Nr_scattered]),linewidth=1,linestyle=':',color=[0.2,0.2,0.2])
#ylim((0,1.1))
title('binding'+str(clustvar)+str(randomDistr)+str(meannoise))
xlabel('Time [ms]',fontsize=fs)
ylabel('Syn. weight [a.u.]',fontsize=fs)
xticks( fontsize = fs)
yticks( fontsize = fs)
#plt.savefig('Inhib_'+str(inhibvar)+'_w.eps', format='eps', dpi=1000)


print('Simulation finished!')