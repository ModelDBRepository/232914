######################################################
# distal groups ensure robust associations
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
## Load Morpho
######################################################
#morph = '../0. Model/Branco2010_Morpho.swc'
#morph_data = BrancoData
#
#dc = distal_compartments_Branco_eff #[distal_compartments_Branco[0], distal_compartments_Branco[3]]
#pc = proximal_compartments_Branco

morph = '../0. Model/Acker2008.swc'
morph_data = AckerData

dc = [52,461] 
pc = [15,213] 

Theta_low = morph_data['thetalow']*mV

#####################################################
# AMPA - NMDA ratio
#####################################################

ampa_cond = .5*1500.*1e-12*siemens  # AMPA maximal conductance
nmda_cond = .5*1500.*1e-12*siemens  # NMDA maximal conductance


#####################################################
# Input Neuron
#####################################################
V_rest = 0.*mV
V_thresh = 0.5*mV

clustvar = 1  #0 = dist cluster, 1=prox cluster
randomDistr = 0 # 0= clustered, 1=distributed
meannoise = 35
deltaActivations = 10

if clustvar ==0:
    compv = dc
else:
    compv = pc



NrPools = 2  #nr of input pools
NrNrns_clst = 15   # nr of neurons per pool
NrNrns_rnd = 0  # nr of neurons per pool
NrNrns = NrNrns_clst+NrNrns_rnd  # nr of neurons per pool

wmax = 1
wa_max = wmax             # maximal AMPA weight
wn_max = wmax             # maximal NMDA weight
init_weight_a = 1.*wa_max  # initial weight
init_weight_n = 1.*wn_max  # initial weight


signal_rate = 40.*Hz # activation rate of the pools
t_stim = 50*ms # length of activation
buffertime = 150*ms # rest time between two activations

nrPoolsAct = 301 #51 # total number of activations
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
N_input = NeuronGroup(NrPools*NrNrns, eqs_in, threshold='v+v2*2*V_thresh>V_thresh', 
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

# connect distributed groups
basal_comps = len(morph_data['basal'])
other_comps = 1+len(morph_data['axon'])

comprv1 = [ 257.,  128.,  455.,   25.,  445.]
comprv2 = [ 190.,  289.,   28.,  380.,  231.]

for rr in range(NrPools):
        if NrNrns_clst != 0:
            Syn_1.connect(i=range(rr*NrNrns,rr*NrNrns+NrNrns_clst),j=neuron[compv[rr]:compv[rr]+1])
        if mod(rr,2) ==0:   
#            rand_post_comp = np.floor(np.random.rand(NrNrns_rnd)*(basal_comps-other_comps)+other_comps) #np.array([ 103.,  243.,  235.,  118.,  125.]) #
#            print(rand_post_comp)   
            if rr ==0:
                rand_post_comp = comprv1
            elif rr == 2:
                rand_post_comp = comprv2
            else:
                print('ERROR random compartments...')
            for pp in range(NrNrns_rnd):
                Syn_1.connect(i=rr*NrNrns+NrNrns_clst+pp,j=neuron[int(rand_post_comp[pp]):int(rand_post_comp[pp]+1)])

#set_init_syn(Syn_1,init_weight)
nr_syn = len(Syn_1.wampa[:])
Syn_1.wampa[:] = init_weight_a*ones(nr_syn)
Syn_1.wnmda[:] = init_weight_n*ones(nr_syn)

#for rr in range(NrPools):
#        Syn_1.wampa[rr*NrNrns+NrNrns_clst:rr*NrNrns+NrNrns_clst+NrNrns_rnd]= np.zeros(NrNrns_rnd)

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

M_dend = StateMonitor(neuron, ['v'], record=True, dt = 25*defaultclock.dt)

M_weights = StateMonitor(Syn_1,['wampa'],record=[0,NrNrns+1,2*NrNrns+1,3*NrNrns+1], dt = 25*defaultclock.dt)

#####################################################
# Run
#####################################################

print('Simulating...')

#noisy current in soma
neuron.main.noise_sigma = .1*meannoise*pA # initial value membrane voltage
neuron.main.noise_avg = meannoise*pA # initial value membrane voltage   

run(300*ms)
for tt in range(int(t_max/(t_stim+buffertime))):
    propv = 1
    N_input.rate_v[mod(tt,int(propv*NrPools))*NrNrns:(mod(tt,int(propv*NrPools))+1)*NrNrns] = signal_rate
#    N_input.rate_v[(mod(tt,int(propv*NrPools))+int(propv*NrPools))*NrNrns:(mod(tt,int(propv*NrPools))+int(propv*NrPools)+1)*NrNrns] = signal_rate
#    N_input.rate_v[(mod(tt,int(propv*NrPools))+2*int(propv*NrPools))*NrNrns:(mod(tt,int(propv*NrPools))+2*int(propv*NrPools)+1)*NrNrns] = signal_rate
    if mod(tt,int(deltaActivations*NrPools/1)) < 1:
        N_input.rate_v = signal_rate  
        
#    if mod(tt,deltaActivations*NrPools) < 1:
#        for pp in range(NrNrns):
#            N_input.v[pp] = 1*mV
#            N_input.v[pp+NrNrns] = 1*mV
#            run(1*ms)
#    else:
#        for pp in range(NrNrns):
#            N_input.v[(mod(tt,NrPools))*NrNrns+pp] = 1*mV
#            run(1*ms)
        
        
    run(t_stim)
    N_input.rate_v = 0*Hz
    run(buffertime)

    # Display the progress
    if np.abs(tt-nrPoolsAct*0.2*np.round(5*tt/nrPoolsAct)) < 0.5:
        print('> '+str(np.round(100*tt/nrPoolsAct))+'%') 


#####################################################
# Plot
#####################################################

Mdend_max = np.amax(M_dend.v/mV,axis=0)

data1 = open('AMPA_axonH2_FBinding_'+str(clustvar)+str(randomDistr)+str(meannoise)+'_volt'+'.txt','w')
data2 = open('AMPA_axonH2_FBinding_'+str(clustvar)+str(randomDistr)+str(meannoise)+'_weight'+'.txt','w')
data3 = open('AMPA_axonH2_FBinding_'+str(clustvar)+str(randomDistr)+str(meannoise)+'_Dend'+'.txt','w')
json.dump((M_soma.v[0]/mV).tolist(),data1)
json.dump((M_weights.wampa[:]).tolist(),data2)
json.dump((Mdend_max).tolist(),data3)
data1.close()
data2.close()
data3.close()


figure(figsize=(10, 4))
t_len = len(M_soma.t)
nrInp_plot = nrPoolsAct
plot(M_soma.t[t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/ms, 
     M_soma.v[0][t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/mV)
plot(M_dend.t[t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/ms, 
     M_dend.v[compv[0]][t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/mV,label=str(compv[0]))
plot(M_dend.t[t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/ms, 
     M_dend.v[compv[1]][t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/mV,label=str(compv[1]))
#plot(M_dend.t[t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/ms, 
#     M_dend.v[compv[2]][t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/mV,label=str(compv[2]))
#plot(M_dend.t[t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/ms, 
#     M_dend.v[compv[3]][t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/mV,label=str(compv[3]))
#plot(M_dend.t[t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/ms, 
#     M_dend.v[compv[4]][t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/mV,label=str(compv[4]))
#plot(M_dend.t[t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/ms, 
#     M_dend.v[compv[5]][t_len-nrInp_plot*int(((t_stim+buffertime)/defaultclock.dt)):-1]/mV,label=str(compv[5]))
ylim((-80,0))
title('V binding '+str(clustvar)+str(randomDistr)+' '+str(signal_rate/Hz)+str(meannoise))
legend()


#
if clustvar==0:
    cstr = 'r'
else:
    cstr = 'b'

figure(figsize=(10, 6))
plot(M_weights.t/ms,np.transpose(M_weights.wampa[0]),linewidth=2,color='k')
plot(M_weights.t/ms,np.transpose(M_weights.wampa[1]),linewidth=2,color=cstr)
#plot(M_weights.t/ms,np.transpose(M_weights.wampa[2]),linewidth=2,color='g')
#plot(M_weights.t/ms,np.transpose(M_weights.wampa[3]),linewidth=2,color='y')
#plot(M_weights.t/ms,np.transpose(M_weights.wampa[NrIn:NrIn+Nr_scattered]),linewidth=1,linestyle=':',color=[0.2,0.2,0.2])
ylim((0,1.1))
title('binding'+str(clustvar)+str(randomDistr)+str(meannoise)+str(signal_rate))


print('Simulation finished!')