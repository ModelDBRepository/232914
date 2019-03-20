from __future__ import division

######################################################
# Reproduce the integration gradient observed in Branco and Hausser 2011 ***
#
syn_location = 'dist' # CHOOSE 'prox' to run with proximal group or 'dist' to run with distal group
######################################################


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

morph = '../0. Model/Acker2008.swc'
morph_data = AckerData

distal_compartments = distal_compartments_Acker
proximal_compartments = proximal_compartments_Acker

#morph = '../../0. Model/Branco2010_Morpho.swc'
#morph_data = BrancoData
#
#distal_compartments = distal_compartments_Branco
#proximal_compartments = proximal_compartments_Branco


Theta_low = morph_data['thetalow']*mV
    
#####################################################
# AMPA - NMDA ratio
#####################################################

condval = 2000
ampa_cond = condval*1e-12*siemens  # AMPA maximal conductance
nmda_cond = condval*1e-12*siemens  # NMDA maximal conductance

#####################################################
# Simulation parameters
#####################################################

# initial weight
init_weight = 1.
# interspike intervals
deltat_M = [1,4,8]
# final time for simulation
fin_time = 200
# number of activated synapses
nrInpts = 10


print('Running '+syn_location)

#####################################################
# Input Neuron
#####################################################
V_rest = -70.*mV
tau_in = 8.*ms
V_thresh = -45.*mV
C = 200.*pF # membrane capacitance
rate = 0.5*Hz

initcomp = 6

if syn_location == 'prox':
    d_compartm = proximal_compartments[initcomp:]
    nrIn = len(d_compartm)    
    ca = 0
    cb = 1
elif syn_location == 'dist':
    d_compartm = distal_compartments[initcomp:]
    nrIn = len(d_compartm)  
    ca = 1
    cb = 0
else:
    print('ERROR: wrong synaptic location')

#------------
# Equations input neuron
#------------
eqs_in = ''' 
dv/dt = (V_rest-v)/tau_in + Idrive/C: volt
Idrive : amp
''' 

#####################################################
# Create neuron
#####################################################
N_input = NeuronGroup(nrIn, eqs_in, threshold='v/mV>-40', reset='v=V_rest',method='linear')#

test_model = BRIANModel(morph)
neuron = test_model.makeNeuron_Ca(morph_data)
neuron.run_regularly('Mgblock = 1./(1.+ exp(-0.062*vu2)/3.57)',dt=defaultclock.dt)

print('Neurons created')

#####################################################
# create Synapses
#####################################################

Syn_1 = Synapses(N_input,neuron,
                model= eq_1_nonPlast,
                on_pre = eq_2_nonPlast,
                method='linear'
                )

for jj in range(nrIn):
    Syn_1.connect(i=jj,j=neuron[d_compartm[jj]:d_compartm[jj]+1])

set_init_syn(Syn_1,init_weight)
 
print('Synapses created') 
 

for kk in range(nrIn):
    
    print('Start compartment '+str(initcomp+kk+1)+' of '+ str(nrIn))
    
    #####################################################
    # Initial Values
    #####################################################
    
    N_input.v = V_rest
    N_input.Idrive = 0*pA
    
    noise_std = 0*pA
    noise_mean = 0*pA
    
    #####################################################
    # Monitors
    #####################################################
    M_soma = StateMonitor(neuron.main, ['v'], record=True)
    M_dend = StateMonitor(neuron[d_compartm[kk]:d_compartm[kk]+1], ['v'], record=True)
    
    #####################################################
    # Run
    #####################################################
    
    
    print('Start simulation ...')
    for ii in range(len(deltat_M)):
        run(fin_time/len(deltat_M)*ms, report=None) 
        for jj in range(nrInpts):
            deltat = deltat_M[ii]
            run(deltat*ms, report=None)
            N_input.v[kk] = 0*mV
        
        run(fin_time*ms-deltat*nrInpts*ms, report=None) 
        # Initial values
        set_init_syn(Syn_1,init_weight)
        set_init_nrn(neuron,Theta_low)
    
    print('Simulation finished!')
    
    #####################################################
    # Plot
    #####################################################
    
    tstep = defaultclock.dt/ms
    wblock = int(fin_time/tstep/len(deltat_M))
    block = wblock+int(fin_time/tstep)
    
    titlestr = morph[12:-8]+'_axonH_11_'+str(condval)+'_'+'_Int'+syn_location+'_'+str(initcomp+kk)
    if np.amax( M_dend.v[0][wblock:block]) > Theta_high:
        var_spike = 0
    else:
        var_spike = 1
    
#    data_01 = open('Data/'+titlestr+'1.txt','w')
#    data2_01 = M_soma.v[0][wblock:block]/mV
#    data_02 = open('Data/'+titlestr+'2.txt','w')
#    data2_02 = M_soma.v[0][block+wblock:2*block]/mV
#    data_03 = open('Data/'+titlestr+'3.txt','w')
#    data2_03 = M_soma.v[0][2*block+wblock:3*block]/mV
#    data_04 = open('Data/'+titlestr+'4.txt','w')
#    data2_04 = [var_spike,d_compartm[kk]]
#    
#    json.dump(data2_01.tolist(),data_01)
#    json.dump(data2_02.tolist(),data_02)
#    json.dump(data2_03.tolist(),data_03)
#    json.dump(data2_04,data_04)
#    
#    data_01.close()
#    data_02.close()
#    data_03.close()
#    data_04.close()
    
    
    lw = 4
    fs1 = 35
    fst = 45
    
    figure(figsize=(16, 8))
    plot(M_soma.t[0:block-wblock]/ms, M_soma.v[0][wblock:block]/mV, linewidth = lw, color = [ca, 0, cb]) 
    plot(M_soma.t[0:block-wblock]/ms, M_soma.v[0][block+wblock:2*block]/mV, linewidth = lw, color = [0.7*ca, 0, 0.7*cb]) 
    plot(M_soma.t[0:block-wblock]/ms, M_soma.v[0][2*block+wblock:3*block]/mV, linewidth = lw, color = [0.5*ca, 0, 0.5*cb]) 
    title('Integration Gradients',fontsize=fst)
    xticks( fontsize = fs1)
    yticks( fontsize = fs1)
    xlabel('Time [ms]',fontsize=fs1)
    ylabel('Voltage [mV]',fontsize=fs1)
    ylim((-76,-62))
    legend(['1ms','4ms','8ms'],title='ISI',fontsize=fs1)
    
    #plt.savefig('Integr_'+syn_location+'.eps', format='eps', bbox_inches='tight')
