######################################################
# Reproduce the rate dependence of plasticity as observed in Sjostrom et al. 2001
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


for paramS in [0,1]: 
    start_scope()
        
    ######################################################
    ## Load Morpho
    ######################################################
#    morph = '../0. Model/Branco2010_Morpho.swc'
#    morph_data = BrancoData
#    
#    distal_compartments_nonmda = distal_compartments_Branco_nonmda
#    distal_compartments_eff = distal_compartments_Branco_eff
#    proximal_compartments = proximal_compartments_Branco
    
    morph = '../0. Model/Acker2008.swc'
    morph_data = AckerData
    
    startcomp = 0
    
    distal_compartments_nonmda = distal_compartments_Acker_nonmda
    distal_compartments_eff = distal_compartments_Acker_eff
    proximal_compartments = proximal_compartments_Acker
    
    #####################################################
    # Sim parameters
    #####################################################    
    
    Theta_low = morph_data['thetalow']*mV
        
    if paramS == 0:
        d_compartm = proximal_compartments
        nrIn = len(d_compartm)  
        str_var = 'prox'
    elif paramS ==1:
        d_compartm = distal_compartments_eff
        nrIn = len(d_compartm)      
        str_var = 'disteff'
    elif paramS ==2:
        d_compartm = distal_compartments_nonmda
        nrIn = len(d_compartm)      
        str_var = 'distnonmda'
        
    print('***')    
    if morph[12:-8] == 'Acker':
        print('-- L5 '+str_var+'--')   
    else:
        print('-- L2/3 '+str_var+'--')  
    
    # rates for the protocol as in sjostrom2001
    hz_array =  np.array([1.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.]) #1.,10.,20.,40.,50.
    # initial weight
    init_weight = 0.5
    # number of pairings
    reps = 5
    
    
    #####################################################
    # Input neurons
    #####################################################
    V_rest = -70.*mV
    tau_in = 8.*ms
    V_thresh = -45.*mV
    C = 200.*pF # membrane capacitance
    
    #------------
    # Equations input neuron
    #------------
    eqs_in = ''' 
    dv/dt = (V_rest-v)/tau_in + Idrive/C: volt
    Idrive : amp
    ds_trace/dt = -s_trace/taux :1
    ''' 
    
    #####################################################
    # create spatial neuron objects
    #####################################################
    
    # IandF input neurons
    N_input = NeuronGroup(2*nrIn, eqs_in, threshold='v>V_thresh', 
                          reset='v=V_rest;s_trace+=x_reset*(taux/ms)', method='linear')#
    
    test_model = BRIANModel(morph)
    neuron = test_model.makeNeuron_Ca(morph_data)
    neuron.run_regularly('Mgblock = 1./(1.+ exp(-0.062*vu2)/3.57)',dt=defaultclock.dt)
    neuron2 = test_model.makeNeuron_Ca(morph_data) 
    neuron2.run_regularly('Mgblock = 1./(1.+ exp(-0.062*vu2)/3.57)',dt=defaultclock.dt)
    
    print('Neurons created...')    
            
    #####################################################
    # create Synapses
    #####################################################
    
    Syn_1 = Synapses(N_input,neuron,
                    model= eq_1_plastAMPA,
                    on_pre = eq_2_plastAMPA,
                    method='heun'
                    )
    Syn_2 = Synapses(N_input,neuron2,
                    model= eq_1_plastAMPA,
                    on_pre = eq_2_plastAMPA,
                    method='heun'
                    )
    
    for jj in range(nrIn):
        Syn_1.connect(i=jj,j=neuron[d_compartm[jj]:d_compartm[jj]+1])
        Syn_2.connect(i=nrIn+jj,j=neuron2[d_compartm[jj]:d_compartm[jj]+1])
    
    print('Synapses created...')    
    
    for zzz in range(nrIn):
        print('Start compartment '+str(zzz+1)+','+str(zzz+1)+' of '+ str(nrIn))
        
        #####################################################
        # Set Initial Neuron Parameter values
        #####################################################
        
        set_init_syn(Syn_1,init_weight)
        set_init_syn(Syn_2,init_weight)
        
        N_input.v = V_rest
        N_input.s_trace = 0.
        
        #####################################################
        # Run
        #####################################################
        weight_change1 = np.zeros(shape(hz_array))
        weight_change2 = np.zeros(shape(hz_array))            
        
        print('Start running ...')    
        
        for jj in range(size(hz_array)):
            pair_interval = 1000./hz_array[jj]-13.
            print('-> '+str(hz_array[jj])+'Hz')        
            
            set_init_syn(Syn_1,init_weight)
            set_init_syn(Syn_2,init_weight)               
            
            # Initial values
            set_init_nrn(neuron,Theta_low)
            set_init_nrn(neuron2,Theta_low)
            N_input.v = V_rest
            N_input.s_trace = 0.
            
            
            run(100*ms) 
            
            # Pairings
            for ii in range(reps):
                neuron.I = 0.*pA
                neuron2.I = 0.*pA
                N_input.Idrive = 0.*mA
                ###### 1st SPIKE
                neuron2.main.I = 1000.*pA        
                N_input.Idrive[zzz] = 2000.*pA
                run(3*ms) 
                neuron2.I = 0.*pA
                N_input.Idrive = 0.*mA
                run(7*ms) 
                ###### 2nd SPIKE
                neuron.main.I = 1000.*pA    
                N_input.Idrive[nrIn+zzz] = 2000.*pA
                run(3*ms) 
                neuron.I = 0.*pA
                N_input.Idrive = 0.*mA
                ######
                run(pair_interval*ms) 
                
#            print(Syn_1.wnmda[zzz])
                
            #store weight changes
            weight_change1[jj] = 100.*(Syn_1.wampa[zzz] + 15.*(Syn_1.wampa[zzz]-init_weight))/init_weight
            weight_change2[jj] = 100.*(Syn_2.wampa[zzz] + 15.*(Syn_2.wampa[zzz]-init_weight))/init_weight
        run(5*ms) 
        
        print('Finished running!')    
        
        #####################################################
        # Plots
        #####################################################
        
        #
        
        titlestr = 'Data/'+morph[12:-8]+'_axonH_Sjostr_'+str_var+'_'+str(zzz)
        
#        data1 = open(titlestr+'_AMPA_w1.txt','w')
#        data2 = open(titlestr+'_AMPA_w2.txt','w')
#        json.dump(weight_change1.tolist(),data1)
#        json.dump(weight_change2.tolist(),data2)
#        data1.close()
#        data2.close()
    
        if paramS==0:
            stitle = 'Prox Ca'
            scolor = 'b'
        else:
            stitle = 'Dist Ca'
            scolor = 'r'
        fig = figure(figsize=(8, 5))
        plt.plot(hz_array,weight_change1,'.-',linewidth=2,color=scolor)
        plt.plot(hz_array,weight_change2,'.:',linewidth=2,color=scolor)
        xlabel('Pairing frequency [Hz]',fontsize=22)
        ylabel('Normalised Weight [%]',fontsize=22)
        legend(['Pre-Post','Post-Pre'],loc='best')    
        plt.subplots_adjust(bottom=0.2,left=0.15,right=0.95,top=0.85)
        title(stitle)
    #    plt.savefig('./IMG/'+str(str_var)+'_final.eps', format='eps', dpi=1000)
    
