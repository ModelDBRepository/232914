######################################################
# plot weight changes after presynaptic stimulation of synapses using Poisson process
# lower rates only to show the depression
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


for paramS in [1]:
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
    
    distal_compartments_nonmda = distal_compartments_Acker_nonmda
    distal_compartments_eff = distal_compartments_Acker_eff
    proximal_compartments = proximal_compartments_Acker
    
    
    #####################################################
    # Sim parameters
    #####################################################
    
    Theta_low = morph_data['thetalow']*mV
    
    if paramS == 0:
        d_compartm = proximal_compartments
        nrCom = len(d_compartm)  
        str_var = 'prox'
        scolor = 'b'
    elif paramS ==1:
        d_compartm = distal_compartments_eff
        nrCom = len(d_compartm)      
        str_var = 'distal'
        scolor = 'r'
    elif paramS ==2:
        d_compartm = distal_compartments_nonmda
        nrCom = len(d_compartm)      
        str_var = 'distnonmda'
        scolor = 'r'
    
    
    hz_array =  np.arange(1,16) #np.array([5.,10.,15.]) # Poisson rates [1.,10.,20.,30.,40.,50.,60.,70.]
    
    
    print(str_var)
    
    init_weight = 0.5 # initial weight
    stim_time = 2000*ms # stimulation time
    NrIn = 10       # nr of synapses
    
    #####################################################
    # Input neurons
    #####################################################
    V_rest = 0.*mV
    V_thresh = 0.5*mV
    
    # Equations input neuron
    eqs_in = ''' 
    dv/dt = (V_rest-v)/ms: volt
    v2 = rand()<rate_v*dt :1  (constant over dt)
    rate_v :Hz
    ds_trace/dt = -s_trace/taux :1
    ''' 
    
    for kkk in range(nrCom):
        Connection = d_compartm[kkk]
        
        #####################################################
        # create neuron objects
        #####################################################
        
        N_input = NeuronGroup(NrIn, eqs_in, threshold='v+v2*2*V_thresh>V_thresh', 
                              reset='v=V_rest;s_trace+=x_reset*(taux/ms)',method='linear')#
    
        test_model = BRIANModel(morph)
        neuron = test_model.makeNeuron_Ca(morph_data)
        neuron.run_regularly('Mgblock = 1./(1.+ exp(-0.062*vu2)/3.57)',dt=defaultclock.dt)  
        
        print('Neurons created ...')    
        
    #    A_LTP = 14.e-4/25.         # potentiation amplitude  
        #####################################################
        # create Synapses
        #####################################################
        
        Syn_1 = Synapses(N_input,neuron,
                        model= eq_1_plastAMPA,
                        on_pre = eq_2_plastAMPA,
                        method='heun'
                        )
        
        Syn_1.connect(i=range(NrIn),j=neuron[Connection:Connection+1])
        
        print('Synapses created ...')
        
        #####################################################
        # Set Initial Neuron Parameters
        #####################################################
            
        set_init_syn(Syn_1,init_weight)
        
        N_input.v = V_rest
        N_input.s_trace = 0.

        # Monitor
#        Mv = StateMonitor(neuron,'v',record=True)
            
        #####################################################
        # Run
        #####################################################
            
        #initialize matrix to store weights
        weight_change1 = np.zeros(shape(hz_array))
        
        print('Start simulation ...')
        for jj in range(size(hz_array)):
            
            print('-> '+str(hz_array[jj])+'Hz')
            
            set_init_syn(Syn_1,init_weight)                  
                
            # Initial values
            set_init_nrn(neuron,Theta_low)
            
            N_input.v = V_rest
            N_input.s_trace = 0.
            
            run(200*ms)         
            ###### activate inputs
            N_input.rate_v = hz_array[jj]*Hz
            run(stim_time) 
            ###### deactivate inputs
            N_input.rate_v = 0*Hz
                
            #store weight changes
            weight_change1[jj] = 100*(np.mean(Syn_1.wampa))/init_weight
        run(5*ms) 
        
        print('Simulation finished!')
        
        #####################################################
        # Plots
        #####################################################
        
        #
        
        data1 = open('Data/'+morph[12:-8]+'_axonH_Poissonw_'+str(str_var)+
            '_DEPR_'+str(kkk)+'_AMPA.txt','w')
        json.dump(weight_change1.tolist(),data1)
        data1.close()

#        fig = plt.figure(figsize=(8, 6))
#        plt.plot(Mv.t/ms,Mv.v[0]/mV,'-',linewidth=2,color='k')
#        plt.plot(Mv.t/ms,Mv.v[Connection]/mV,'-',linewidth=2,color='r')
#        plt.title(str(Connection))

    fig = plt.figure(figsize=(8, 6))
    plt.plot(hz_array,weight_change1,'.-',linewidth=2,color=scolor)
    plt.xlabel('Presynaptic rate [Hz]',fontsize=22)
    plt.ylabel('Normalised weight [%]',fontsize=22)
    plt.subplots_adjust(bottom=0.2,left=0.15,right=0.95,top=0.85)
    plt.title(str_var,fontsize=30)
#    plt.savefig('./IMG/'+str(str_var)+'_60.eps', format='eps', dpi=1000)
    
