######################################################
# Connectivity between 2 neurons
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



for zzz in range(1):
    start_scope()
    namestr = '30'
    print('>> filename will be '+namestr)

    ######################################################
    ## Load Morpho
    ######################################################
    #morph = '../0. Model/Branco2010_Morpho.swc'
    #morph_data = BrancoData
    #
    #dc = distal_compartments_Branco_eff
    #pc = proximal_compartments_Branco
    
    morph = '../0. Model/Acker2008.swc'
    morph_data = AckerData
    
    dc = distal_compartments_Acker_eff
    pc = proximal_compartments_Acker
    
    Theta_low = morph_data['thetalow']*mV
    
    #####################################################
    loc1 = 'all'           # synaptic location: choose prox (proximal), dist (distal), branch (one basal branch) or all (all basals)
    syn_arrange1 = 'distrib'  # synaptic arrangement: choose clust (clustered) or distrib (distributed)
    loc2 = 'all'           # synaptic location: choose prox (proximal), dist (distal), branch (one basal branch) or all (all basals)
    syn_arrange2 = 'distrib'  # synaptic arrangement: choose clust (clustered) or distrib (distributed)
    #####################################################
    
    
    t_max = 5*second
    
    init_weight = .5
    
    NrSyn = 15
    ampa_cond = 1000.*1e-12*siemens  # AMPA maximal conductance
    nmda_cond = 2000.*1e-12*siemens  # NMDA maximal conductance
    
    if loc1 == 'prox':
        compv1 = pc
    elif loc1 == 'dist':
        compv1 = dc
    elif loc1 == 'branch':
        ii = 0
        b = dc[int(np.random.rand(1)*len(dc))]
        compv1 = []
        while b-ii not in pc:
            compv1 = compv1 + [b-ii]
            ii = ii+1
    elif loc1 == 'all':
        compv1 = list(morph_data['basal'])
    else:
        print('ERROR: synaptic location key error')
        
    if syn_arrange1 == 'clust':
        syn_loc1 = np.array([int(np.random.rand(1)*len(compv1))]*NrSyn)
        saveloc1 = str(syn_loc1[0])
    elif syn_arrange1 == 'distrib':
        syn_loc1 = np.floor((np.random.rand(NrSyn)*len(compv1)))
        saveloc1 = str(int(syn_loc1[0]))
    else:
        print('ERROR: synaptic arrangement key error')
        
    if loc2 == 'prox':
        compv2 = pc
    elif loc2 == 'dist':
        compv2 = dc
    elif loc1 == 'branch':
        ii = 0
        b = dc[int(np.random.rand(1)*len(dc))]
        compv2 = []
        while b-ii not in pc:
            compv2 = compv2 + [b-ii]
            ii = ii+1
    elif loc2 == 'all':
        compv2 = list(morph_data['basal'])
    else:
        print('ERROR: synaptic location key error')
        
    if syn_arrange2 == 'clust':
        syn_loc2 = np.array([int(np.random.rand(1)*len(compv2))]*NrSyn)
        saveloc2 = str(syn_loc2[0])
    elif syn_arrange2 == 'distrib':
        syn_loc2 = np.floor((np.random.rand(NrSyn)*len(compv2)))
        saveloc2 = str(int(syn_loc2[0]))
    else:
        print('ERROR: synaptic arrangement key error')
    
    print('Start: '+loc1+'/'+syn_arrange1+saveloc1+' - '+loc2+'/'+syn_arrange2+saveloc2)
    
    ratesv = np.arange(0,20,2)
    storew = np.zeros((len(ratesv),len(ratesv),2*NrSyn))
    
    for rr1 in range(len(ratesv)):
        for rr2 in range(rr1,len(ratesv)):
            
            start_scope()
            print('scope')
            
            rate_1 = ratesv[rr1]
            rate_2 = ratesv[rr2]
    
            #####################################################
            # Create neuron
            #####################################################
            
            test_model = BRIANModel(morph)
            
            neuron = test_model.makeNeuron_Ca(morph_data)
            neuron.run_regularly('Mgblock = 1./(1.+ exp(-0.062*vu2)/3.57)',dt=defaultclock.dt)
            
            neuron2 = test_model.makeNeuron_Ca(morph_data)
            neuron2.run_regularly('Mgblock = 1./(1.+ exp(-0.062*vu2)/3.57)',dt=defaultclock.dt)
            
            print('Neurons created...')
            
            
            ######################################################
            ## create Synapses
            ######################################################
            #
            Syn_1 = Synapses(neuron,neuron2,
                            model= eq_1_plastAMPA,
                            on_pre = eq_2_plastAMPA,
                            method='heun'
                            )
            #                
            #                
            Syn_4 = Synapses(neuron2,neuron,
                            model= eq_1_plastAMPA,
                            on_pre = eq_2_plastAMPA,
                            method='heun'
                            )
            #
            for rr in range(NrSyn):
                    Syn_1.connect(i=0,j=neuron2[compv1[int(syn_loc1[rr])]:compv1[int(syn_loc1[rr])]+1])
            set_init_syn(Syn_1,init_weight)
            #
            #
            for rr in range(NrSyn):
                    Syn_4.connect(i=0,j=neuron[compv2[int(syn_loc2[rr])]:compv2[int(syn_loc2[rr])]+1])
            set_init_syn(Syn_4,init_weight)
            #
            print('Synapses created...')
            # 
            
            #####################################################
            # Monitors
            #####################################################
            #M_soma = StateMonitor(neuron.main, ['v'], record=True)#, dt = 25*defaultclock.dt)
            #M_soma2 = StateMonitor(neuron2.main, ['v'], record=True)#, dt = 25*defaultclock.dt)
            
            ##M_in = StateMonitor(N_input, ['v'], record=True)#, dt = 25*defaultclock.dt)
            #
            #M_dend = StateMonitor(neuron[compv[0]:compv[0]+1], ['v'], record=True)
            #M_dend2 = StateMonitor(neuron2[compv[0]:compv[0]+1], ['v'], record=True)#, dt = 25*defaultclock.dt)
            
            #M_weights = StateMonitor(Syn_1,['wampa'],record=True, dt = 400*defaultclock.dt)
            #M_weights2 = StateMonitor(Syn_4,['wampa'],record=True, dt = 400*defaultclock.dt)
            
            #####################################################
            # Run
            #####################################################
            neuron.main.rate_v = 0*Hz
            neuron2.main.rate_v = 0*Hz
            run(500*ms)
            
            neuron.main.rate_v = rate_1*Hz
            neuron2.main.rate_v = rate_2*Hz
            print('Simulating...'+str(rr1)+' '+str(rr2))        
            run(t_max)
            
            
            print('Simulation finished!')
            
            storew[rr1,rr2,:NrSyn] = Syn_1.wampa[:]
            storew[rr1,rr2,NrSyn:] = Syn_4.wampa[:]
            
            
            fname = loc1+'_'+syn_arrange1+'_'+namestr # 'testfile' #loc1+'_'+syn_arrange1+'_'+saveloc1+saveloc2+str(zzz)
            
#            data12 = open('ConnectData/AMPA_axonH2_'+str(NrSyn)+'_'+fname+'_LoHi.txt','w')
#            p1 = storew[:,:,:NrSyn]
#            json.dump(p1.tolist(),data12)
#            data12.close()
#            
#            data1 = open('ConnectData/AMPA_axonH2_'+str(NrSyn)+'_'+fname+'_HiLo.txt','w')
#            p2 = storew[:,:,NrSyn:]
#            json.dump(p2.tolist(),data1)
#            data1.close()
#            
#            data13 = open('ConnectData/AMPA_axonH2_'+str(NrSyn)+'_'+fname+'_loc1.txt','w')
#            p3 = syn_loc1
#            json.dump(p3.tolist(),data13)
#            data13.close()
#            
#            data14 = open('ConnectData/AMPA_axonH2_'+str(NrSyn)+'_'+fname+'_loc2.txt','w')
#            p4 = syn_loc2
#            json.dump(p4.tolist(),data14)
#            data14.close()
    
    #####################################################
    # Plot
    #####################################################
    
    #plt.figure()
    #plt.pcolor(storew[:,:,0])
    #plt.colorbar()
    #plt.xticks(np.arange(0.5,len(ratesv),1),ratesv)
    #plt.yticks(np.arange(0.5,len(ratesv),1),ratesv)
    #plt.ylabel('pre (low)')
    #plt.xlabel('post (high)')
    #plt.title('low-high '+str(compv[0]))
    #
    #plt.figure()
    #plt.pcolor(storew[:,:,1])
    #plt.colorbar()
    #plt.xticks(np.arange(0.5,len(ratesv),1),ratesv)
    #plt.yticks(np.arange(0.5,len(ratesv),1),ratesv)
    #plt.ylabel('post (low)')
    #plt.xlabel('pre (high)')
    #plt.title('high-low '+str(compv[0]))
    #
    #plt.figure()
    #plt.pcolor(1*(storew[:,:,1]>.5)+2*(storew[:,:,0]>.5))
    #plt.colorbar()
    #plt.xticks(np.arange(0.5,len(ratesv),1),ratesv)
    #plt.yticks(np.arange(0.5,len(ratesv),1),ratesv)
    #plt.title('uni vs bidirectional '+str(compv[0]))
    
    
    
    #np.savetxt('ConnectData/AckerCONN'+str(NrSyn)+'_'+fname+'_LoHi.dat',storew[:,:,:NrSyn])
    #np.savetxt('ConnectData/AckerCONN'+str(NrSyn)+'_'+fname+'_HiLo.dat',storew[:,:,NrSyn:])
    
    #Mdend_max = np.amax(M_dend.v/mV,axis=0)
    
    #figure(figsize=(10, 4))
    ##plot(M_soma.t/ms, M_soma.v[0]/mV)
    #plot(M_dend.t/ms, M_dend.v[0]/mV)
    #plot(M_soma2.t/ms, M_soma2.v[0]/mV)
    #plot(M_dend2.t/ms, M_dend2.v[0]/mV)
    
    
    #figure(figsize=(10, 4))
    #plot(M_weights.t/ms, M_weights.wampa[0], linewidth=2)
    #plot(M_weights2.t/ms, M_weights2.wampa[0], linewidth=2)