
from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import json

import sys
sys.path.append('../Model CPU/')
sys.path.append('../Model GPU/')

from f_neurons import f_neurons
from f_parameters import f_parameters
from f_generateInputs3 import f_generateInputs3

#from tf_conductance_wInput import tf_conductance_wInput
from tf_conductance_wInput_nmdaFix import tf_conductance_wInput_nmdaFix
from tf_voltage_SpecInhib2 import tf_voltage_SpecInhib2
from tf_spikes import tf_spikes
#from tf_plasticity_wInput import tf_plasticity_wInput
from tf_plasticity_wInput_nmda import tf_plasticity_wInput_nmda
from tf_createVariables import tf_createVariables
from tf_createConstants import tf_createConstants

#from f_instRate import f_instRate
#from f_Inpt_x import f_Inpt_x
from f_instWeights import f_instWeights
from f_plotWeights_subplots import f_plotWeights_subplots

gpu_options = tf.GPUOptions(allow_growth=True) #per_process_gpu_memory_fraction=.95
device = "/gpu:0" # /cpu:0

#######################
# Simulation Parameters
#######################
NrScnds = 100#15

storeV = 1

maxWeight = 1 
minWeight = 0.01*maxWeight 
startWeight = maxWeight 
nmdaw = tf.constant(.5, dtype = tf.float32) 

LTPfact_inp = .15 #0.2 #factor for distal LTP

NrFeatures = 4  # number of features
Nrn_perFeat = 10  # number of neurons per feature
NrON = NrFeatures*Nrn_perFeat  # total nr of recurrent neurons

NrIn_perFeat = 50  # nr of inputs per feature
Nr_InpSynapses = NrFeatures*NrIn_perFeat  # total nr of input synapses

Nr_D = 15 #D_for_inputs+D_for_reccur 
nrInpts = 10

rate = 350 
inp_time = 10 
buffertime = 200 

inh_tc = 30
Ipow = 1
Itc2 = 2

##########################
# Import model parameters
##########################
timeStep,STDP_par, Neuron_par, EPSC_par = f_parameters()  
STDP_par[0] = STDP_par[0]  # ALTD
STDP_par[1] = STDP_par[1]  # ALTP
Neuron_par[4] = Nr_D 
Neuron_par[17] = 150 #noise avg
Neuron_par[18] = 15 #noise STD
STDP_par[4] = maxWeight 
STDP_par[5] = minWeight 

t_max = NrScnds*1000      

gamma_value = 120     

#########################
# Initialize for data storage
########################## 
saveVar = 1000 #150 #[ms]
Plot_var = Neuron_par[0]*np.ones((16,int(np.ceil(t_max/(saveVar)))+2))   #-1
Spikes_var = np.zeros((NrON,int(np.ceil(t_max/(saveVar)))+2))   #-1
instWeight = np.zeros(9)
        
##########################
# Import neurons
##########################
S_potential, D_potential, Nrn_u_delay, Nrn_u_lowpass, Nrn_traces, Nrn_conductance = f_neurons( NrON , Neuron_par, timeStep) 

####################
#Create input spike train
#####################

t_max2 = inp_time+buffertime 
inp_steps = int(t_max2/timeStep)
#itervv = np.int(np.ceil(t_max/t_max2))


#################################
# Define Synaptic Connections
#################################
#Synapses = np.zeros((NrON,NrON,2,Nr_D) )
#Synapses[1,0,0,0] = 1
#Synapses[0,1,0,1] = 1



def store_data(Synapses, Plot_var,NrScnds,stepv):
    data12 = open('data/fin_AMPA_p15_MemNWnew_Struct_'+str(gamma_value)+'_'+str(NrScnds)+'_Syn'+str(stepv)+'.txt','w')
    json.dump(Synapses.tolist(),data12)
    data12.close()
    data1 = open('data/fin_AMPA_p15_MemNWnew_Struct_'+str(gamma_value)+'_'+str(NrScnds)+'_iW.txt','w')
    json.dump(Plot_var[0:3,:].tolist(),data1)
    data1.close()

storeVar = 20000


#==== Initialise ====
Synapses = np.zeros((Nr_InpSynapses+NrON,NrON,2,Nr_D))

#==== Inputs ====
A = np.zeros((NrIn_perFeat,NrON,2,Nr_D))
A[:,:,1,0] = np.ones((NrIn_perFeat,NrON))

for dd in np.arange(NrON):
    for ii in np.arange(NrIn_perFeat):
        A[ii,dd,:,:] = np.transpose(A[ii,dd,:,np.random.permutation(np.size(A,axis=3))])  # permute over dendrites
        A[ii,dd,:,:] = A[ii,dd,np.random.permutation(np.size(A,axis=2)),:]  # permute over dist/prox


for nn in np.arange(NrFeatures):
    Synapses[nn*NrIn_perFeat:(nn+1)*NrIn_perFeat,nn*Nrn_perFeat:(nn+1)*Nrn_perFeat,:,:] =  (
        maxWeight*A[:,nn*Nrn_perFeat:(nn+1)*Nrn_perFeat,:,:])


#==== Recurrent ====
A = np.zeros((NrON,NrON,2,Nr_D))
   
for dd in np.arange(NrON):
    seqrand = []
    for ee in range(NrFeatures):
        seqv = np.array(range(Nrn_perFeat))
        seqrand = np.append(seqrand,np.random.permutation(seqv))
    for ii in np.arange(NrON):
        A[ii,dd,1*(seqrand[ii]<int(.5*Nrn_perFeat)),int(np.floor(ii/Nrn_perFeat))] = 1*(ii!=dd)  # permute over dist/prox
        
for dd in np.arange(NrON):
    for ii in np.arange(NrON):
#        A[ii,dd,:,:] = A[ii,dd,np.random.permutation(np.size(A,axis=2)),:]  # permute over dist/prox
        A[ii,dd,0,:] = np.transpose(A[ii,dd,0,np.random.permutation(np.size(A,axis=3))])  # permute over dendrites prox


Synapses[Nr_InpSynapses:Nr_InpSynapses+NrON,0:NrON,:,:] = maxWeight*A 




PlasticW1 = np.zeros((Nr_InpSynapses,NrON,2,Nr_D))
PlasticW2 = Synapses[Nr_InpSynapses:Nr_InpSynapses+NrON,:,:,:]>0 
plasticWinp = np.append(PlasticW1,PlasticW2,axis=0)


NrON_0 = np.size(Synapses,axis=0)
NrON_1 = np.size(Synapses,axis=1)
input_x = np.zeros(NrON_0-NrON_1)

##################
# gpu 
##################


#sess = tf.Session() 

sess = tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        gpu_options=gpu_options,
        ))
  
    
with tf.device(device):
       
    
    #######################
    # Variables
    #######################
    (spk_gen,volt_S,u_delay,Nrn_trace,Nrn_u_lp,
     volt_D,Conductances,Synapses_out,Inh_feedb,
     inputX,sim_index) = tf_createVariables(NrON_1,NrON_0,S_potential,Nrn_u_delay,
                        Nrn_traces,Nrn_u_lowpass,D_potential,
                       Nrn_conductance,Synapses,input_x)
    
    Inh_feedb4 = tf.Variable(tf.to_float(np.zeros(NrON)))
    inh_tc2 = tf.constant(Itc2, dtype = tf.float32) 
    ##### extra variables
    spk_train = tf.placeholder(tf.float32, shape=(NrON_0,inp_steps), name='spk_train')
    inputSpks = tf.Variable(tf.zeros((NrON_0,inp_steps)),name='inputspikes')
    isteps = tf.constant(inp_steps, dtype= tf.int32)
    
    #######################
    # Constants
    #######################
    (onestep,timeSteptf,AMPA_g,AMPA_t,NMDA_g,NMDA_t,LTPfact,
            E_L,g_L,C,uM_Tconst,uP_Tconst,x_tau,x_reset,delta_T,V_Trest,V_T_tconst,Noise_tau,Noise_avg,
            Noise_std,Homeostat_tau,E_AMPA,E_NMDA,prox_att,dist_att,prox_Backprop,dist_Backprop,spineFactor,
            V_Tjump,thresh,spike_width_soma,spike_width_dend,spike_latency,refrac_time,v_reset,A_LTD,A_LTP,
            theta_Low,theta_High,w_max,w_min,Homeostat_target,PlasticW,Conn_boolean) = tf_createConstants(timeStep,EPSC_par,
                                                                Neuron_par,STDP_par,
                                                                plasticWinp,Synapses,LTPfact_inp)
                                                                
                 
    
    # constants for  INHIBITION 
    Inh_tau = tf.constant(inh_tc, dtype = tf.float32) 
#    alphav = tf.constant(5*1.5,dtype=tf.float32)
    betav =  tf.Variable(tf.to_float(0))#,dtype=tf.float32)
    gammav =  tf.constant(gamma_value,dtype=tf.float32)
    E_GABA =  tf.constant(-70,dtype=tf.float32)
    inh_amp = tf.constant(15000,dtype=tf.float32)
    inh_power = tf.constant(Ipow,dtype=tf.float32)
    ifeedb = tf.Variable(tf.to_float(np.zeros(NrON)))
    
    
    ####################
    #Evolution of input spike trace
    #####################
    inputSpks2 = inputSpks[0:Nr_InpSynapses,tf.mod(tf.to_int32(sim_index),isteps)]
    inputX2 = tf.add(tf.subtract(inputX,tf.multiply((timeSteptf/x_tau),inputX)), tf.multiply(inputSpks2,x_reset))
    
    ##############################
    # Conductances
    ##############################    
    Conduct2 = tf_conductance_wInput_nmdaFix(Conductances,Nrn_trace,NrON_1,Nr_D,inputSpks2,Synapses_out,AMPA_g,AMPA_t,NMDA_g,NMDA_t,timeSteptf,nmdaw)
        
    ##############################
    # Voltage
    ##############################  
    volt_S2,volt_D2,Nrn_trace2,Nrn_u_lp2,u_delay2,Inh_feedb2,Inh_feedb3 = tf_voltage_SpecInhib2(timeSteptf,Nrn_trace,g_L,volt_S,E_L,NrON_1,
               spineFactor,Nr_D,volt_D,Conduct2,E_AMPA,dist_att,E_NMDA,delta_T,
               Nrn_u_lp,uM_Tconst,uP_Tconst,u_delay,Homeostat_tau,ifeedb,inh_amp,E_GABA,Inh_tau,C,betav,gammav,
               inh_power,x_reset,x_tau,V_T_tconst,V_Trest,Noise_tau,Noise_avg,Noise_std,
               Conn_boolean[Nr_InpSynapses:,:,:,:],Inh_feedb4,inh_tc2)
#    Synapses_out
    ##############################
    # SPIKES
    ##############################    
    volt_S4,volt_D4,Nrn_trace3 = tf_spikes(timeSteptf,NrON_1,Nr_D,Nrn_trace2,volt_S2,spike_width_dend,spike_width_soma,v_reset,volt_D2,
              u_delay2,V_Tjump,V_Trest,spike_latency,thresh,refrac_time)
    
    ##############################
    # Plasticity
    ##############################
#    Syn2 = tf_plasticity_wInput(w_min,w_max,NrON_0,NrON_1,Nr_D,Nrn_trace3,volt_S4,volt_D4,Nrn_u_lp2,inputX2,theta_Low,Homeostat_target,
#                  inputSpks2,theta_High,LTPfact,A_LTP,A_LTD,timeSteptf,PlasticW,Synapses_out,Conn_boolean)
    
    Syn2 = tf_plasticity_wInput_nmda(w_min,w_max,NrON_0,NrON_1,Nr_D,Nrn_trace3,volt_S4,volt_D4,Nrn_u_lp2,inputX2,theta_Low,Homeostat_target,
                  inputSpks2,theta_High,LTPfact,A_LTP,A_LTD,timeSteptf,PlasticW,Synapses_out,Conn_boolean,u_delay2)
    
    ####################
    #Evoke spikes
    ##################### 
    assign_op = tf.group(volt_S.assign(spk_gen))
                    
    assign_sTrain = tf.group(inputSpks.assign(spk_train))
         
                    
    ####################
    #ITERATION
    #####################
    step = tf.group(
                    sim_index.assign(tf.add(sim_index, onestep)),
                    Synapses_out.assign(Syn2),
                    ifeedb.assign(Inh_feedb2),
                    Inh_feedb4.assign(Inh_feedb3),
                    volt_S.assign(volt_S4),
                    volt_D.assign(volt_D4),
                    u_delay.assign(u_delay2),
                    Nrn_u_lp.assign(Nrn_u_lp2),
                    Nrn_trace.assign(Nrn_trace3), 
                    Conductances.assign(Conduct2),
                    inputX.assign(inputX2)
                    )
                        
                   
                    

    ####################
    #INITIALISE
    #####################
    sess.run(tf.global_variables_initializer())
    
        
    
    
    
    ####################
    #INTEGRATION
    #####################
    print('START '+device[1:4]+' ...')
    saveIndex = 0
    
    for t in np.arange( timeStep,t_max,timeStep):
        
      
          ### Add spikes manually ####
#        if np.abs(t-10)<timeStep/2:
#            S_potential[1] = Neuron_par[3]+1 
#            sess.run(assign_op,feed_dict={spk_gen:S_potential})
#        if np.abs(t-30)<timeStep/2:
#            S_potential[0] = Neuron_par[3]+1 
#            sess.run(assign_op,feed_dict={spk_gen:S_potential})
      
          ### Generate inputs ####
        if int(1000*(t-timeStep))%int(1000*t_max2) < int(1000*timeStep):
            FeatureProbabilities = [.25,.25,.25]
            ispks,ftr = f_generateInputs3(1,buffertime,inp_time,rate,NrON,NrIn_perFeat,NrFeatures,timeStep,FeatureProbabilities)
            sess.run(assign_sTrain,feed_dict={spk_train:ispks})
    
        sess.run([step])
        
        if int(t*1000)%int(storeVar*1000) < .9*int(1000*timeStep):
            store_data(Synapses, Plot_var[0:3,:],NrScnds,saveIndex)
        
        if storeV:
            if int(t*1000)%int(saveVar*1000) < .9*int(1000*timeStep):    
#                print(t)
#                print(t%saveVar)
                ### Pull Data from gpu ###
                Synapses = Synapses_out.eval(session=sess)
#                S_potential = volt_S.eval(session=sess)
#                D_potential = volt_D.eval(session=sess)
    #            Nrn_conductance = Conductances.eval(session=sess)
    #            Nrn_u_delay = u_delay.eval(session=sess)
    #            Nrn_u_lowpass = Nrn_u_lp.eval(session=sess)
#                Nrn_traces = Nrn_trace.eval(session=sess)
    #            input_x = inputX.eval(session=sess)
                
                ### Process Data ###
                rec_Synapses = Synapses[Nr_InpSynapses:Nr_InpSynapses+NrON,0:NrON,:,:]
                
                instWeight[0]  = f_instWeights(rec_Synapses, 
                      np.arange(0,4*Nrn_perFeat),np.arange(0,4*Nrn_perFeat),0) 
                instWeight[1]  = f_instWeights(rec_Synapses, 
                      np.arange(0,4*Nrn_perFeat),np.arange(0,4*Nrn_perFeat),1) 
                instWeight[2]  = f_instWeights(rec_Synapses, 
                      np.arange(0,4*Nrn_perFeat),np.arange(0,4*Nrn_perFeat),np.arange(2)) 
                
                ### Save Data ###
                Plot_var[0,saveIndex] =  instWeight[0]
                Plot_var[1,saveIndex] =  instWeight[1]
                Plot_var[2,saveIndex] =  instWeight[2]
                
#                Spikes_var[:,saveIndex] =  np.ones(NrON)*(Nrn_traces[:,3]==1)
#                
#                Plot_var[6,saveIndex] =  D_potential[1,1,1]#
#                Plot_var[7,saveIndex] =  S_potential[1] 
#                Plot_var[8,saveIndex] =  S_potential[2]#D_potential[64,1,0]#D_potential[4*Nrn_perFeat,1,0]
#                Plot_var[9,saveIndex] =  D_potential[2,1,2]#S_potential[4*Nrn_perFeat] 
#                Plot_var[10,saveIndex] =  D_potential[3,1,3]
#                Plot_var[11,saveIndex] =  S_potential[5] 
#                Plot_var[12,saveIndex] =  S_potential[6]
#                Plot_var[13,saveIndex] =  S_potential[11] 
#                Plot_var[14,saveIndex] =  S_potential[21]
#                Plot_var[15,saveIndex] =  S_potential[31] 
                
                saveIndex = saveIndex + 1    
        
            
          ##################################
          # Display the progress
          ###################################
        if np.abs(t-t_max*0.2*np.round(5*t/t_max)) < 0.5*timeStep:
            print('> '+str(np.round(100*t/t_max))+'%') 

#===========================================
#
Synapses = Synapses_out.eval(session=sess)      

sess.close()
print('it works!') 

if storeV:
    xvalues = np.arange(0,saveIndex)
    
    plt.figure()
    plt.plot(xvalues,Plot_var[0,:saveIndex],label='prox')
    plt.plot(xvalues,Plot_var[1,:saveIndex],label='dist')
    plt.legend()
    
#    plt.figure()
#    for iii in range(NrON):
#        plt.plot(xvalues,iii*Spikes_var[iii,:saveIndex],'.')
#    plt.legend()
#    
#    plt.figure()
#    for jjj in range(10):
#        plt.plot(xvalues,Plot_var[6+jjj,:saveIndex])
#    plt.plot(xvalues,Plot_var[15,:saveIndex])

Synapses_resh2 = np.sum(Synapses[Nr_InpSynapses:Nr_InpSynapses+NrON,0:NrON,:,:],axis=3) 
Synapses_resh3 = np.zeros((2*NrON,NrON))
Synapses_resh3[0:NrON,0:NrON] = Synapses_resh2[:,:,0] 
Synapses_resh3[NrON:2*NrON,0:NrON] = Synapses_resh2[:,:,1] 
f_plotWeights_subplots(Synapses_resh3, NrON,maxWeight)

store_data(Synapses, Plot_var[0:3,:],NrScnds,saveIndex)