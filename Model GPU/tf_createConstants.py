import tensorflow as tf

def tf_createConstants(timeStep,EPSC_par,Neuron_par,STDP_par,plasticWinp,Synapses,LTPfact_inp):
    

    onestep = tf.constant(1.0)
    timeSteptf = tf.constant(timeStep)
    
    #Conduct
    AMPA_g = tf.constant(EPSC_par[8], dtype= tf.float32) #AMPA max conductance
    AMPA_t = tf.constant(EPSC_par[4], dtype= tf.float32) #AMPA tconst
    NMDA_g = tf.constant(EPSC_par[9], dtype= tf.float32) # NMDA max conductance
    NMDA_t = tf.constant(EPSC_par[5], dtype= tf.float32) # AMPA tconst    
       
    
    #Neuron 
    
    E_L = tf.constant(Neuron_par[0], dtype= tf.float32 ) # resting potential
    g_L = tf.constant(Neuron_par[1], dtype= tf.float32 ) # leak conductance
    C = tf.constant(Neuron_par[2], dtype= tf.float32)  # membrane capacitance
    uM_Tconst = tf.constant(Neuron_par[6], dtype= tf.float32)   # time constant for the low-pass-filtered membrane potential
    uP_Tconst = tf.constant(Neuron_par[7], dtype= tf.float32)   # time constant for the low-pass-filtered membrane potential
    x_tau = tf.constant(Neuron_par[8], dtype= tf.float32)  #timeconstant for spiketrace
    x_reset = tf.constant(Neuron_par[9]/Neuron_par[8], dtype= tf.float32) # reset value for spiketrace
    delta_T = tf.constant(Neuron_par[11], dtype= tf.float32 ) #slope factor
    V_Trest = tf.constant(Neuron_par[12], dtype= tf.float32)  # threshold rest value
    V_T_tconst = tf.constant(Neuron_par[14], dtype= tf.float32 )                   # threshold potential
    Noise_tau = tf.constant(Neuron_par[16] , dtype= tf.float32 )                  # timeconstant of noise
    Noise_avg = tf.constant(Neuron_par[17], dtype= tf.float32 )                   # mean of noise
    Noise_std = tf.constant(Neuron_par[18] , dtype= tf.float32 )                  # std of noise
    Homeostat_tau = tf.constant(Neuron_par[19], dtype= tf.float32)  #timeconstant for homeostasis    
    E_AMPA = tf.constant(EPSC_par[0] , dtype= tf.float32) # ampa reversal potential
    E_NMDA = tf.constant(EPSC_par[1], dtype= tf.float32)  # nmda reversal potential
    prox_att = tf.constant(EPSC_par[6], dtype= tf.float32)  # proximal to soma attenuation
    dist_att = tf.constant(EPSC_par[7], dtype= tf.float32)  # dist to soma attenuation
    prox_Backprop = tf.constant(0.99, dtype= tf.float32 ) # backpropagation of I_ext, I_noise and I_z
    dist_Backprop = tf.constant(0.95, dtype= tf.float32)  # backpropagation of I_ext, I_noise and I_z 
    spineFactor = tf.constant(1.5, dtype= tf.float32) 
    
    #Spikes
    V_Tjump = tf.constant(Neuron_par[13], dtype= tf.float32  )                  # threshold potential
    thresh = tf.constant(Neuron_par[3] , dtype= tf.float32  )                   # thresh = spiking threshold     
    spike_width_soma = tf.constant(Neuron_par[20], dtype= tf.float32)  # spike width soma [ms]
    spike_width_dend = tf.constant(Neuron_par[21], dtype= tf.float32)  # spike width dendrite [ms]
    spike_latency = tf.constant(Neuron_par[22], dtype= tf.float32)  # latency of the spike in dendrites [ms]
    refrac_time = tf.constant(Neuron_par[23], dtype= tf.float32)  #absolute refractory time (total abs. refractory time = spike_width_dend+spike_latency+refrac_time) [ms]
    v_reset = tf.constant(Neuron_par[24], dtype= tf.float32)  # reset voltage after spike [mV]
#     delay = Neuron_par(16)                  #delay between spike and EPSP onset

    
    #Plastic
    A_LTD = tf.constant(STDP_par[0], dtype= tf.float32 )
    A_LTP = tf.constant(STDP_par[1], dtype= tf.float32 )   
    theta_Low = tf.constant(STDP_par[2], dtype= tf.float32) 
    theta_High = tf.constant(STDP_par[3], dtype= tf.float32)     
    w_max = tf.constant(STDP_par[4], dtype= tf.float32) 
    w_min = tf.constant(STDP_par[5], dtype= tf.float32)     
    Homeostat_target = tf.constant(STDP_par[6] , dtype= tf.float32)
    LTPfact = tf.constant(LTPfact_inp,dtype=tf.float32)
    
    PlasticW = tf.constant(plasticWinp,
                               dtype= tf.float32,  name='plasticW')
    Conn_boolean = tf.constant(Synapses>0,
                           dtype= tf.float32,  name='conn_boolean')
    
    return (onestep,timeSteptf,AMPA_g,AMPA_t,NMDA_g,NMDA_t,LTPfact,
            E_L,g_L,C,uM_Tconst,uP_Tconst,x_tau,x_reset,delta_T,V_Trest,V_T_tconst,Noise_tau,Noise_avg,
            Noise_std,Homeostat_tau,E_AMPA,E_NMDA,prox_att,dist_att,prox_Backprop,dist_Backprop,spineFactor,
            V_Tjump,thresh,spike_width_soma,spike_width_dend,spike_latency,refrac_time,v_reset,A_LTD,A_LTP,
            theta_Low,theta_High,w_max,w_min,Homeostat_target,PlasticW,Conn_boolean)