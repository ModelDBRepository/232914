import tensorflow as tf


def tf_conductance_wInput_nmdaFix(Conductances,Nrn_trace,NrON_1,Nr_D,inputSpks2,Synapses_out,AMPA_g,AMPA_t,NMDA_g,NMDA_t,timeSteptf,nmdaw):


    g_AMPA = Conductances[:,0:2,:] #AMPA conductances
    g_NMDA = Conductances[:,2:4,:] #NMDA conductances
    
    ### only presynaptic neurons that fire a spike 
    Pre_spikes_ext = tf.tile(tf.expand_dims(tf.expand_dims( tf.concat([inputSpks2,tf.to_float(tf.equal(Nrn_trace[:,3], 1))],0) ,1),2),[1,NrON_1,Nr_D])
    
    ### total presyn weight of cells that just fired 
    prox_pre_weights = tf.reshape(tf.reduce_sum(tf.multiply(Pre_spikes_ext,Synapses_out[:,:,0,:]),0),(NrON_1,1,Nr_D))
    dist_pre_weights = tf.reshape(tf.reduce_sum(tf.multiply(Pre_spikes_ext,Synapses_out[:,:,1,:]),0),(NrON_1,1,Nr_D))
    
    tot_w = tf.concat([prox_pre_weights, dist_pre_weights],1)
    tot_w_nmda = tf.multiply(tf.to_float(tf.greater(tot_w,0)),nmdaw)
    
    #-----------------
    # Conductances
    #-----------------      
    ### AMPA conductances 
    g_AMPA = tf.subtract(tf.add(g_AMPA, tf.multiply(tot_w,tf.to_float(AMPA_g))) , (timeSteptf)*g_AMPA/AMPA_t ) #.*repmat(Nrn_traces(:,4)==0,[1,2,Nr_D]);
    ### NMDA conductance 
    g_NMDA = tf.subtract(tf.add(g_NMDA, tf.multiply(tot_w_nmda,tf.to_float(NMDA_g))) , (timeSteptf)*g_NMDA/NMDA_t ) #.*repmat(Nrn_traces(:,4)==0,[1,2,Nr_D]);
    
    Conduct2 = tf.concat([g_AMPA, g_NMDA],1)    
    
    
    return Conduct2