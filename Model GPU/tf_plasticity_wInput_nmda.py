import tensorflow as tf


def tf_plasticity_wInput_nmda(w_min,w_max,NrON_0,NrON_1,Nr_D,Nrn_trace3,volt_S4,volt_D4,Nrn_u_lp2,inputX2,theta_Low,Homeostat_target,
                  inputSpks2,theta_High,LTPfact,A_LTP,A_LTD,timeSteptf,PlasticW,Synapses_out,Conn_boolean,u_delay):

    #----------
    #hard bounds matrices
    #----------
    Bnds_min = tf.scalar_mul( w_min, tf.ones((NrON_0,NrON_1,2,Nr_D))) 
    Bnds_mx = tf.scalar_mul( w_max,tf.ones((NrON_0,NrON_1,2,Nr_D)))
    
    #----------
    # traces
    #----------
    Homeostat_volt = tf.expand_dims(tf.expand_dims(Nrn_trace3[:,0],1),2)
    u_m1 = Nrn_u_lp2[:,0:1,:] 
    u = tf.identity(volt_D4)
    u_m2 = Nrn_u_lp2[:,2:3,:] 
    
    if NrON_0>NrON_1:
        x_trace = tf.expand_dims(tf.concat([ inputX2, Nrn_trace3[:,1]],0),1)
    else:
        x_trace = tf.expand_dims(Nrn_trace3[:,1],1)
    
    #----------
    #Calculate LTD matrix
    #----------
    u_dep_post = tf.multiply(tf.to_float(tf.greater(u_m1,theta_Low)) ,tf.subtract(u_m1,theta_Low))
    
    ### only homeostasis if target value is not 0 ###
    u_homeo_m = tf.tile(tf.multiply(Homeostat_volt, tf.multiply( 1/Homeostat_target,Homeostat_volt)),[1,2,Nr_D])
    
    u_dep_post2 = tf.add(u_dep_post,
                        tf.multiply(tf.to_float(tf.greater(Homeostat_target,0)),tf.subtract(tf.multiply(u_dep_post,u_homeo_m),u_dep_post)))
    
    pre_syn_fire = tf.expand_dims(tf.concat([inputSpks2,tf.to_float(tf.equal(Nrn_trace3[:,3], 1))],0) ,1)
    LTD_matrix = tf.reshape(tf.matmul(pre_syn_fire,tf.reshape(u_dep_post2,[1,2*NrON_1*Nr_D])),[NrON_0,NrON_1,2,Nr_D]) 
    LTD = tf.multiply(-1*A_LTD,LTD_matrix)  #prox and dist
    
    #----------
    #Calculate LTP matrix
    #----------
    u_spike = tf.multiply(tf.to_float(tf.greater(u,theta_High)) ,tf.subtract(u,theta_High))
    
    u_m2_rec = tf.multiply(tf.to_float(tf.greater(u_m2,theta_Low) ),tf.subtract(u_m2,theta_Low))
    
    u_pot_post = tf.multiply(u_m2_rec,u_spike)
    
    LTP_matrix_1 = tf.reshape(tf.matmul(x_trace,tf.reshape(u_pot_post,[1,2*NrON_1*Nr_D])),[NrON_0,NrON_1,2,Nr_D])
    LTP_matrix_prox = LTP_matrix_1[:,:,0,None,:]
    
    distvv = tf.reshape(tf.expand_dims(u_delay[:,1,:,-40],1),[1,NrON_1,1,Nr_D])
    distvv2 = tf.to_float(tf.greater(tf.tile(distvv,[NrON_0,1,1,1]),-15))
    LTP_matrix_dist = LTP_matrix_1[:,:,1,None,:] - (1-LTPfact)*tf.multiply(LTP_matrix_1[:,:,1,None,:],distvv2)
    
    LTP_matrix = tf.concat([LTP_matrix_prox,LTP_matrix_dist],2)
    LTP = tf.multiply(A_LTP*timeSteptf,LTP_matrix)  #prox and dist
    
    # weight changes
    Syn = tf.add(tf.multiply(tf.add(LTP,LTD),PlasticW),Synapses_out)
    
    # check for hard bounds on weights
    Syn2 = tf.multiply(tf.minimum(tf.maximum(Syn,Bnds_min),Bnds_mx),Conn_boolean)
    
    
    return Syn2