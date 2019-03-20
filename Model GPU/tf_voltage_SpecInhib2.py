import tensorflow as tf


def tf_voltage_SpecInhib2(timeSteptf,Nrn_trace,g_L,volt_S,E_L,NrON_1,spineFactor,Nr_D,volt_D,Conduct2,E_AMPA,dist_att,E_NMDA,delta_T,
               Nrn_u_lp,uM_Tconst,uP_Tconst,u_delay,Homeostat_tau,Inh_feedb,inh_amp,E_GABA,Inh_tau,C,betav,gammav,
               inh_power,x_reset,x_tau,V_T_tconst,V_Trest,Noise_tau,Noise_avg,Noise_std,Synapses,Inh_feedb4,inh_tc2):

    #------------
    # Currents
    #------------
    
    ### external current ###
    I_ext = Nrn_trace[:,4]   ## soma
#     I_ext_D = repmat([prox_Backprop*Nrn_trace(:,5) dist_Backprop*Nrn_trace(:,5)],1,1,Nr_D)  ## dendrites
    
    ### leak current ###
    I_L = g_L*tf.subtract(volt_S,E_L*tf.ones(NrON_1))  ## soma
    I_L_D = tf.multiply(tf.tile(tf.expand_dims([[spineFactor*g_L,spineFactor*g_L]],2),[NrON_1,1,Nr_D]),tf.subtract(volt_D,E_L*tf.ones((NrON_1,2,Nr_D))))  ## dendrites
           
    ### Noise ###
    I_noise = Nrn_trace[:,5]  ## soma
#     I_noise_D = repmat([prox_Backprop*Nrn_trace(:,6) dist_Backprop*Nrn_trace(:,6)],1,1,Nr_D)  ## dendrites
    
    ### AMPA current ###
    I_prox_ampa = tf.scalar_mul(-1,tf.multiply(Conduct2[:,0,None,:],tf.subtract(volt_D[:,0,None,:],E_AMPA*tf.ones((NrON_1,1,Nr_D))) ))
    I_dist_ampa = tf.multiply(-1*(1/dist_att),tf.multiply(Conduct2[:,1,None,:],tf.subtract(volt_D[:,1,None,:],E_AMPA*tf.ones((NrON_1,1,Nr_D))) ))
        
    I_AMPA = tf.concat([I_prox_ampa,I_dist_ampa],1) 
    
    ### NMDA current ###
    corrf1 = tf.constant(100, dtype= tf.float32)
    corrf2 = tf.constant(2, dtype= tf.float32)
    I_prox_nmda = tf.scalar_mul(-1,tf.multiply(Conduct2[:,2,None,:],tf.div(tf.subtract(volt_D[:,0,None,:],E_NMDA*tf.ones((NrON_1,1,Nr_D))),(tf.add(tf.to_float(1),tf.div(tf.exp(tf.multiply(-0.065,volt_D[:,0,None,:])),3.57))) )))
    I_dist_nmda = tf.multiply(-1*(1/dist_att),tf.multiply(Conduct2[:,3,None,:],tf.div(tf.subtract(volt_D[:,1,None,:],E_NMDA*tf.ones((NrON_1,1,Nr_D))),(tf.add(tf.to_float(1),tf.div(tf.exp(tf.multiply(-0.065,volt_D[:,1,None,:])),3.57))) )))
        
    I_NMDA = tf.concat([I_prox_nmda, I_dist_nmda],1)
        
    ### Exponential term (fast Na activation in soma) ###
    I_exp = tf.scalar_mul(g_L*delta_T,tf.exp(tf.div(tf.subtract(volt_S,Nrn_trace[:,2]),delta_T)) )
    
    ### Synacptic current ###
    I_syn_dend = tf.add(I_AMPA,I_NMDA) 
    
    ### Dendrites to soma current ###
    
    I_soma_prox =  2500*tf.multiply(tf.subtract(tf.tile(tf.expand_dims(tf.expand_dims(volt_S,1),2),[1,1,Nr_D]),
        volt_D[:,0,None,:]),tf.tile(tf.expand_dims(tf.expand_dims(tf.to_float(tf.equal(Nrn_trace[:,3],0)),1),2),
        [1,1,Nr_D])) 
        
    I_prox_dist =  1500*tf.subtract(volt_D[:,0,None,:],volt_D[:,1,None,:]) 
    backpr_som_prox = 0.02  #0.04
    backpr_prox_dist = 0.15 #15  #0.09 
    V_fact = 25 
    
    
    ### Total dendritic current ###
    
    tval21 = tf.add(tf.multiply(I_soma_prox,tf.to_float(tf.greater(I_soma_prox,0 ))), 
                          tf.multiply(V_fact*backpr_som_prox,
                                 tf.multiply(I_soma_prox,
                                        tf.to_float(tf.less(I_soma_prox,0 )) 
                                        )
                                 )
                          )
    tval22 = tf.add(tf.multiply(backpr_prox_dist,
                                   tf.multiply(I_prox_dist,
                                          tf.to_float(tf.less(I_prox_dist,0 )) 
                                          )
                                   ), 
                           tf.multiply(I_prox_dist,tf.to_float(tf.greater(I_prox_dist,0 ) ))
                           )                     
    tval2 = tf.subtract(tval21,tval22) 
    
    
    tval31 = tf.multiply(backpr_prox_dist,tf.multiply(I_prox_dist,tf.to_float(tf.less(I_prox_dist,0 ))))
    tval32 = tf.multiply(I_prox_dist,tf.to_float(tf.greater(I_prox_dist,0 )))                                           
    tval3 = tf.add(tval31,tval32 )
                                                   
    tval1 = tf.concat([tval2,tval3],1)
    I_dend = tf.add(I_syn_dend,tval1)#I_syn_dend
    
    
#     I_dend(1,1,1) = I_dend(1,1,1) + 9000 
#     I_dend(1,2,1) = I_dend(1,2,1) + 15500 
    
    #---------------------
    # Currents
    #------------------------
    
    ### low-pass filtered membrane potentials ###
    lp1 = tf.add(Nrn_u_lp[:,0:2,:] , tf.multiply((timeSteptf/(uM_Tconst)),tf.subtract( u_delay[:,:,:,0],Nrn_u_lp[:,0:2,:])) )
    lp2 = tf.add(Nrn_u_lp[:,2:4,:] , tf.multiply((timeSteptf/(uP_Tconst)),tf.subtract( u_delay[:,:,:,0],Nrn_u_lp[:,2:4,:]))  )
    Nrn_u_lp2 = tf.concat([lp1,lp2],1)
    
    traces_0 = tf.add(Nrn_trace[:,0],tf.multiply((timeSteptf/Homeostat_tau),tf.subtract(tf.multiply((volt_S-E_L),tf.to_float(tf.greater(volt_S,E_L))),Nrn_trace[:,0])))  #rectified to avoid the point u = -u_ref
    
    ### store membrane potential of previous timeSteptf ###
    Nrn_u_delay_shift = u_delay[:,:,:,1:] 
    Nrn_u_delay_end =  tf.expand_dims(volt_D,3)
    Nrn_u_delay2 = tf.concat([Nrn_u_delay_shift,Nrn_u_delay_end],3)
    
    ### Somatic Voltage evolution ###
    Synapses2 = tf.reduce_sum(tf.reduce_sum(Synapses,2),2)
    Inh_feedb2 = tf.add(tf.multiply(Inh_feedb,(1-(timeSteptf/Inh_tau))), 
                        tf.squeeze(tf.matmul(tf.transpose(
                        tf.expand_dims(tf.to_float(tf.equal(Nrn_trace[:,3],1)),1)),tf.to_float(tf.greater(Synapses2,0)))))
    Inh_feedb3 = tf.add(Inh_feedb4,tf.multiply((timeSteptf/inh_tc2),tf.subtract(Inh_feedb2,Inh_feedb4)))
    
    ampvar = -inh_amp*tf.multiply(tf.multiply(tf.subtract(volt_S,E_GABA),tf.to_float(tf.less(volt_S,0))),tf.to_float(tf.greater(Inh_feedb2,0.01))) #*7*(Voltage_S>E_GABA) #
    
#    Inh_current = np.multiply(ampvar,(np.tanh(alphav*(Inh_feedb-betav))+1))
    Inh_current = tf.multiply(ampvar,tf.pow(tf.multiply(tf.subtract(Inh_feedb3,betav),
                                              tf.to_float(tf.greater(Inh_feedb3,betav))/gammav),inh_power))
    
    volt_S2 = tf.add(volt_S ,tf.multiply((timeSteptf/C),
                                    tf.add(tf.subtract(I_exp,  tf.add(I_L, tf.squeeze(tf.reduce_sum(tf.multiply(backpr_som_prox,tf.multiply(I_soma_prox,tf.to_float(tf.less(I_soma_prox,0)))),2) )))
        , tf.add(tf.add(I_ext , I_noise),Inh_current) )))  #- sum((1/V_fact)*I_soma_prox.*(I_soma_prox>0),3)
    
    
    ### Dendritic voltage evolution ###
    C_matr = tf.multiply(spineFactor*C,tf.ones((NrON_1,2,Nr_D)))
    #    C_matr = tf.tile(,[NrON,2,Nr_D]) 
    volt_D2 = tf.add(volt_D,tf.multiply(timeSteptf,tf.div(tf.subtract(I_dend,I_L_D),C_matr )))   
    
    ### x (spike trace) ###
    traces_1 = tf.subtract(tf.add(Nrn_trace[:,1],tf.multiply(tf.to_float(tf.equal(Nrn_trace[:,3], 1)),x_reset )),tf.multiply((timeSteptf/x_tau),Nrn_trace[:,1])) 
    
    ### threshold V_T evolution ###
    traces_2 = tf.add(Nrn_trace[:,2], tf.multiply( (timeSteptf/V_T_tconst),tf.subtract(V_Trest,Nrn_trace[:,2]) ))
    
    
    ### Noise ###
    auxn1 = tf.multiply(tf.to_float(tf.sqrt(tf.to_float(2*Noise_tau) )),tf.random_normal([NrON_1]))
    auxn2 = tf.multiply(tf.to_float(Noise_avg),tf.ones(NrON_1))    
    
    traces_5 = tf.add(Nrn_trace[:,5] , 
                      tf.multiply(tf.to_float(timeSteptf/Noise_tau),tf.subtract( 
        tf.add(tf.multiply(auxn1,tf.to_float(Noise_std)), 
                auxn2),Nrn_trace[:,5] ) )) #filter the noise
       
    #
    traces_34 = Nrn_trace[:,3:5] 
    Nrn_trace2 = tf.concat([tf.stack([traces_0,traces_1,traces_2],axis=1),traces_34,tf.expand_dims(traces_5,1)],1)
    
    return volt_S2,volt_D2,Nrn_trace2,Nrn_u_lp2,Nrn_u_delay2,Inh_feedb2,Inh_current