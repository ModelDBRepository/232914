import tensorflow as tf


def tf_spikes(timeSteptf,NrON_1,Nr_D,Nrn_trace2,volt_S2,spike_width_dend,spike_width_soma,v_reset,volt_D2,
              Nrn_u_delay2,V_Tjump,V_Trest,spike_latency,thresh,refrac_time):

    
    uno = tf.ones(NrON_1)
    #    uno3 = np.ones((NrON,2,np.size(Volt_D2,axis=2)))
    spike_Height_soma = 30
    spike_Height1 = tf.multiply(tf.to_float(10),tf.ones((NrON_1,1,Nr_D)))# tf.tile(tf.expand_dims(tf.expand_dims(2,1),2),[NrON_1,1,Nr_D])
    spike_Height2 = tf.multiply(tf.to_float(-3),tf.ones((NrON_1,1,Nr_D)))#tf.tile(tf.expand_dims(tf.expand_dims(-5,1),2),[NrON_1,1,Nr_D])
    spike_Height = tf.concat([spike_Height1,spike_Height2],1)    
    spikeFired = Nrn_trace2[:,3] 
    spikeFired2 = tf.multiply(Nrn_trace2[:,3],timeSteptf) 
    sfired = tf.tile(tf.expand_dims(tf.expand_dims(spikeFired2,1),2),[1,2,Nr_D])  # repeat for all compartments
    ut = tf.identity(volt_S2)  
    
    
    diff_width = spike_width_dend-spike_width_soma 
    #if spikeFired==1 (or 1.5 for dend)
    volt_S3 = tf.add(tf.multiply(volt_S2,tf.to_float(tf.less(spikeFired2,spike_width_soma))),
                    tf.multiply(tf.to_float(v_reset),tf.multiply(uno,tf.to_float(tf.greater_equal(spikeFired2,spike_width_soma))))) # floor(spikeFired2)==spike_width) #
    volt_D3 = tf.add(tf.multiply(volt_D2,tf.to_float(tf.less(sfired,spike_width_soma+diff_width+spike_latency))),
                    tf.multiply(Nrn_u_delay2[:,:,:,0],tf.to_float(tf.greater_equal(sfired,spike_width_soma+diff_width+spike_latency))))

    trace_2 = tf.add(tf.multiply(Nrn_trace2[:,2],tf.to_float(tf.greater_equal(tf.abs(spikeFired2-spike_width_soma),timeSteptf))),
                     tf.multiply((V_Tjump+V_Trest),tf.multiply(uno,tf.to_float(tf.less(tf.abs(tf.subtract(spikeFired2,spike_width_soma)),timeSteptf))))) #floor(spikeFired2)==spike_width)     
    
    
    #if spikeFired==0 and ut>thresh    
    volt_S4 = tf.add(tf.multiply(volt_S3,tf.maximum(tf.multiply(uno,tf.to_float(tf.greater_equal(spikeFired2,spike_width_soma))),
                                         tf.multiply(uno,tf.to_float(tf.less_equal(ut,thresh))))),  
                tf.multiply(tf.to_float(spike_Height_soma),tf.multiply(tf.to_float(tf.less(spikeFired2,spike_width_soma)),tf.to_float(tf.greater(ut,thresh)))))

#    volt_S4 = tf.Print(volt_S4, [volt_S4], message="This is volt_S4: ")    
    
    dv1 = tf.multiply(volt_D3,tf.maximum(tf.to_float(tf.greater_equal(sfired,spike_width_soma+diff_width+spike_latency)),tf.to_float(tf.equal(sfired,0))))
    dv2 = tf.multiply(tf.multiply(volt_D3,tf.to_float(tf.less(sfired,spike_latency))),tf.to_float(tf.greater(sfired,0)))
    dv3 = tf.to_float(tf.logical_and(tf.logical_and(tf.greater_equal(sfired,spike_latency),
                                        tf.less(sfired,spike_width_soma+diff_width+spike_latency)),
                        tf.greater(sfired,0)))
    volt_D4 = tf.add(tf.add(dv1,dv2),  
                    tf.multiply(tf.maximum(tf.to_float(spike_Height),Nrn_u_delay2[:,:,:,0]),dv3))   #.*repmat((Volt_S2>thresh),1,2,size(Volt_D2,3))
    dv4 = tf.multiply(tf.multiply(uno,
        tf.to_float(tf.equal(spikeFired,0))),tf.to_float(tf.greater(ut,thresh)))
    trace_3 = tf.multiply(tf.add(tf.multiply(tf.add(Nrn_trace2[:,3],1),tf.to_float(tf.not_equal(spikeFired,0))),dv4) ,
                     tf.to_float(tf.less_equal(spikeFired2,spike_width_soma+diff_width+spike_latency+refrac_time)))
    
    trace_01 = Nrn_trace2[:,0:2]  
    trace_45 = Nrn_trace2[:,4:6]   
    Nrn_trace3 = tf.concat([trace_01,tf.stack([trace_2,trace_3],axis=1),trace_45],1)
             
    
    return volt_S4,volt_D4,Nrn_trace3