import tensorflow as tf

def tf_createVariables(NrON_1,NrON_0,S_potential,Nrn_u_delay,Nrn_traces,Nrn_u_lowpass,D_potential,
                       Nrn_conductance,Synapses,input_x):
#    Ipt_spikes = tf.placeholder(tf.float32, shape=(NrON_0-NrON_1), name='Ipt_spikes')
    spk_gen = tf.placeholder(tf.float32, shape=(NrON_1), name='spk_gen')
                
    volt_S = tf.Variable(tf.to_float(S_potential),  name='volt_S')
    u_delay = tf.Variable(tf.to_float(Nrn_u_delay),  name='u_delay')
    Nrn_trace = tf.Variable(tf.to_float(Nrn_traces),  name='Nrn_trace')
    Nrn_u_lp = tf.Variable(tf.to_float(Nrn_u_lowpass),  name='Nrn_u_lp')
    volt_D = tf.Variable(tf.to_float(D_potential),  name='volt_D')
    Conductances = tf.Variable(tf.to_float(Nrn_conductance), name='Conductances')
    Synapses_out = tf.Variable(tf.to_float(Synapses), name='WeightMatrix')
    
    Inh_feedb = tf.Variable(tf.to_float(0.0),name='Inh_feedb')
    inputX = tf.Variable(tf.to_float(input_x),  name='inputX')
    sim_index = tf.Variable(0.0, name="sim_index")
    
    return spk_gen,volt_S,u_delay,Nrn_trace,Nrn_u_lp,volt_D,Conductances,Synapses_out,Inh_feedb,inputX,sim_index