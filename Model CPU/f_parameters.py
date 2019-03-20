

def  f_parameters():
    #Parameters: define all the parameters needed  

    timeStep = 0.025 #[ms]
    
    #Parameters to Calculate STDP
    STDP_par = [ 
        5e-4,  #A_LTD          15e-5
        12e-4,  #A_LTP          7.0e-5
        -69,  #theta_Low      -70.6+2
        -15,  #theta_High     -45.3
        1,  #(4) w_max          1
        0,  #w_min          0.001
        -1,  #homeostatic target potential (<0 for no homeostasis)
        0.2,  # correction factor for LTP due to NMDA spikes
        ]            
    
    #Parameters: Neuron
    Neuron_par = [
        -69,  #E_L                -70.6 mV
        40,  #g_L                30 nS
        281,  #C                  281 pF
        20,  #thresh             20 mV
        1,  # (4) Nr of dendrites
        0,  #%
        35,  #uM_Tconst          10
        35,  #uP_Tconst          7
        15,  #x_Tconst           15
        1,  # (9) x_reset      1
        -69,  #u_reset            -70.6
        2,  #delta_T            2
        -50.4,  #V_Trest            -50.4
        20,  #V_Tjump            20
        50,  # (14) V_T_tau      50
        0,  #delay between pre spike and response [ms] 0
        20,  # time constant for the noise
        0,  # noise average
        0 ,  #noise St Dev
        10000,  #(19) timeconstant for homeostasis
        1,  # spike width soma [ms]
    	1 ,  #spike width dendrite [ms]
    	0.3 ,  # latency of the spike in dendrites [ms]
    	0,  #absolute refractory time (total abs. refractory time = spike_width_dend+spike_latency+refrac_time) [ms]
    	-69+14  #(24) reset voltage after spike [mV]
        ]         
    
    EPSC_par = [
        0   ,   #              % (0) AMPA reversal potential
        0    ,   #             % NMDA reversal potential
        4*5  ,   #              % Amplitude of AMPA current
        30     ,   #            % Amplitude of NMDA current
        2 ,   #        % (4) AMPA conductance decay time
        50 ,   #       % NMDA conductance decay time
        0.95  ,   #    % attenuation: from prox to soma
        0.4   ,   #   % attenuation: from dist to soma
        60 ,   #        % AMPA conductance
        120    #           % (9) NMDA conductance
        ]

    return timeStep, STDP_par, Neuron_par, EPSC_par

