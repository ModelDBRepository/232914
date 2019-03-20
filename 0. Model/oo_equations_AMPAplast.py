#####################################################
# Equations for the biophysical neuron
#####################################################

# Passive/noise
eqs_base='''      
Ileak = gLeak*(EL-v) : amp/meter**2
gLeak :  siemens/meter**2
I : amp (point current)
dIstim/dt = (-Istim/(.1*ms)+15*nA/timeStep*v2) :amp (point current)
v2 = rand()<rate_v*dt :1 (constant over dt)
rate_v :Hz
dInoise/dt = (noise_avg-Inoise)/tau_noise + xi*noise_sigma*(tau_noise/2)**(-.5) : amp (point current)
noise_sigma : amp
noise_avg : amp
vu = v/mV + vshiftna :1
vu3 = v/mV + vshiftca :1
vu2 = v/mV  :1
Iactive = (gNav*m**3*h*(ENav-v)+
    (tadj*gCav*m2**2*h2 + gIt*m3**2*h3)*(ECav-v)+
    (gKv*n  + gKa_prox*n2*l + gKa_dist*n3*l2 + gKL*n4*l3)*(EKv-v)): amp/meter**2
'''

#VG NA+
eqs_Nav = """
dm/dt = (minf-m)/mtau : 1
dh/dt = (hinf-h)/htau : 1
mtau = 1.*ms/(tadj*(Ra * (vu - tha) / (1. - exp(-(vu - tha)/qa))-Rb * (vu - tha) / (1. - exp((vu - tha)/qa)))) :second
minf = 1./(1. + (Rb/Ra)*exp(-(vu - tha)/qa)) :1
htau = 1.*ms/(tadj*(-Rd * (vu - thi1) / (1. - exp((vu - thi1)/qi))+Rg * (vu - thi2) / (1. - exp(-(vu - thi2)/qi)))) :second
hinf = 1./(1. + (Rg/Rd)*exp((vu - thi1)/qi)) :1
gNav : siemens/meter**2
thi1 :1
thi2 :1
"""

#VG CA2+
eqs_Cav = """
dm2/dt = (minf2-m2)/mtau2 : 1
dh2/dt = (hinf2-h2)/htau2 : 1
mtau2 = 1.*ms/(tadj*(0.055*(-27. - vu2)/(exp((-27. - vu2)/3.8) - 1.) + 0.94*exp((-75.-vu2)/17.))) :second
minf2 = (0.055*(-27. - vu2)/(exp((-27.-vu2)/3.8) - 1.))/((0.055*(-27. - vu2)/(exp((-27.-vu2)/3.8) - 1.))+(0.94*exp((-75.-vu2)/17.))) :1
htau2 = 1.*ms/(tadj*((0.000457*exp((-13.-vu2)/50.))+(0.0065/(exp((-vu2-15.)/28.) + 1.)))):second
hinf2 = (0.000457*exp((-13.-vu2)/50.))/((0.000457*exp((-13.-vu2)/50.))+(0.0065/(exp((-vu2-15.)/28.) + 1.))) :1
dm3/dt = (minf3-m3)/mtau3 : 1
dh3/dt = (hinf3-h3)/htau3 : 1
mtau3 =  ( am + 1.0 / ( exp((vu3+vm1)/wm1) + exp(-(vu3+vm2)/wm2) ) )*ms :second
minf3 = 1.0 / ( 1. + exp(-(vu3+v12m)/vwm) ) :1
htau3 = ( ah + 1.0 / ( exp((vu3+vh1)/wh1) + exp(-(vu3+vh2)/wh2) ) )*ms :second
hinf3 = 1.0 / ( 1. + exp((vu3+v12h)/vwh) ):1
gCav : siemens/meter**2
gIt : siemens/meter**2
"""


#, Proximal A-type Ka_prox, Distal A-type Ka_dist 
eqs_K = """
#### DELAYED-RECTIFIER K+ #################
dn/dt = (ninf-n)/ntau : 1
ntau = 1.*ms/(tadj * (Ra_kv * (vu2 - tha_kv) / (1. - exp(-(vu2 - tha_kv)/qa_kv))-Rb_kv * (vu2 - tha_kv) / (1. - exp((vu2 - tha_kv)/qa_kv)))):second
ninf = 1. /(1. + (Rb_kv/Ra_kv)*exp(-(vu2 - tha_kv)/qa_kv)):1
gKv : siemens/meter**2
#### PROXIMAL A-TYPE K+ #################
dn2/dt = (ninf_2-n2)/ntau_2 : 1
dl/dt = (linf-l)/ltau : 1
zeta=zetan+pw/(1+exp((vu2-tq)/qq)) :1
alpn = exp(1e-3*zeta*(vu2-vhalfn)*9.648e4/(8.315*(273.16+celsius_temp))) :1
betn = exp(1e-3*zeta*gmn*(vu2-vhalfn)*9.648e4/(8.315*(273.16+celsius_temp))):1
ninf_2 = 1./(1.+alpn) :1
ntau_2 = (betn*ms/(qt*a0n*(1+alpn)) - nmin*ms)*(betn/(qt*a0n*(1+alpn))>nmin) + nmin*ms : second
alpl = exp(1e-3*zetal*(vu2-vhalfl)*9.648e4/(8.315*(273.16+celsius_temp))) :1
linf = 1./(1.+ alpl) :1
ltau = ((0.26*ms*(vu2+50.) - lmin*ms)/qtl)*(0.26*(vu2+50.) > lmin) + lmin*ms/qtl :second
gKa_prox : siemens/meter**2
#### DISTAL A-TYPE K+ #################
dn3/dt = (ninf_3-n3)/ntau_3 : 1
dl2/dt = (linf2-l2)/ltau2 : 1
zeta2 =zetan2+pw2/(1+exp((vu2-tq2)/qq2)) :1
alpn2 = exp(1e-3*zeta2*(vu2-vhalfn2)*9.648e4/(8.315*(273.16+celsius_temp))) :1
betn2 = exp(1e-3*zeta2*gmn2*(vu2-vhalfn2)*9.648e4/(8.315*(273.16+celsius_temp))):1
ninf_3 = 1./(1.+alpn2) :1
ntau_3 = (betn2*ms/(qt*a0n2*(1+alpn2))-nmin2*ms)*(betn2/(qt*a0n2*(1+alpn2))>nmin2) + nmin2*ms : second
alpl2 = exp(1e-3*zetal2*(vu2-vhalfl2)*9.648e4/(8.315*(273.16+celsius_temp))) :1
linf2 = 1./(1.+ alpl2) :1
ltau2 = ((0.26*ms*(vu2+50.) - lmin2*ms)/qtl2)*(0.26*(vu2+50.)>lmin2) +lmin2*ms/qtl2 :second
gKa_dist : siemens/meter**2
#### SLOWLY INACTIVATING K+ #################
dn4/dt = (ninf_4-n4)/ntau_4 : 1
dl3/dt = (linf3-l3)/ltau3 : 1
zeta3 = zetan3+pw3/(1+exp((vu2-tq3)/qq3)) :1
alpn3 = exp(1e-3*zeta3*(vu2-vhalfn3)*9.648e4/(8.315*(273.16+celsius_temp))) :1
betn3 = exp(1e-3*zeta3*gmn3*(vu2-vhalfn3)*9.648e4/(8.315*(273.16+celsius_temp))) :1
ninf_4 = 1./(1. + exp(-(vu2+60.)/20.)) :1
ntau_4 = (betn3*ms/(qt*a0n3*(1+alpn3))-nmin3*ms)*(betn3/(qt*a0n3*(1+alpn3))>nmin3) + nmin3*ms : second
linf3 = 1./(1.+ exp((vu2+30.)/20.)) :1
ltau3 = ((0.26*ms*(vu2+50.)*20.-lmin3*ms)/qtl3)*(0.26*(vu2+50.)*20. > lmin3) + lmin3*ms/qtl3 : second
gKL : siemens/meter**2
"""

eqs_synapses = """
Is = gs*(Es-v) : amp (point current)
gs  : siemens
I_NMDA = gNMDA*(ENMDA-v)*Mgblock: amp (point current)
gNMDA : siemens
Mgblock :1 
"""

eqs_gaba = '''
Igaba = ggaba*(Egaba-v) :amp (point current)
ggaba : siemens
'''

eqs_plasticity = """
dv_ca/dt = (v-v_ca)/tau_ca :volt
dv_ltd/dt = (v_ca-v_ltd)/tau_ltd :volt
dv_ltp/dt = (v_ca-v_ltp)/tau_ltp :volt
ds_trace/dt = -s_trace/taux :1
"""


# total membrane current
eqs_memb='''     
Im = Ileak+ Iactive  : amp/meter**2    #
'''
eqs = eqs_synapses + eqs_gaba + eqs_memb + eqs_base + eqs_Nav + eqs_K + eqs_Cav  + eqs_plasticity

#####################################################
# Synapses
#####################################################

eq_1_plastAMPA =   ('''
            dg/dt = -g/taus : siemens (clock-driven)
            gs_post = g : siemens (summed)
            dwampa/dt = ((wampa<wa_max)*(v_ltp_post/mV - Theta_low/mV)
            *(v_ltp_post/mV - Theta_low/mV > 0)*A_LTP*s_trace_pre
            *(v_post/mV - Theta_high/mV)*(v_post/mV - Theta_high/mV > 0)):1 (clock-driven)
            dg_nmda/dt = -g_nmda/tau_NMDA : siemens (clock-driven)
            gNMDA_post = g_nmda : siemens (summed) 
            wnmda :1 
            dg_gaba/dt = -g_gaba/tau_gaba : siemens (clock-driven)
            ggaba_post = g_gaba : siemens (summed)
            wgaba :1
            ''')

eq_2_plastAMPA = ('''            
            g += wampa*ampa_cond     
            wa_minus = A_LTD*(v_ltd_post/mV - Theta_low/mV)*(v_ltd_post/mV - Theta_low/mV > 0)
            wampa = clip(wampa-wa_minus,w_min,wa_max)       
            g_nmda += wnmda*nmda_cond      
            g_gaba += wgaba*gaba_cond     
            ''' )
            
eq_1_nonPlast =   ('''
            dg/dt = -g/taus : siemens (clock-driven)
            gs_post = g : siemens (summed)
            dg_nmda/dt = -g_nmda/tau_NMDA : siemens (clock-driven)
            gNMDA_post = g_nmda : siemens (summed)  
            wampa:1
            wnmda:1
            ''')

eq_2_nonPlast = ('''            
            g += wampa*ampa_cond          
            g_nmda += wnmda*nmda_cond    
            ''' )
            
eq_1_gaba =   ('''
            dg_gaba/dt = -g_gaba/tau_gaba : siemens (clock-driven)
            ggaba_post = g_gaba : siemens (summed)
            wgaba :1
            ''')

eq_2_gaba = ('''                 
            g_gaba += wgaba*gaba_cond     
            ''' )