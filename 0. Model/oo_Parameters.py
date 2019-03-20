# Relevant parameters for the model

from brian2 import *

#####################################################
# Set Parameters
#####################################################

timeStep = 25.*us # timeStep
defaultclock.dt = timeStep # timeStep
spinedist = 50           # spine distance
celsius_temp = 32.       # temperature (see Acker&Antic 2008)
tadj = 2.3**((celsius_temp - 23.)/10.)  # temperature adjustment factor

#------------
# PASSIVE
#------------
R_axial = 90.*ohm*cm                # axial resistance
spine_factor = 1.5                  # factor to account for spines
g_leak = 0.04*msiemens/cm**2        # leak conductance
g_leak_dend = g_leak*spine_factor   # leak conductance in presence of spines
Capacit = 1.*uF/cm**2               # membrane capacitance
Capacit_dend = Capacit*spine_factor # membrance capacitance in presence of spines
EL=-60*mV                          # resting membrane voltage


#------------
# AMPA and NMDA
#------------
taus = 2.*ms            # AMPA time constant
Es = 0.*mV              # AMPA reversal potential
ENMDA = 0.*mV           # NMDA reversal potential
Egaba = -72.*mV           # NMDA reversal potential
tau_NMDA = 50.*ms       # NMDA time constant
tau_gaba = 10.*ms       # NMDA time constant
syn_weight = 0.5        # initial synaptic weight
wa_max = 1.             # maximal AMPA weight
wn_max = 1.             # maximal NMDA weight
w_min = .0001            # maximal NMDA weight
ampa_cond = 1500.*1e-12*siemens  # AMPA maximal conductance
nmda_cond = 3000.*1e-12*siemens  # NMDA maximal conductance
gaba_cond = 1500.*1e-12*siemens  # NMDA maximal conductance


#------------
# Plasticity and noise
#------------
Theta_low_Branco = EL - 12*mV  # depolarization threshold for plasticity
Theta_low_Acker = EL - 9*mV  # depolarization threshold for plasticity
Theta_high = -15.*mV    # threshold for potentiation events
x_reset = 5.            # spike trace reset value
taux = 20.*ms           # spike trace time constant
A_LTD = 4.e-4           # depression amplitude 
A_LTP = 14.e-4          # potentiation amplitude  
tau_ca = 5.*ms          # filter1 timeconstant
tau_ltd = 15.*ms          #filter2 timeconstant
tau_ltp = 45.*ms          #filter2 timeconstant
tau_noise = 20.*ms      # noise filter timeconstant
noise_std = 0*pA       # noise standard deviation
noise_mean = 0*pA     # noise mean

probConnect = 0.0000001*timeStep/ms#0.02   #probConnect
probDisconnect = 0.000025*timeStep/ms#0.02    # probDisconnect
strpower = 1 #str power
sbias = 150

#------------
# ACTIVE Conductances (based on Acker&Antic 2008)
#------------

#Sodium channels
somaNa = 200.*psiemens/um**2
axonNa = 5000.*psiemens/um**2
basalNa = 150.*psiemens/um**2
apicalNa = 250.*psiemens/um**2 

#Calcium channels
somaCa = 5.*psiemens/um**2
dendCa = .5*psiemens/um**2
ratio_ca = .8   

#Delayed-rectifier Potassium channels
dendgKv = 40.*psiemens/um**2
somagKv = 400.*psiemens/um**2
axongKv = 500.*psiemens/um**2

#IL channel 
axongKl = 0*psiemens/um**2

#A-type Potassium channels 
somaKap = 300.*psiemens/um**2
basalKa = 150.*psiemens/um**2 
apicalKa = 300.*psiemens/um**2 

#Leak conductance in axon
axongL = g_leak 

#------------
# Ion Channel Parameters (see Acker&Antic 2008)
#------------

#Na
vshiftna = -10.
ENav = 60.*mV            
tha  = -35.
qa   = 9.
Ra   = 0.182
Rb   = 0.124	
thi1_all = -65.
thi2_all = -65.
thi1_axn = -58.
thi2_axn = -58.
qi   = 6.
thinf  = -65.
qinf  = 6.2
Rg   = 0.02
Rd   = 0.024

#Cav / T-type
ECav = 140.*mV     
vshiftca = 10.
v12m=45.  
v12h=65. 
vwm =7.4      
vwh=5.0   
am=3.      
ah=30.   
vm1=50.  
vm2=125.  
vh1=56.  
vh2=415. 
wm1=20.   
wm2=15. 
wh1=4.   
wh2=50.

#Kv
EKv=-80.*mV             
tha_kv  = 25.
qa_kv   = 9.
Ra_kv   = 0.02
Rb_kv   = 0.002

#Kap
vhalfn=11.
vhalfl=-56.
#a0l=0.05
a0n=0.05
zetan=-1.5
zetal=3.
gmn=0.55
gml=1.
lmin=2.
nmin=0.1
pw=-1.
tq=-40.
qq=5.
qt=5.**((celsius_temp-24.)/10.)
qtl=1.

#Kad
vhalfn2=-1.
vhalfl2=-56.
#a0l2=0.05 
a0n2=.1 
zetan2=-1.8 
zetal2=3.
gmn2=0.39 
gml2=1.
lmin2=2.
nmin2=0.1
pw2=-1.
tq2=-40.
qq2=5.
qtl2=1.

#KL
vhalfn3=11.
vhalfl3=-56.
#a0l3=0.05
a0n3=0.05
zetan3=-1.5
zetal3=3.
gmn3=0.55
gml3=1.
lmin3=2.
nmin3=0.1
pw3=-1.
tq3=-40.
qq3=5.
qtl3=1.




