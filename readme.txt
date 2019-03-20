================================================================================================
Modeling somatic and dendritic spike mediated plasticity at the single neuron and network level
================================================================================================

Figures 1-5:

(-) The folder '0. Model' contains general scripts concerning the biophysical model used in simulations for Figs. 1-5: model morphology, model equations, model parameters, ... 

(-) Folders Fig 1 to Fig 5 contain scripts to generate the necessary data used for respective figures in the article, using the biophysical neuron model the from '0. Model' folder.

(-) All scripts were written to be compatible with Python3.5 and Brian2 v2.0.

Figure 6:

(-) The folders 'Model CPU' and 'Model GPU' contain general scripts concerning the reduced model used in simulations for Fig 6. 

(-) The folder Fig 6 contains scripts to generate the necessary data used for figure 6. 

(-) All scripts were written to be compatible with Python3.5 with the tensorflow package (to run on GPU).


Jacopo Bono
September 2017

20190320 An update from Jacopo Bono for compatibility with newer
versions of the Brian2 package.  The fix is very minimal (1 change in
line 136 of the file "oo_equations_AMPAplast.py" which replaces the
current version (in the folder "0. Model/")
