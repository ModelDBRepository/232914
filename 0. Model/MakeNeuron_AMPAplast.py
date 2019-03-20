# This script contains the Brianmodel class
# Calling makeneuron_ca() on a Brianmodel object will create a biophysical neuron
# Multiple other functions allow for plotting, animating, ...



from __future__ import division

#folder with parameters, equations and morphology

import os, sys
mod_path = os.path.abspath(os.path.join('..','0. Model'))
sys.path.append(mod_path)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
import matplotlib.colors as colorz
import matplotlib.cm as clrm
import brian2 as br2
from brian2 import uF, cm, um, ohm, ms, siemens, mV, nA, us,psiemens

# This is the 3D plotting toolkit
from mpl_toolkits.mplot3d import Axes3D

#import parameters and equations for neuron
from oo_Parameters import *
from oo_equations_AMPAplast import *
from MorphologyData import *
from Visualisation_functions import *
from oo_initScripts import set_init_nrn

br2.start_scope()
br2.defaultclock.dt = defaultclock.dt



class BRIANModel(object):
    """
    Neuron object in brian2
    """
    def __init__(self, swc_model):
        """
        Parameters
        ----------
        swc_model: a char
            path of the file containing the neuron model in .swc format
        """

        # Brian morphology
        self.morpho = br2.Morphology.from_file(swc_model)
        morpho = self.morpho
        # Store compartment numbers       
        self.segment,self.segment_swc = get_swc(swc_model) 
        # Initialise an dictionary for distances to the soma per compartment
        self.distances = {}
        # Initialise an dictionary for lines to plot the neuron
        self.lines = {}


        # Add the first section as soma
        self.sections = {morpho.type: [self.morpho[0], 0, 0]}
        
        # Set a name and distances for the soma
        self.sections['soma'][0].name = 'soma'
        self.sections['soma'][0].f_x = self.morpho[0].x/meter
        self.sections['soma'][0].f_y = self.morpho[0].y/meter
        self.sections['soma'][0].f_z = self.morpho[0].z/meter
        self.sections['soma'][0].dist = 0
        self.distances['soma'] = [0.]
        
        # Initialize the dendrites numerotation
        dend_b = 0

        # Register soma's children in a sections dictionary
        for sec in morpho.children:

            # Create an attribut "name" for all children of the soma
            if str(sec.type) == "dend":
                sec.name = sec.type[:4]+"_"+str(dend_b)
                dend_b += 1
            else:
                sec.name = sec.type

            # store final coordinates of the parent (=soma) segment
            sec.f_x = self.morpho[0].x[0]/meter
            sec.f_y = self.morpho[0].y[0]/meter
            sec.f_z = self.morpho[0].z[0]/meter
            sec.dist = self.distances['soma'][0]
            
            # add distances to the parent
            self.distances = calc_dist(self.distances, sec)

            # get the coordinates for all compartments in this section
            xn = sec.x/meter
            yn = sec.y/meter
            zn = sec.z/meter
            # get first coordinates (and make integer)
            a=(int(round(xn[0]*1e9)),int(round(yn[0]*1e9)),int(round(zn[0]*1e9))) 
            # id for the section (they correspond to lnum in .swc)
            line_num = self.segment[a]
            # add id and section to the 'sections' dictionary
            self.sections[sec.name] = [sec,line_num,line_num]            
                       
            
        # Initialize the level value
        level = [sec for sec in morpho.children]
        while level != []:
            for i, sec in enumerate(level):
                for j, child in enumerate(sec.children):
                    
                    # Create an attribut "name" for all children of sec
                    name = sec.name + str(j)
                    child.name = name
                    # Store parent final coordinates
                    child.f_x = sec.x[-1]/meter
                    child.f_y = sec.y[-1]/meter
                    child.f_z = sec.z[-1]/meter
                    # Store distances to the soma
                    child.dist = self.distances[sec.name][-1]
                    self.distances = calc_dist(self.distances, child)
                    # Get the coordinates for all compartments in this section
                    xn = child.x/meter
                    yn = child.y/meter
                    zn = child.z/meter
                    # get first coordinates (and make integer)
                    a=(int(round(xn[0]*1e9)),int(round(yn[0]*1e9)),int(round(zn[0]*1e9)))
                    
                    # id for the section (corresponds to lnum in .swc)
                    line_num = self.segment[a]
                    # add id and section to the 'sections' dictionary
                    self.sections[name] = [child, line_num,line_num]            
            level = [sec.children for sec in level]
            # Flatten the list at this level
            level = [sublist for sl in level for sublist in sl]
            
    ################################################################################
    # THE FUNCTION BELOW CAN BE CALLED TO CREATE A BIOPHYSICAL NEURON
    ################################################################################

    def makeNeuron_Ca(self,morphodata):
        """return spatial neuron"""
        # Set Biophysics
        neuron = self.biophysics(morphodata)
        return neuron
        
    def biophysics(self,morpho_data):
        """Inserting biophysics"""
                    
        neuron = br2.SpatialNeuron(morphology=self.morpho, model=eqs, \
            Cm=Capacit, Ri=R_axial, threshold  = "v/mV>0", refractory = "v/mV > -10",
            threshold_location = 0, reset = 's_trace += x_reset*(taux/ms)',method='heun') #
        
        # define the different parts of the neuron
        N_soma = neuron[morpho_data['soma'][0]:morpho_data['soma'][-1]+1]
        N_axon = neuron[morpho_data['axon'][0]:morpho_data['axon'][-1]+1]
        N_basal = neuron[morpho_data['basal'][0]:morpho_data['basal'][-1]+1]
        N_apical = neuron[morpho_data['apical'][0]:morpho_data['apical'][-1]+1]
        Theta_low = morpho_data['thetalow']*mV
        
        # insert leak conductance
        neuron.gLeak = g_leak
        
        
        # noise
        neuron.noise_sigma = 0*pA # initial value membrane voltage
        neuron.noise_avg = 0*pA # initial value membrane voltage
        N_soma.noise_sigma = noise_std # initial value membrane voltage
        N_soma.noise_avg = noise_mean # initial value membrane voltage    
        
        ####################
        # ACTIVE CHANNELS
        ####################
        
        # Na channels soma, axon, apical dendrites
        N_soma.gNav = somaNa
        N_axon.gNav = axonNa
        N_apical.gNav = apicalNa
        neuron.thi1 = thi1_all
        N_axon.thi1 = thi1_axn
        neuron.thi2 = thi2_all
        N_axon.thi2 = thi2_axn
        
        #Kv channels
        N_soma.gKv = somagKv
        N_basal.gKv = dendgKv
        N_apical.gKv = dendgKv
        N_axon.gKv = axongKv
        
        #Ca channels sina
        N_soma.gCav = ratio_ca*somaCa
        N_soma.gIt = (1-ratio_ca)*somaCa
        
        #Ka channels soma
        N_soma.gKa_prox = somaKap
        
        #Ka channels dendrites, Na channels basal dendrites, Ca channels dendrites, axon initial segment
        for sec in self.sections:
            secNr = self.sections[sec][2]
            seclen = len(self.sections[sec][0].x)
            
            #BASAL
            if secNr in morpho_data['basal']:      
                # decreasing Na channels
                gNa_diff = 0.5*np.array(self.distances[sec][:])*psiemens/um**2
                neuron[secNr:secNr+seclen].gNav = np.multiply(basalNa - gNa_diff,basalNa - gNa_diff>0 ) 
                
                # increasing Ka channels
                gKa_diff = 0.7*np.array(self.distances[sec][:])*psiemens/um**2
                ratio_A = np.multiply(1. - (1./300.)*np.array(self.distances[sec][:]),1. - (1./300.)*np.array(self.distances[sec][:])>0)
                neuron[secNr:secNr+seclen].gKa_prox = ratio_A*np.multiply(basalKa + gKa_diff,basalKa + gKa_diff>0 )
                neuron[secNr:secNr+seclen].gKa_dist = (1.-ratio_A)*np.multiply(basalKa + gKa_diff,basalKa + gKa_diff>0 )
                
                # Ca channels
                neuron[secNr:secNr+seclen].gCav = dendCa*ratio_ca*(np.array(self.distances[sec][:])>30) + somaCa*ratio_ca*(np.array(self.distances[sec][:])<=30)
                neuron[secNr:secNr+seclen].gIt = dendCa*(1.-ratio_ca)*(np.array(self.distances[sec][:])>30) + somaCa*(1.-ratio_ca)*(np.array(self.distances[sec][:])<=30) 
                
                
                #spines
                addSpines = np.array(self.distances[sec][:]) > spinedist
                noSpines = np.array(self.distances[sec][:]) <= spinedist                
                neuron[secNr:secNr+seclen].gLeak = noSpines*g_leak + addSpines*g_leak_dend
                neuron[secNr:secNr+seclen].Cm = noSpines*Capacit + addSpines*Capacit_dend
                
            #APICAL
            if secNr in morpho_data['apical']:       
                #ratio of Ka channels
                ratio_A = np.multiply(1. - (1./300.)*np.array(self.distances[sec][:]),1. - (1./300.)*np.array(self.distances[sec][:])>0)
                neuron[secNr:secNr+seclen].gKa_prox = ratio_A*apicalKa
                neuron[secNr:secNr+seclen].gKa_dist = (1.-ratio_A)*apicalKa
                
                # Ca channels
                neuron[secNr:secNr+seclen].gCav = dendCa*ratio_ca*(np.array(self.distances[sec][:])>30) + somaCa*ratio_ca*(np.array(self.distances[sec][:])<=30)
                neuron[secNr:secNr+seclen].gIt = dendCa*(1.-ratio_ca)*(np.array(self.distances[sec][:])>30) + somaCa*(1.-ratio_ca)*(np.array(self.distances[sec][:])<=30) 
                
                #spines
                addSpines = np.array(self.distances[sec][:]) > spinedist
                noSpines = np.array(self.distances[sec][:]) <= spinedist                
                neuron[secNr:secNr+seclen].gLeak = noSpines*g_leak + addSpines*g_leak_dend
                neuron[secNr:secNr+seclen].Cm = noSpines*Capacit + addSpines*Capacit_dend
                
            #AXON
            if secNr in morpho_data['axon']:                       
                #KL current
                addKL = np.array(self.distances[sec][:]) > 35           
                neuron[secNr:secNr+seclen].gKL = addKL*axongL
                neuron[1:6].gKv = [40.,100.,500.,500.,500.]*psiemens/um**2 
                neuron[1:6].gKL =  [20.,35.,125.,250.,0]*psiemens/um**2 
                neuron[1:6].gNav = 4*np.array([8000.,7000.,5000.,5000.,5000.])*psiemens/um**2
#                neuron[1:6].gNav = [8000.,7000.,5000.,5000.,5000.]*psiemens/um**2
                neuron[1:3].gCav = somaCa*ratio_ca
                neuron[1:3].gIt = somaCa*(1.-ratio_ca)
                                
        
        # SET INITIAL VALUES
        set_init_nrn(neuron,Theta_low)
                
        
        return neuron
   
    ################################################################################
    # The functions below are mainly for visualization of the neuron morphology
    ################################################################################
        
    def print_dist(self, sec_number):
        ''' print the distance and diameter of a section to the soma'''
        for sec in self.sections:
                for ii in range(len(self.sections[sec][0].x)):
                    if (self.sections[sec][2]+ii == sec_number):
#                        print( 'Section '+ str(sec_number)+ ', part of: '+str(sec))
#                        print( 'Distance to soma: '+ str(self.distances[sec][ii]))
#                        print( 'Diameter: '+ str(self.sections[sec][0].diameter[ii]*1.e6))
                        sectiondistance = self.distances[sec][ii]
                        sectiondiameter = self.sections[sec][0].diameter[ii]*1.e6
        return [sectiondistance,sectiondiameter]
        
    def save_dist_vs_nr(self,maxNr):
        ''' save the distance and diameter in function of section nr'''
        dist_nr = np.zeros(maxNr)
        diam_nr = np.zeros(maxNr)
        print('saving distances')
        for sec in self.sections:
                for ii in range(len(self.sections[sec][0].x)):
                    dist_nr[self.sections[sec][2]+ii] = self.distances[sec][ii]
                    diam_nr[self.sections[sec][2]+ii] = self.sections[sec][0].diameter[ii]*1.e6
        return dist_nr,diam_nr
    
    def calc_distCompartments(self,sec_range,distances):
        ''' calculate the terminal ends of dendrites '''
        term_vec = [0]
        dist_vec = distances[sec_range]
        for jj in range(len(dist_vec)-1):
            if np.abs(dist_vec[jj+1]-dist_vec[jj])>20:
                term_vec = np.append(term_vec,np.array([sec_range[0]+jj]),axis=0)
        return term_vec,dist_vec
        

    def show_shape3d(self, fov=400, fig=None, ax=None):
        """Show a 3D plot of the morphology"""
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            # Set fov
            ax.set_xlim(-fov, fov)
            ax.set_ylim(-fov, fov)
            ax.set_zlim(-fov, fov)
            ax.set_aspect("equal")

#        data = self.sections["soma"][1]
        for sec in self.sections.values():
            self.lines = add_line3d(ax, self.lines, sec[0])
            
#        cmap = plt.get_cmap('Spectral')
#        cmap = plt.get_cmap('summer')
##        print np.max(data)
#        cNorm = colorz.Normalize(vmin=0, vmax=1)
#        scalarMap = clrm.ScalarMappable(norm=cNorm, cmap=cmap)
#        for sec in self.sections:
#            for ii in range(len(self.lines[sec])):
##                print ii
#                vsec = self.sections[sec][1][ii]
#                cval = scalarMap.to_rgba(vsec)
#                self.lines[sec][ii].set_color(cval)
            
    def show_segm(self, fov=500, fig=None, ax=None, segm_range = 0,colorv = 'r', segm_range2 = [0],colorv2 = 'k'):
        """Show a 2D plot of the morphology, highlight sections in range 'segm_range' """
        if fig is None:
            fig, ax = plt.subplots()
            # Set fov
            ax.set_xlim(-240, 110)
            ax.set_ylim(-200, 350)
            ax.set_aspect("equal")
        
        #add lines
        for sec in self.sections.values():
            self.lines = add_line(ax, self.lines, sec[0])
        
        # change colors
        for sec in self.sections:
            for ii in range(len(self.sections[sec][0].x)):
                if (self.sections[sec][2]+ii in segm_range):
                    cval = colorv
                    self.lines[sec][ii].set_linewidth(3)
                elif (self.sections[sec][2]+ii in segm_range2):
                    cval = colorv2
                    self.lines[sec][ii].set_linewidth(3)
                else:
                    cval = 'black'
                self.lines[sec][ii].set_color(cval)
                
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both',which='both',bottom='off',left='off',right='off',  top='off', labelbottom='off', labelleft='off')

#        savefig('./'+'Neuron'+'.eps', format='eps', dpi=1000)
#        fig.suptitle(str(int(distMin))+'um to '+str(int(distMax))+' um')
                
    def show_segm_byName(self, fov=500, fig=None, ax=None, segmName='soma'):
        """Show a 2D plot of the morphology, highlight section with name 'segmName' """
        if fig is None:
            fig, ax = plt.subplots()
            # Set fov
            ax.set_xlim(-fov, fov)
            ax.set_ylim(-fov, fov)
            ax.set_aspect("equal")

        for sec in self.sections.values():
            self.lines = add_line(ax, self.lines, sec[0])
            
        for sec in self.sections:
            for ii in range(len(self.sections[sec][0].x)):
                if (str(sec) == segmName):
                    cval = 'red'
                    print (self.sections[sec][2]+ii)
                    self.lines[sec][ii].set_linewidth(3)
                else:
                    cval = 'black'
                self.lines[sec][ii].set_color(cval)
                
            
    def show_shape(self, fovx=200,fovy=200, fig=None, ax=None):
        """Show a 2D plot of the morphology"""
        if fig is None:
            fig, ax = plt.subplots()
            # Set fov
            ax.set_xlim(-fovx, fovx)
            ax.set_ylim(-fovy, fovy)
            ax.set_aspect("equal")

        for sec in self.sections.values():
            self.lines = add_line(ax, self.lines, sec[0])
            
#        cmap = plt.get_cmap('spectral')
#        print np.max(data)
#        cNorm = colorz.Normalize(vmin=0, vmax=1)
#        scalarMap = clrm.ScalarMappable(norm=cNorm, cmap=cmap)
#        for sec in self.sections:
#            for ii in range(len(self.lines[sec])):
##                print ii
#                vsec = self.sections[sec][1][ii]
#                cval = scalarMap.to_rgba(vsec)
#                self.lines[sec][ii].set_color(cval)

    def animate3d(self):
        """ Make an animation (3D) """
        try:
            nf = len(self.sections["soma"][1][0])
            print( 'nf: '+str(nf))
        except TypeError:
            print( "running simulation first")
            self.run()
            nf = len(self.sections["soma"][1][0])

        data = self.sections["soma"][1][0]
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.set_xlim(-250, 250)
        ax.set_ylim(-250, 250)
        ax.set_zlim(-250, 250)
        ax.set_aspect('equal')
        self.show_shape3d(fig=fig, ax=ax)

        cmap = plt.get_cmap('afmhot')
        cNorm = colorz.Normalize(vmin=-75*mV, vmax=-0*mV)
        scalarMap = clrm.ScalarMappable(norm=cNorm, cmap=cmap)

        def anim3d(i):
            for sec in self.sections:
                for ii in range(len(self.lines[sec])):
#                    print ii
                    vsec = self.sections[sec][1][ii][i]
                    cval = scalarMap.to_rgba(vsec)
                    self.lines[sec][ii].set_color(cval)
            return self.lines,

        # call the animator.
        anim3d = animation.FuncAnimation(fig, anim3d, interval=20,
                                       frames=nf)
        mywriter = animation.FFMpegWriter()
        anim3d.save('basic_animation.avi', fps=30,writer=mywriter)
        
    def animate(self):
        """ Make an animation (2D) """
        try:
            nf = len(self.sections["soma"][1][0])
            print( 'nf: '+str(nf))
        except TypeError:
            print( "running simulation first")
            self.run()
            nf = len(self.sections["soma"][1][0])

        data = np.zeros([1,nf])
        for x in self.sections.values():
            data = np.append(data,np.array(x[1]),axis=0)
        data = data[1:,:]
        fig, ax = plt.subplots()
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_aspect('equal')
        self.show_shape(fig=fig, ax=ax)

        cmap = plt.get_cmap('afmhot')
        cNorm = colorz.Normalize(vmin=np.amin(data), vmax=np.amax(data))
        scalarMap = clrm.ScalarMappable(norm=cNorm, cmap=cmap)
        def anim(i):
            for sec in self.sections:
                for ii in range(len(self.lines[sec])):
#                    print ii
                    vsec = self.sections[sec][1][ii][i]
                    cval = scalarMap.to_rgba(vsec)
                    self.lines[sec][ii].set_color(cval)
            return self.lines,

        # call the animator.
        anim = animation.FuncAnimation(fig, anim, interval=20,
                                       frames=nf)
        mywriter = animation.FFMpegWriter()
        anim.save('basic_animation.avi', fps=30,writer=mywriter)
        
    def show_property(self, var_to_show,sspike=1,nspike=1,sspikemin=0,nspikemin=0):
        """Show a 2D plot of the morphology, highlight the distribution of a parameter 'var_to_show' """
        data = var_to_show
        
        fig, ax = plt.subplots()
        ax.set_xlim(-250, 75)
        ax.set_ylim(-200, 150)
        ax.set_aspect('equal')
#        ax.set_axis_bgcolor((0.93,0.93,0.93))
        self.show_shape(fig=fig, ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both',which='both',bottom='off',left='off',right='off',  top='off', labelbottom='off', labelleft='off')

        minmin = np.amin(data)
        maxmax = np.amax(data)
                
#        cmap = plt.get_cmap('seismic') 
#        cNorm = colorz.Normalize(vmin=minmin, vmax= maxmax ) # np.amin(data)  np.amax(data)
#        scalarMap = clrm.ScalarMappable(norm=cNorm, cmap=cmap)
#        scalarMap.set_array([minmin,maxmax])
        
        orig_cmap = clrm.coolwarm #coolwarm, seismic,bwr,rainbow, jet
        shiftc = 1 - maxmax/(maxmax + abs(minmin))
        newcmap = shiftedColorMap(orig_cmap, start=0, midpoint=shiftc,
                                  stop=1.0, name='shiftedcmap',
                                  somspike=sspike,nmdaspike=nspike)
              
        cNorm = colorz.Normalize(vmin=minmin, vmax= maxmax )
        scalarMap = clrm.ScalarMappable(norm=cNorm,cmap=newcmap)
        scalarMap.set_array([minmin,maxmax])
        
        cbar = plt.colorbar(scalarMap,ticks = np.arange(minmin,maxmax+1))
        lblvar = list(range(sspikemin,sspikemin+sspike))+[' ']+list(range(nspikemin+nspike-1,nspikemin-1,-1))
        cbar.ax.set_yticklabels(lblvar)
        
        for sec in self.sections:
            sec_nr = self.sections[sec][2]
            for ii in range(len(self.lines[sec])):
                vsec = var_to_show[sec_nr+ii]
                cval = scalarMap.to_rgba(vsec)
                self.lines[sec][ii].set_color(cval)
        
#        title('Location-dependence evoking spikes',fontsize=25)
#        text(-350,150,'NMDA spike',color='r',fontsize=20)
#        text(-350,100,'Somatic spike',color='b',fontsize=20)
#        axis('off')  
        
        

     
    def v_record(self, neuron):
        """Set a monitor for the voltage of all segments"""
        return br2.StateMonitor(neuron, 'v', record=True)
        
    def run(self,morphodata):
        """run """
        
        # Set Biophysics
        neuron = self.biophysics(morphodata)
        # Record in every section
        monitor = self.v_record(neuron)
        
        # Initialize the model
        neuron.v = EL        
        neuron.I = 0.*nA
        neuron.I[0] = 0.2*nA
        
        #run
        br2.run(.5*ms)
        
        #store data in sec[1]
        for sec in self.sections.values():  
            kk = sec[2]
            sec[1] = monitor.v[kk:kk+len(sec[0].x)]
#            sec[1] = monitor.gKL[kk:kk+len(sec[0].x)]
        return monitor, neuron

if __name__ == "__main__":
#    test_MDL = '../0. Model/Branco2010_Morpho.swc'
#    morphodata = BrancoData    
#    distal_compartments = distal_compartments_Branco
#    proximal_compartments = proximal_compartments_Branco

    test_MDL = '../0. Model/Acker2008.swc'
    morphodata = AckerData
    distal_compartments = distal_compartments_Acker_eff
    proximal_compartments = proximal_compartments_Acker
    
    test_model = BRIANModel(test_MDL)
    
    
#    M, nrn = test_model.run(morphodata)
#    test_model.show_shape(fovx=200,fovy=500)
#    test_model.show_segm(segm_range=range(32,55,1))
    
#    dv,av = test_model.save_dist_vs_nr(1181)
#    tvect,div = test_model.calc_distCompartments(morphodata['basal'],dv)
    
#    test_model.show_property([5000]+list(morphodata['axon'])+list(morphodata['basal'])+list(morphodata['apical']))
    ii = 16
#    test_model.show_segm(segm_range=range(proximal_compartments_Acker[ii],
#        proximal_compartments_Acker[ii]+3))#
    test_model.show_segm(segm_range=distal_compartments,colorv='r',segm_range2=proximal_compartments,colorv2='b')
    
