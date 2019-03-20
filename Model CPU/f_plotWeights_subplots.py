
from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt


def  f_plotWeights_subplots( weightData, NrON,maxW):
    #plotWeights: plot the synaptic weights i.f.o. time
    #   Detailed explanation goes here
    
        # set fonts
        titleFont = 34
        normalFont = 34
        
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        
        location1p2d = 1
    
        proxDist = (location1p2d-1)*NrON
        weightData2 = weightData[proxDist:NrON+proxDist, :]
        
        
        for diagVar in range(NrON):
            weightData2[diagVar,diagVar] = -0.1
        
        pt1 = ax1.pcolor(weightData2, cmap='Greys', vmin=0, vmax=maxW)
            
        location1p2d = 2
        
        proxDist = (location1p2d-1)*NrON
        weightData2 = weightData[proxDist:NrON+proxDist, :]
        
        
        for diagVar in range (NrON):
            weightData2[diagVar,diagVar] = -0.1
            
        
        pt2 = ax2.pcolor(weightData2, cmap='Greys', vmin=0, vmax=maxW)
        cbar1 = plt.colorbar(pt2)
        cbar1.set_clim(0,maxW)
#        cbar2 = plt.colorbar(pt1)
#        cbar2.set_clim(0,1)
        

    
    
