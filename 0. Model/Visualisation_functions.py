
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
import matplotlib.colors as colorz
import matplotlib.cm as clrm
import brian2 as br2
from brian2 import uF, cm, um, ohm, ms, siemens, mV, nA, us,psiemens

def get_swc(filename):
    '''
    Reads a SWC file containing a neuronal morphology.
    Store compartment number of the file in a dictionary with the coordinates as key.
    This dictionary will be used later to assign the compartment number to the sections of the neuron.
    
    Information below from http://www.mssm.edu/cnic/swc.html    
    SWC File Format
    
    The format of an SWC file is fairly simple. It is a text file consisting of a header with various fields beginning with a # character, and a series of three dimensional points containing an index, radius, type, and connectivity information. The lines in the text file representing points have the following layout. 
    n T x y z R P
    n is an integer label that identifies the current point and increments by one from one line to the next.
    T is an integer representing the type of neuronal segment, such as soma, axon, apical dendrite, etc. The standard accepted integer values are given below:
    * 0 = undefined
    * 1 = soma
    * 2 = axon
    * 3 = dendrite
    * 4 = apical dendrite
    * 5 = fork point
    * 6 = end point
    * 7 = custom

    x, y, z gives the cartesian coordinates of each node.
    R is the radius at that node.
    P indicates the parent (the integer label) of the current point or -1 to indicate an origin (soma).

    By default, the soma is assumed to have spherical geometry.
    '''
    
    lines = open(filename).read().splitlines() #create array of the lines in the swc file
    types = ['undefined', 'soma', 'axon', 'dend', 'apical', 'fork',
             'end', 'custom']
    previousn = -1

    segment_swc = {}  # list of segments, coordinates as in SWC file
    segment = {}  # list of segments, coordinates are averages of initial and final value
    
    for line in lines:
        if line[0] != '#':  
            numbers = line.split() #create array of the values in a line
            n = int(numbers[0]) - 1 # 0-based indexing
            T = types[int(numbers[1])]
            x = int(round(float(numbers[2])*1e3))
            y = int(round(float(numbers[3])*1e3))
            z = int(round(float(numbers[4])*1e3))
            R = float(numbers[5])*1e-6
            P = int(numbers[6]) - 1  # 0-based indexing
            if (n != previousn + 1):
                raise ValueError( "Bad format in file " + filename)
            if (x,y,z) in segment_swc:
                print('ERROR')
                error_n = segment_swc[(x,y,z)]
                print('Segmentswc ' + str(n) + ' same xyz as ' + str(error_n) + ' !!')
            else:
                if n > 0:
                    b = [key for key, value in segment_swc.items() if value == P][0]   
                    #calculate half-way distance between parent section and child section:
                    xhalf = int(round(0.5*(b[0]+x)))
                    yhalf = int(round(0.5*(b[1]+y)))
                    zhalf = int(round(0.5*(b[2]+z)))
                    
                    if (xhalf,yhalf,zhalf) in segment:
                        print('ERROR')
                        error_n = segment[(xhalf,yhalf,zhalf)]
                        print('Segment ' + str(n) + ' same xyz as ' + str(error_n) + ' !!') 
                                 
                    segment[(xhalf,yhalf,zhalf)] = n
                    segment_swc[(x,y,z)] = n
                else:
                    segment[(x,y,z)] = n
                    segment_swc[(x,y,z)] = n
            previousn = n
            
    return segment,segment_swc

        
        

def add_line(ax, lines, sec):
    
    """Add a line to a 2-D shape plot"""
    diams = br2.asarray(sec.diameter)
    xs = br2.asarray(sec.x)*1e6
    ys = br2.asarray(sec.y)*1e6
    # include the last point of the parent section:
    xs = [sec.f_x*1e6] + xs.tolist()
    ys = [sec.f_y*1e6] + ys.tolist()

    linewdth = 1.5   
    
    for i in range(len(xs[1:])):
        if sec.name != "soma":
            if sec.name in lines:
                lines[sec.name].append(ax.plot(xs[i:i+2],
                                      ys[i:i+2],
                                      '-k', color='black',
                                      alpha=1,
                                      linewidth=linewdth*diams[i]*1e6)[0])
            else:
                lines[sec.name] = [ax.plot(xs[i:i+2],
                                      ys[i:i+2],
                                      '-k', color='black',
                                      alpha=1,
                                      linewidth=linewdth*diams[i]*1e6)[0]]
        else:
            # Plot soma segment(s) as a circle(s)
            circle = plt.Circle((sec.f_x*1e6, sec.f_y*1e6), 0.5*diams[i]*1e6, color='black', alpha=1)
            lines[sec.name] = [circle]
            # Add the soma
            ax.add_artist(circle)                  
    return lines

def add_line3d(ax, lines, sec):
    """Add a line to a 3-D shape plot"""
    diams = br2.asarray(sec.diameter)
    xs = br2.asarray(sec.x)*1e6
    ys = br2.asarray(sec.y)*1e6
    zs = br2.asarray(sec.z)*1e6
    # include the last point of the parent section
    xs = [sec.f_x*1e6] + xs.tolist()
    ys = [sec.f_y*1e6] + ys.tolist()
    zs = [sec.f_z*1e6] + zs.tolist()

    cval = 'black'
    for i in range(len(xs[1:])):
        diam = diams[i]*1e2
        if sec.name != "soma":
            if sec.name in lines:
                lines[sec.name].append(ax.plot(xs[i:i+2],
                                      ys[i:i+2],
                                      zs[i:i+2],
                                      '-k', color=cval,
                                      alpha=1,
                                      linewidth=diam*1e4)[0])
            else:
                lines[sec.name] = [ax.plot(xs[i:i+2],
                                      ys[i:i+2],
                                      zs[i:i+2],
                                      '-k', color=cval,
                                      alpha=1,
                                      linewidth=diam*1e4)[0]]
        else:
            # Plot soma segment(s) as a circle(s)
            circle = plt.Circle((0, 0), diam, color=cval, alpha=1)
            lines[sec.name] = [circle]
            # Add the soma
            ax.add_artist(circle)
    return lines

def calc_dist(distances, sec):
    """Calculate the distance from soma"""
    xs = br2.asarray(sec.x)*1e6
    ys = br2.asarray(sec.y)*1e6
    zs = br2.asarray(sec.z)*1e6
    #the last point of the parent section:
    xs = [sec.f_x*1e6] + xs.tolist()
    ys = [sec.f_y*1e6] + ys.tolist()
    zs = [sec.f_z*1e6] + zs.tolist()
    #distance to soma of the parent section:
    parent_dist = sec.dist
    #initiate the distance of current section:
    sect_dist = parent_dist
    for i in range(len(xs[1:])): #length should be one less, since below we have i+1
        sect_dist += np.sqrt((xs[i]-xs[i+1])**2+(ys[i]-ys[i+1])**2+(zs[i]-zs[i+1])**2) 
        if sec.name in distances:
            distances[sec.name].append(sect_dist)
        else:
            distances[sec.name] = [sect_dist]
        
    return distances
    
def show_property2(tm, var_to_show):
        """Show a 2D plot of the morphology, highlight the distribution of a parameter 'var_to_show' """
        data = var_to_show
        
        cmap = plt.get_cmap('rainbow') #coolwarm, seismic,bwr,afmhot
        
        minmin = np.amin(data)
        maxmax = np.amax(data)
        
        cNorm = colorz.Normalize(vmin=minmin, vmax= maxmax ) # np.amin(data)  np.amax(data)
        
        scalarMap = clrm.ScalarMappable(norm=cNorm, cmap=cmap)
        scalarMap.set_array([minmin,maxmax])
        plt.colorbar(scalarMap)
        
        for sec in tm.sections:
            sec_nr = tm.sections[sec][2]
            for ii in range(len(tm.lines[sec])):
                vsec = var_to_show[sec_nr+ii]
                cval = scalarMap.to_rgba(vsec)
                tm.lines[sec][ii].set_color(cval)
                
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap',
                    somspike=5,nmdaspike=5):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])
    indv = 0

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        
        steps1 = 128/somspike
        steps2 = 128/nmdaspike
        vec1 = np.arange(steps1/2,128,steps1)
        vec2 = np.arange(128+steps2/2,256,steps2)
        cvec = np.append(vec1,vec2,axis=0)/257
        
        if indv+1 < len(cvec)-1:
            if ri>cvec[indv+1]:
                indv = indv+1
        if ri>=cvec[indv] and ri<cvec[indv+1]:
            avgi = np.mean([cvec[indv],cvec[indv+1]])
            r,g,b,a = cmap(avgi)
        elif ri<cvec[indv]:
            r,g,b,a = cmap(0.001)
        elif ri>=cvec[indv+1]:
            r,g,b,a = cmap(.999)
        if indv==somspike-1:
            r=.5
            g=.5
            b=.5
        
#        if ri >120/257 and ri<130/257:
#            r=.1
#            g=.1
#            b=.1

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colorz.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap