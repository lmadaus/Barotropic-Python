#/usr/bin/env python
from __future__ import print_function, division

import sys
import os

from datetime import datetime, timedelta

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic

from numpy import *
from numpy import max as maxval
from numpy import min as minval

from netCDF4 import Dataset

from scipy.ndimage.filters import minimum_filter, maximum_filter

import spharm
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-i','--index',dest='timedex',default='-1',help='Time index to use')

args = parser.parse_args()

time_idx = int(args.timedex)


# Set parameters
#nlon = 360
#nlat = int(nlon/2)
Re = 6378100.  # Radius of earth (m)
M = 36
dt = 900. # Timestep (seconds)
nu = 1E-4
damping_order = 1
use_hyperdiffusion = True
restart = None
plot_output = True 

#time_idx = -1

# Load in the initial conditions from netcdf file
#from scipy.io.netcdf import netcdf_file
infile = Dataset('uwnd.2014.nc','r')
lev_idx = list(infile.variables['level'][:]).index(500.)

# Sort out when we are in time
epoch_time = datetime(2800,1,1,0)
# Find hours since the epoch_time
hours_since = infile.variables['time'][time_idx]
start_time = epoch_time + timedelta(hours=hours_since)

#dump_output = start_time + timedelta(hours=36)

print("START TIME:", start_time.strftime('%Y%m%d%H'))
#print "DUMP TIME:", dump_output.strftime('%Y%m%d%H')
if restart == None:
    infile = Dataset('uwnd.2014.nc','r')
    u_in = infile.variables['uwnd'][time_idx,lev_idx,1:-1,:]
    u_in = squeeze(u_in)
    #u_in = multiply(u_in,infile.variables['uwnd'].scale_factor) + infile.variables['uwnd'].add_offset
    ulast = u_in[:,0]
    uint = transpose(u_in)
    newu = vstack((uint,ulast))
    u_in = transpose(newu)

    infile = Dataset('vwnd.2014.nc','r')
    v_in = infile.variables['vwnd'][time_idx,lev_idx,1:-1,:]
    v_in = squeeze(v_in)
    #v_in = multiply(v_in,infile.variables['vwnd'].scale_factor) + infile.variables['vwnd'].add_offset
    vlast = v_in[:,0]
    vint = transpose(v_in)
    newv = vstack((vint,vlast))
    v_in = transpose(newv)

    # Get the lats and lons
    lat_list = list(infile.variables['lat'][1:-1])
    lon_list = list(infile.variables['lon'][:])
    lon_list.append(360.)
    #print lon_list
    #raw_input()
else:
    import cPickle
    infile = open(restart,'r')
    u_in,v_in,lat_list,lon_list = cPickle.load(infile)
    infile.close()
    


ntrunc = len(lat_list)

# Create a radian grid
lat_list_r = [x * pi/180. for x in lat_list]
lon_list_r = [x * pi/180. for x in lon_list]


# Meshgrid
lons,lats = meshgrid(lon_list, lat_list)
lamb, theta = meshgrid(lon_list_r, lat_list_r)


dlamb = gradient(lamb)[1]
dtheta = gradient(theta)[0]


# Here is the Coriolis parameter
f = 2 * 7.292E-5 * sin(theta)



# Set up the spherical harmonic transform object
s = spharm.Spharmt(len(lon_list),len(lat_list),rsphere=Re,gridtype='regular',legfunc='computed')

def integrate(u,v):
    # Initial conditions
    #u = 25 * cos(theta) - 30 * cos(theta)**3 + 300 * sin(theta)**2 * cos(theta)**6
    #v = zeros(shape(u))
    # first convert to vorticity
    vort_spec, div_spec = s.getvrtdivspec(u,v)
    div_spec = zeros(shape(vort_spec))  # Only want non-divergent part of wind 
    u,v = s.getuv(vort_spec,div_spec)
    psi,chi = s.getpsichi(u,v)
    vort_now = s.spectogrd(vort_spec)

    # Plot first figures
    curtime = start_time
    plot_figures(0,curtime,u,v,vort_now,psi)

    for n in range(480):
        
        # Compute spectral vorticity from u and v wind
        vort_spec, div_spec = s.getvrtdivspec(u,v)  
        
        # Now get the actual vorticity
        vort_now = s.spectogrd(vort_spec)
        
        #print vort
        div = zeros(shape(vort_now))  # Divergence is zero in barotropic vorticity
        # Compute tendency with beta as only forcing
        vort_tend_rough = -2. * 7.292E-5/(Re**2) * d_dlamb(psi) - Jacobian(psi,vort_now)
        
        if use_hyperdiffusion:
            vort_tend = add_hyperdiffusion(vort_now,vort_tend_rough)
        else:
            vort_tend = vort_tend_rough

        if n == 0:
            # First step just do forward difference
            # Vorticity at next time is just vort + vort_tend * dt
            vort_next = vort_now + vort_tend * dt
        else:
            # Otherwise do leapfrog
            vort_next = vort_prev + vort_tend * 2 * dt 


        # Invert this new vort to get the new psi (or rather, uv winds)
        # First go back to spectral space
        vort_spec = s.grdtospec(vort_next)
        div_spec = s.grdtospec(div)

        # Now use the function to get new u and v grid
        u,v = s.getuv(vort_spec,div_spec)
        psi,chi = s.getpsichi(u,v)
        #raw_input()
        
        # Change vort_now to vort_prev
        # and if not first stepadd Robert filter (from Held barotropic model)
        # to dampen out crazy modes
        r = 0.2
        if n == 0:
            vort_prev = vort_now
        else:
            vort_prev = (1.-2.*r)*vort_now + r*(vort_next + vort_prev) 
        curtime = start_time + timedelta(hours = ((n+1)*dt/3600.))

        # Output every six hours
        if (n+1) % 24 == 0 and plot_output:
            # Go from psi to geopotential
            #phi = divide(psi * f,9.81)
            print "Saving time step", (n+1)/4
            # plot this every so many hours
            plot_figures((n+1)/4,curtime,u,v,vort_next,psi)
        #if dump_output == curtime:
        #    import cPickle
        #    print "Dumping output"
        #    outfile = open('%s_output.pickle' % (dump_output.strftime('%Y%m%d%H')),'w')
        #    cPickle.dump((u,v,lat_list,lon_list),outfile)
        #    outfile.close()
        #    exit()

def add_hyperdiffusion(cur_vort, vort_tend):
    # Add spectral hyperdiffusion and return a new
    # vort_tend
    # Convert to spectral grids
    vort_spec = s.grdtospec(cur_vort)
    vort_tend_spec = s.grdtospec(vort_tend)
    total_length = shape(vort_spec)[0]

    # Reshape to 2-d arrayw
    vort_spec = reshape(vort_spec,(ntrunc,-1))
    vort_tend_spec = reshape(vort_tend_spec,(ntrunc,-1))
    new_vort_tend_spec = array(vort_tend_spec,dtype=complex)

    DES = compute_dampening_eddy_sponge(shape(vort_tend_spec))

    # Now loop through each value
    for n in range(shape(vort_spec)[1]):
        for m in range(shape(vort_spec)[0]):
            #num = vort_tend_spec[m,n] - DES[m,n] * vort_spec[m,n]
            #den = complex(1.,0) + DES[m,n]
            #print "Numer, denom"
            #print num, den
            # Do complex divison


            #real_part = (num.real * den.real + num.imag * den.imag) / (den.real**2 + den.imag**2)
            #imag_part = (num.imag * den.real - num.real * den.imag) / (den.real**2 + den.imag**2)
            #print "real, imag"
            #print real_part, imag_part
            #new_vort_tend_spec[m,n] = complex(real_part, imag_part)
            num = vort_tend_spec[m,n] - DES[m,n] * vort_spec[m,n]
            den = complex(1.,0) + DES[m,n] * complex(dt,0.)

            new_vort_tend_spec[m,n] = num / den

    # Reshape the new vorticity tendency and convert back to grid
    new_vort_tend_spec = reshape(new_vort_tend_spec, (total_length,-1))

    new_vort_tend = s.spectogrd(new_vort_tend_spec)

    return new_vort_tend


def compute_dampening_eddy_sponge(fieldshape):
    # Computes the eddy sponge by getting the eigenvalues 
    # of the Laplacian for
    # each spectral coefficient and multiplying them by
    # some dampening factor nu (specified at top of script)
    
    # Need some arrays
    m_vals = range(fieldshape[0])
    n_vals = range(fieldshape[1])

    spherical_wave = zeros(fieldshape)
    eigen_laplacian = zeros(fieldshape)
    
    fourier_inc = 1

    for n in n_vals:
        for m in m_vals:
            fourier_wave = m * fourier_inc
            spherical_wave[m,n] = fourier_wave + n

    # Now for the laplacian
    eigen_laplacian = divide(multiply(spherical_wave,add(spherical_wave,1.)),Re**2)

    # Dampening Eddy Sponge values
    DES = multiply(eigen_laplacian, nu) 
            # Do complex divison
    #DES = multiply(DES,2*dt)
    #for n in n_vals:
    #    for m in m_vals:
    #        print DES[m,n], complex(DES[m,n],0.)
    #        DES_cpx[m,n] = complex(DES[m,n],0.)
    DES_cpx = array(DES, dtype=complex)

    return DES_cpx
 







def plot_figures(n,curtime,u,v,vort,psi):
    plt.figure(figsize=(12,12))

    # Setup a basemap for eventual plotting
    bmap_globe = Basemap(projection='merc',llcrnrlat=-70, urcrnrlat=70,\
               llcrnrlon=0,urcrnrlon=360,lat_ts=20,resolution='c')

    # Make the field cyclic

    #lon_list = list(infile.variables['lon'][:])
    #vort, lon_list = addcyclic(vort,lon_list)
    #lons = meshgrid(lon_list,lat_list)
    x,y = bmap_globe(lons,lats)

    plt.contourf(x,y,vort,linspace(-1.E-4,1.E-4,10), cmap=matplotlib.cm.RdBu,extend='both',antialiasing=False)
    plt.hold(True)
    plt.quiver(x,y,u,v)
    bmap_globe.drawcoastlines()
    plt.title('Zeta and wind at %d hours (%s)' % (n,curtime.strftime('%Y%m%d%H')))
    plt.savefig('globe_plot_%03d.png' % (n), bbox_inches='tight')
    plt.close()

    # North America plot with geopotential height
    phi = divide(psi * 7.292E-5, 9.81)
    plt.figure(figsize=(10,8))
    # Calculate wind magnitude
    windmag = sqrt(u**2 + v**2)
    

    bmap_na = Basemap(projection='eqdc',lon_0=-107,lat_0=50,lat_1=45.,lat_2=55.,width=12000000,height=9000000,resolution='l')
    xn,yn = bmap_na(lons,lats)
    plt.contourf(xn,yn,windmag,arange(15,48,3),cmap=matplotlib.cm.jet,extend='max',antialiasing=False)
    #plt.contourf(xn,yn,phi,linspace(-500,500,25),cmap=matplotlib.cm.RdBu_r,extend='both',antialiasing=False)
    #plt.colorbar()
    #plt.contourf(x,y,vort_next,linspace(-1.E-4,1.E-4,10), cmap=matplotlib.cm.RdBu,extend='both')
    plt.hold(True)
    #local_min, local_max = extrema(phi,mode='wrap',window=2)
    hgtconts = plt.contour(xn,yn,phi,linspace(-500,500,26),colors='k')
    #plt.clabel(hgtconts,fontsize='x-small',inline_spacing = -1, fmt = '%4.0f')
    #plt.quiver(xn,yn,u,v)
    bmap_na.drawcoastlines()
    bmap_na.drawcountries()
    bmap_na.drawstates()

    # Now overlay calculated highs and lows
    #xlows = x[local_min]; xhighs = x[local_max]
    #ylows = y[local_min]; yhighs = y[local_max]
    #yoffset = 0.022*(bmap_na.ymax - bmap_na.ymin)
    #xyplotted = []
    #dmin = yoffset
    # Plot the lows
    #for x,y in zip(xlows,ylows):
    #    if x < bmap_na.xmax and x > bmap_na.xmin and y < bmap_na.ymax and y > bmap_na.ymin:
    #        dist = [sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
    #        if not dist or min(dist) > dmin:
    #            plt.text(x,y,'L',fontsize=14,fontweight='bold',ha='center',va='center',color='b')
    #            xyplotted.append((x,y))
    # Now plot the highs
    #xyplotted = []
    #for x,y in zip(xhighs,yhighs):
    #    if x < bmap_na.xmax and x > bmap_na.xmin and y < bmap_na.ymax and y > bmap_na.ymin:
    #        dist = [sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
    #        if not dist or min(dist) > dmin:
    #            plt.text(x,y,'H',fontsize=14,fontweight='bold',ha='center',va='center',color='r')
    #            xyplotted.append((x,y))
   
    


    plt.title('Hgt anomalies and wind at %d hours (%s)' %(n,curtime.strftime('%Y%m%d%H')))
    print "Saving time step", n
    plt.savefig('na_plot_%03d.png' % (n), bbox_inches='tight')
    plt.close()

    os.system('mv *.png ~/public_html/research/barotropic')              


def extrema(mat,mode='wrap',window=10):
    """ Function to find the indices of local extrema in the
    input array (adapted from mpl-Basemap cookbook"""
    mn = minimum_filter(mat, size=window, mode=mode)
    mx = maximum_filter(mat, size=window, mode=mode)

    return nonzero(mat == mn), nonzero(mat==mx)


def d_dlamb(field):
    # Finds a finite-difference approximation to gradient in
    # the lambda (longitude) direction
    out = divide(gradient(field)[1],dlamb) 
    return out

def d_dtheta(field):
    # Finds a finite-difference approximation to gradient in
    # the theta (latitude) direction
    out = divide(gradient(field)[0],dtheta)
    return out

def divergence_spher(u,v):
    # Compute the divergence field in spherical coordinates
    term1 = 1./(Re*cos(theta)) * d_dlamb(u)
    term2 = 1./(Re*cos(theta)) * d_dtheta(v * cos(theta))
    return term1 + term2  

def vorticity_spher(u,v):
    # Computes normal component of vorticity in spherical
    # coordinates
    term1 = 1./(Re*cos(theta)) * d_dlamb(v)
    term2 = 1./(Re*cos(theta)) * d_dtheta(u*cos(theta))
    return term1 - term2

def wind_stream(psi):
    # Compute u and v winds from streamfunction in spherical
    # coordinates
    u = -1./Re * d_dtheta(psi)
    v = 1./(Re * cos(theta)) * d_dlamb(psi)
    return u,v

def Jacobian(A,B):
    # Returns the Jacobian of two fields in spherical coordinates
    term1 = d_dlamb(A) * d_dtheta(B)
    term2 = d_dlamb(B) * d_dtheta(A)
    return 1./(Re**2 * cos(theta)) * (term1 - term2)


if __name__ == '__main__':
    integrate(u_in,v_in)





