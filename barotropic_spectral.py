#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
from netCDF4 import Dataset
from scipy.ndimage.filters import minimum_filter, maximum_filter
import spharm


# Set parameters
Re = 6378100.              # Radius of earth (m)
M = 36
dt = 900.                  # Timestep (seconds)
ntimes = 480               # Number of time steps to integrate
time_idx = 0             # Which time in the input files to use for ICs
use_hyperdiffusion = True  # Whether or not to apply hyperdiffusion
damping_order = 1          # Order of dampening
nu = 1E-4                  # Dampening coefficient for hyperdiffusion
restart = None             # Filename of restart file to load
plot_output = True         # Whether or not to plot the output



def get_ncep_initial_conditions(uwinds='uwnd.2015.nc',vwinds='vwnd.2015.nc'):
    """ Load in the initial conditions from 
    NCAR NCEP Reanalysis netcdf files for
    uwind and vwind.
    returns a dictionary of :
        'u_in'   : u_in, 
        'v_in'   : v_in,
        'lons'   : longitude mesh grid,
        'lats'   : latitude mesh grid,
        'lamb'   : longitude radian mesh grid,
        'theta'  : latitude radian mesh grid,
        'dlamb'  : gradient of lambda,
        'dtheta' : gradient of theta,
        'f'      : coriolis paramter at all points
        '"""

    print("Loading initial conditions from NCEP netcdf:", uwinds, vwinds)
    #from scipy.io.netcdf import netcdf_file
    infile = Dataset(uwinds,'r')
    lev_idx = list(infile.variables['level'][:]).index(500.)

    # Sort out when we are in time
    epoch_time = datetime(1800,1,1,0)
    # Find hours since the epoch_time
    hours_since = infile.variables['time'][time_idx]
    start_time = epoch_time + timedelta(hours=hours_since)
    infile.close()
    #dump_output = start_time + timedelta(hours=36)

    print("START TIME:", start_time.strftime('%Y%m%d%H'))
    #print "DUMP TIME:", dump_output.strftime('%Y%m%d%H')

    # Check for restart and, if so, load in the initial state from that
    # Otherwise, look for the pickled restart file
    if restart is None:
        # Load the U winds
        with Dataset(uwinds,'r') as infile:
            # Stay away from the poles in the latitude dimension
            u_in = infile.variables['uwnd'][time_idx,lev_idx,1:-1,:]
            #u_in = squeeze(u_in)
            # Duplicate the last longitude column for periodicity
            ulast = u_in[:,0]
            uint = np.transpose(u_in)
            newu = np.vstack((uint,ulast))
            u_in = np.transpose(newu)
    
        # Similar for V
        with Dataset(vwinds,'r') as infile:
            v_in = infile.variables['vwnd'][time_idx,lev_idx,1:-1,:]
    
            #v_in = squeeze(v_in)
            vlast = v_in[:,0]
            vint = np.transpose(v_in)
            newv = np.vstack((vint,vlast))
            v_in = np.transpose(newv)

            # Get the lats and lons respecting the dimensions
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
        



    return {'u_in' : u_in,
            'v_in' : v_in,
            'lons' : lon_list,
            'lats' : lat_list,
            'start_time' : start_time}



# Initial conditions
#u = 25 * cos(theta) - 30 * cos(theta)**3 + 300 * sin(theta)**2 * cos(theta)**6
#v = zeros(shape(u))

def integrate(init_cond,bmaps, ntimes=480):
    """ Function that integrates the barotropic model using
    spherical harmonics
    
    Input: 
    init_cond : dictionary of intial conditions
        containing u and v initial conditions, the latitude
        and longitudes describing the grids, and a starting time
    bmaps :  a dictionary of global and regional basemaps and
        projected coordinates
    ntimes : Number of timesteps to integrate


    """


    # Get the initial u and v wind fields
    u = init_cond['u_in']
    v = init_cond['v_in']
    ntrunc = len(init_cond['lats'])
    start_time = init_cond['start_time']

    # Create a radian grid
    lat_list_r = [x * np.pi/180. for x in init_cond['lats']]
    lon_list_r = [x * np.pi/180. for x in init_cond['lons']]


    # Meshgrid
    lons,lats = np.meshgrid(init_cond['lons'], init_cond['lats'])
    lamb, theta = np.meshgrid(lon_list_r, lat_list_r)


    dlamb = np.gradient(lamb)[1]
    dtheta = np.gradient(theta)[0]


    # Here is the Coriolis parameter
    f = 2 * 7.292E-5 * np.sin(theta)



    # Set up the spherical harmonic transform object
    s = spharm.Spharmt(len(init_cond['lons']),len(init_cond['lats']),rsphere=Re,gridtype='regular',legfunc='computed')

    # Use the object to plot the initial conditions
    # First convert to vorticity using spharm object
    vort_spec, div_spec = s.getvrtdivspec(u,v)
    div_spec = np.zeros(vort_spec.shape)  # Only want non-divergent part of wind 
    # Re-convert this to u-v winds to get the non-divergent component
    # of the wind field
    u,v = s.getuv(vort_spec,div_spec)
    # Use these winds to get the streamfunction (psi) and 
    # velocity potential (chi)
    psi,chi = s.getpsichi(u,v)
    # Convert the spectral vorticity to grid
    vort_now = s.spectogrd(vort_spec)

    # Plot Initial Conditions
    curtime = start_time
    plot_figures(0,curtime,u,v,vort_now,psi,bmaps)

    # Now loop through the timesteps
    for n in xrange(ntimes):
        
        # Compute spectral vorticity from u and v wind
        vort_spec, div_spec = s.getvrtdivspec(u,v)  
        
        # Now get the actual vorticity
        vort_now = s.spectogrd(vort_spec)
        
        div = np.zeros(vort_now.shape)  # Divergence is zero in barotropic vorticity

        # Here we actually compute vorticity tendency
        # Compute tendency with beta as only forcing
        vort_tend_rough = -2. * 7.292E-5/(Re**2) * d_dlamb(psi,dlamb) -\
                Jacobian(psi,vort_now,theta,dtheta,dlamb)
        
        # Apply hyperdiffusion if requested for smoothing
        if use_hyperdiffusion:
            vort_tend = add_hyperdiffusion(s,vort_now,vort_tend_rough, ntrunc)
        else:
            vort_tend = vort_tend_rough

        if n == 0:
            # First step just do forward difference
            # Vorticity at next time is just vort + vort_tend * dt
            vort_next = vort_now + vort_tend[:,:,0] * dt
        else:
            # Otherwise do leapfrog
            vort_next = vort_prev + vort_tend[:,:,0] * 2 * dt 


        # Invert this new vort to get the new psi (or rather, uv winds)
        # First go back to spectral space
        vort_spec = s.grdtospec(vort_next)
        div_spec = s.grdtospec(div)

        # Now use the spharm methods to get new u and v grid
        u,v = s.getuv(vort_spec,div_spec)
        psi,chi = s.getpsichi(u,v)
        #raw_input()
        
        # Change vort_now to vort_prev
        # and if not first step add Robert filter 
        # (from Held barotropic model)
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
            print("Plotting hour", (n+1)/4)
            plot_figures((n+1)/4, curtime, u, v, vort_next, psi, bmaps)
        #if dump_output == curtime:
        #    import cPickle
        #    print "Dumping output"
        #    outfile = open('%s_output.pickle' % (dump_output.strftime('%Y%m%d%H')),'w')
        #    cPickle.dump((u,v,lat_list,lon_list),outfile)
        #    outfile.close()
        #    exit()

def add_hyperdiffusion(s, cur_vort, vort_tend, ntrunc):
    """ Add spectral hyperdiffusion and return a new
    vort_tend """
    # Convert to spectral grids
    vort_spec = s.grdtospec(cur_vort)
    vort_tend_spec = s.grdtospec(vort_tend)
    total_length = vort_spec.shape[0]

    # Reshape to 2-d arrayw
    vort_spec = np.reshape(vort_spec,(ntrunc,-1))
    vort_tend_spec = np.reshape(vort_tend_spec,(ntrunc,-1))
    new_vort_tend_spec = np.array(vort_tend_spec,dtype=np.complex)

    DES = compute_dampening_eddy_sponge(vort_tend_spec.shape)

    # Now loop through each value
    for n in xrange(vort_spec.shape[1]):
        for m in xrange(vort_spec.shape[0]):
            num = vort_tend_spec[m,n] - DES[m,n] * vort_spec[m,n]
            den = np.complex(1.,0) + DES[m,n] * np.complex(dt,0.)

            new_vort_tend_spec[m,n] = num / den

    # Reshape the new vorticity tendency and convert back to grid
    new_vort_tend_spec = np.reshape(new_vort_tend_spec, (total_length,-1))

    new_vort_tend = s.spectogrd(new_vort_tend_spec)

    return new_vort_tend


def compute_dampening_eddy_sponge(fieldshape):
    """ Computes the eddy sponge by getting the eigenvalues 
    of the Laplacian for each spectral coefficient and 
    multiplying them by a dampening factor nu 
    (specified at top of script) 
    From Held and Suarez
    """
    
    # Need some arrays
    m_vals = range(fieldshape[0])
    n_vals = range(fieldshape[1])

    spherical_wave = np.zeros(fieldshape)
    eigen_laplacian = np.zeros(fieldshape)
    
    fourier_inc = 1

    for n in n_vals:
        for m in m_vals:
            fourier_wave = m * fourier_inc
            spherical_wave[m,n] = fourier_wave + n

    # Now for the laplacian
    eigen_laplacian = np.divide(np.multiply(spherical_wave,np.add(spherical_wave,1.)),Re**2)

    # Dampening Eddy Sponge values
    DES = np.multiply(eigen_laplacian, nu) 
            # Do complex divison
    #DES = multiply(DES,2*dt)
    #for n in n_vals:
    #    for m in m_vals:
    #        print DES[m,n], complex(DES[m,n],0.)
    #        DES_cpx[m,n] = complex(DES[m,n],0.)
    DES_cpx = np.array(DES, dtype=np.complex)

    return DES_cpx
 




def create_basemaps(lons,lats):
    """ Setup global and regional basemaps for eventual plotting """
    print("Creating basemaps for plotting")

    long, latg = np.meshgrid(lons,lats)

    # Set up a global map
    bmap_globe = Basemap(projection='merc',llcrnrlat=-70, urcrnrlat=70,\
               llcrnrlon=0,urcrnrlon=360,lat_ts=20,resolution='c')
    xg,yg = bmap_globe(long,latg)
   
    # Set up a regional map (currently North America)
    bmap_reg = Basemap(projection='eqdc',lon_0=-107,lat_0=50,lat_1=45.,lat_2=55.,width=12000000,height=9000000,resolution='l')
    xr,yr = bmap_reg(long,latg)

    return {'global' : bmap_globe, 
            'global_x' : xg, 
            'global_y' : yg,
            'regional' : bmap_reg,
            'regional_x' : xr, 
            'regional_y' : yr, 
            }

def plot_figures(n,curtime,u,v,vort,psi,bmaps):
    """ Make global and regional plots"""
    
    
    plt.figure(figsize=(12,12))
    plt.contourf(bmaps['global_x'],bmaps['global_y'],vort,np.linspace(-1.E-4,1.E-4,10), cmap=matplotlib.cm.RdBu,extend='both',antialiasing=False)
    plt.hold(True)
    plt.quiver(bmaps['global_x'],bmaps['global_y'],u,v)
    bmaps['global'].drawcoastlines()
    plt.title('Zeta and wind at %d hours (%s)' % (n,curtime.strftime('%Y%m%d%H')))
    plt.savefig('globe_plot_%03d.png' % (n), bbox_inches='tight')
    plt.close()

    # North America plot with geopotential height
    phi = np.divide(psi * 7.292E-5, 9.81)
    plt.figure(figsize=(10,8))
    # Calculate wind magnitude
    windmag = np.sqrt(u**2 + v**2)
    

    plt.contourf(bmaps['regional_x'],bmaps['regional_y'],windmag,np.arange(15,48,3),cmap=matplotlib.cm.jet,extend='max',antialiasing=False)
    plt.hold(True)
    hgtconts = plt.contour(bmaps['regional_x'],bmaps['regional_y'],phi,np.linspace(-500,500,26),colors='k')
    bmaps['regional'].drawcoastlines()
    bmaps['regional'].drawcountries()
    bmaps['regional'].drawstates()

    plt.title('Hgt anomalies and wind at %d hours (%s)' %(n,curtime.strftime('%Y%m%d%H')))
    print("Saving hour", n)
    plt.savefig('reg_plot_%03d.png' % (n), bbox_inches='tight')
    plt.close()

    #os.system('mv *.png ~/public_html/research/barotropic')              


def extrema(mat,mode='wrap',window=10):
    """ Function to find the indices of local extrema in the
    input array (adapted from mpl-Basemap cookbook"""
    mn = minimum_filter(mat, size=window, mode=mode)
    mx = maximum_filter(mat, size=window, mode=mode)

    return nonzero(mat == mn), nonzero(mat==mx)


def d_dlamb(field,dlamb):
    """ Finds a finite-difference approximation to gradient in
    the lambda (longitude) direction"""
    out = np.divide(np.gradient(field)[1],dlamb) 
    return out

def d_dtheta(field,dtheta):
    """ Finds a finite-difference approximation to gradient in
    the theta (latitude) direction """
    out = np.divide(np.gradient(field)[0],dtheta)
    return out

def divergence_spher(u,v,theta,dtheta,dlamb):
    """ Compute the divergence field in spherical coordinates """
    term1 = 1./(Re*np.cos(theta)) * d_dlamb(u,dlamb)
    term2 = 1./(Re*np.cos(theta)) * d_dtheta(v * np.cos(theta)),dtheta
    return term1 + term2  

def vorticity_spher(u,v,theta,dtheta,dlamb):
    """ Computes normal component of vorticity in spherical
    coordinates """
    term1 = 1./(Re*np.cos(theta)) * d_dlamb(v,dlamb)
    term2 = 1./(Re*np.cos(theta)) * d_dtheta(u*np.cos(theta),dtheta)
    return term1 - term2

def wind_stream(psi,theta,dtheta,dlamb):
    """ Compute u and v winds from streamfunction in spherical
    coordinates """
    u = -1./Re * d_dtheta(psi,dtheta)
    v = 1./(Re * np.cos(theta)) * d_dlamb(psi,dlamb)
    return u,v

def Jacobian(A,B,theta,dtheta,dlamb):
    """ Returns the Jacobian of two fields in spherical coordinates """
    term1 = d_dlamb(A,dlamb) * d_dtheta(B,dtheta)
    term2 = d_dlamb(B,dlamb) * d_dtheta(A,dtheta)
    return 1./(Re**2 * np.cos(theta)) * (term1 - term2)


if __name__ == '__main__':
    ics = get_ncep_initial_conditions()
    bmaps = create_basemaps(ics['lons'],ics['lats'])
    integrate(ics,bmaps,ntimes=ntimes)





