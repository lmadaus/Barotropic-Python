#!/usr/bin/env python
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import spharm
import namelist as nl


def integrate(init_cond, bmaps):
    """ Function that integrates the barotropic model using
    spherical harmonics
    
    Input: 
    init_cond : dictionary of intial conditions
        containing u and v initial conditions, the latitude
        and longitudes describing the grids, and a starting time
    bmaps :  a dictionary of global and regional basemaps and
        projected coordinates
    """


    # Get the initial u and v wind fields
    u = init_cond['u_in']
    v = init_cond['v_in']
    if nl.M is None:
        ntrunc = len(init_cond['lats'])
    else:
        ntrunc = nl.M
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
    f = 2 * nl.omega * np.sin(theta)


    # Set up the spherical harmonic transform object
    s = spharm.Spharmt(len(init_cond['lons']),len(init_cond['lats']),rsphere=nl.Re,gridtype='regular',legfunc='computed')

    # Use the object to plot the initial conditions
    # First convert to vorticity using spharm object
    vort_spec, div_spec = s.getvrtdivspec(u, v, ntrunc=ntrunc)
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
    for n in range(nl.ntimes):
        
        # Compute spectral vorticity from u and v wind
        vort_spec, div_spec = s.getvrtdivspec(u, v, ntrunc=ntrunc)
        
        # Now get the actual vorticity
        vort_now = s.spectogrd(vort_spec)
        
        div = np.zeros(vort_now.shape)  # Divergence is zero in barotropic vorticity

        # Here we actually compute vorticity tendency
        # Compute tendency with beta as only forcing
        vort_tend_rough = -2. * nl.omega/(nl.Re**2) * d_dlamb(psi,dlamb) -\
                Jacobian(psi,vort_now,theta,dtheta,dlamb)
        
        # Apply hyperdiffusion if requested for smoothing
        if nl.use_hyper:
            vort_tend = add_hyperdiffusion(s, vort_now, vort_tend_rough, ntrunc)
        else:
            vort_tend = vort_tend_rough

        if n == 0:
            # First step just do forward difference
            # Vorticity at next time is just vort + vort_tend * dt
            vort_next = vort_now + vort_tend[:,:] * nl.dt
        else:
            # Otherwise do leapfrog
            vort_next = vort_prev + vort_tend[:,:] * 2 * nl.dt 


        # Invert this new vort to get the new psi (or rather, uv winds)
        # First go back to spectral space
        vort_spec = s.grdtospec(vort_next, ntrunc=ntrunc)
        div_spec = s.grdtospec(div, ntrunc=ntrunc)

        # Now use the spharm methods to get new u and v grid
        u,v = s.getuv(vort_spec,div_spec)
        psi,chi = s.getpsichi(u,v)
        
        # Change vort_now to vort_prev
        # and if not first step add Robert filter 
        # (from Held barotropic model)
        # to dampen out crazy modes
        if n == 0:
            vort_prev = vort_now
        else:
            vort_prev = (1.-2.*nl.r)*vort_now + nl.r*(vort_next + vort_prev)
        cur_fhour = (n+1) * nl.dt / 3600.
        curtime = start_time + timedelta(hours = cur_fhour)

        # Output every [output_frequ] hours
        if cur_fhour % nl.output_freq == 0 and nl.plot_output:
            # Go from psi to geopotential
            print("Plotting hour", cur_fhour)
            plot_figures(int(cur_fhour), curtime, u, v, vort_next, psi, bmaps)
            

def add_hyperdiffusion(s, cur_vort, vort_tend, ntrunc):
    """ Add spectral hyperdiffusion and return a new
    vort_tend """
    # Convert to spectral grids
    vort_spec = s.grdtospec(cur_vort)
    vort_tend_spec = s.grdtospec(vort_tend)
    total_length = vort_spec.shape[0]

    # Reshape to 2-d array
    vort_spec = np.reshape(vort_spec,(ntrunc,-1))
    vort_tend_spec = np.reshape(vort_tend_spec,(ntrunc,-1))
    new_vort_tend_spec = np.array(vort_tend_spec,dtype=np.complex)

    DES = compute_dampening_eddy_sponge(vort_tend_spec.shape)

    # Now loop through each value
    for n in range(vort_spec.shape[1]):
        for m in range(vort_spec.shape[0]):
            num = vort_tend_spec[m,n] - DES[m,n] * vort_spec[m,n]
            den = np.complex(1.,0) + DES[m,n] * np.complex(nl.dt,0.)

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
    

    for n in n_vals:
        for m in m_vals:
            fourier_wave = m * nl.fourier_inc
            spherical_wave[m,n] = fourier_wave + n

    # Now for the laplacian
    eigen_laplacian = np.divide(np.multiply(spherical_wave,np.add(spherical_wave,1.)),nl.Re**2)

    # Dampening Eddy Sponge values
    DES = np.multiply(eigen_laplacian, nl.nu) 
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
    cs = plt.contourf(bmaps['global_x'],bmaps['global_y'],vort,np.linspace(-1.E-4,1.E-4,10),
                      cmap=matplotlib.cm.RdBu,extend='both',antialiasing=False)
    plt.colorbar(orientation='horizontal')
    plt.quiver(bmaps['global_x'],bmaps['global_y'],u,v)
    bmaps['global'].drawcoastlines()
    plt.title('Zeta and wind at %d hours (%s)' % (n,curtime.strftime('%Y%m%d%H')))
    plt.savefig('{}/globe_plot_{:03d}.png'.format(nl.figdir,n), bbox_inches='tight')
    plt.close()

    # North America plot with geopotential height
    phi = np.divide(psi * 7.292E-5, 9.81)
    plt.figure(figsize=(10,8))
    # Calculate wind magnitude
    windmag = np.sqrt(u**2 + v**2)
    

    cs = plt.contourf(bmaps['regional_x'],bmaps['regional_y'],windmag,np.arange(15,48,3),
                 cmap=matplotlib.cm.jet,extend='max',antialiasing=False)
    plt.colorbar(orientation='horizontal')
    hgtconts = plt.contour(bmaps['regional_x'],bmaps['regional_y'],phi,np.linspace(-500,500,26),colors='k')
    bmaps['regional'].drawcoastlines()
    bmaps['regional'].drawcountries()
    bmaps['regional'].drawstates()

    plt.title('Hgt anomalies and wind at %d hours (%s)' %(n,curtime.strftime('%Y%m%d%H')))
    print("Saving hour", n)
    plt.savefig('{}/reg_plot_{:03d}.png'.format(nl.figdir,n), bbox_inches='tight')
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
    term1 = 1./(nl.Re*np.cos(theta)) * d_dlamb(u,dlamb)
    term2 = 1./(nl.Re*np.cos(theta)) * d_dtheta(v * np.cos(theta)),dtheta
    return term1 + term2  

def vorticity_spher(u,v,theta,dtheta,dlamb):
    """ Computes normal component of vorticity in spherical
    coordinates """
    term1 = 1./(nl.Re*np.cos(theta)) * d_dlamb(v,dlamb)
    term2 = 1./(nl.Re*np.cos(theta)) * d_dtheta(u*np.cos(theta),dtheta)
    return term1 - term2

def wind_stream(psi,theta,dtheta,dlamb):
    """ Compute u and v winds from streamfunction in spherical
    coordinates """
    u = -1./nl.Re * d_dtheta(psi,dtheta)
    v = 1./(nl.Re * np.cos(theta)) * d_dlamb(psi,dlamb)
    return u,v

def Jacobian(A,B,theta,dtheta,dlamb):
    """ Returns the Jacobian of two fields in spherical coordinates """
    term1 = d_dlamb(A,dlamb) * d_dtheta(B,dtheta)
    term2 = d_dlamb(B,dlamb) * d_dtheta(A,dtheta)
    return 1./(nl.Re**2 * np.cos(theta)) * (term1 - term2)


if __name__ == '__main__':
    bmaps = create_basemaps(ics['lons'],ics['lats'])
    integrate(ics,bmaps,ntimes=ntimes)





