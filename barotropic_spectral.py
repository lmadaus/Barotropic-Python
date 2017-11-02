#!/usr/bin/env python
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import spharm
import namelist as NL # <---- IMPORTANT! Namelist containing constants and other model parameters

class Model:
    """
    Class for storing/plotting flow fields for a homogeneous (constant density), 
    non-divergent, and incompressible fluid on a sphere and integrating them forward 
    with the barotropic vorticity equation.
    """
    
    def __init__(self, ics, forcing=None):
        """
        Initializes the model.
        
        Requires:
        ics -----> Dictionary of linearized fields and space/time dimensions
                   keys: u_bar, v_bar, u_prime, v_prime, lats, lons, start_time
        forcing -> a 2D array (same shape as model fields) containing a
                   vorticity tendency [s^-2] to be imposed at each integration time step
        """
        # 1) STORE SPACE/TIME VARIABLES
        # Get the latitudes and longitudes (as lists)
        self.lats = ics['lats']
        self.lons = ics['lons']
        self.start_time = ics['start_time']  # datetime
        self.curtime = self.start_time
        
        
        # 2) GENERATE/STORE NONDIVERGENT INITIAL STATE
        # Set up the spherical harmonic transform object
        self.s = spharm.Spharmt(self.nlons(),self.nlats(),rsphere=NL.Re,gridtype='regular',legfunc='computed')
        # Truncation for the spherical transformation
        if NL.M is None:
            self.ntrunc = self.nlats()
        else:
            self.ntrunc = NL.M
        # Use the object to get the initial conditions
        # First convert to vorticity using spharm object
        vortb_spec, div_spec = self.s.getvrtdivspec(ics['u_bar'], ics['v_bar'])
        vortp_spec, div_spec = self.s.getvrtdivspec(ics['u_prime'], ics['v_prime'])
        div_spec = np.zeros(vortb_spec.shape)  # Only want NON-DIVERGENT part of wind 
        # Re-convert this to u-v winds to get the non-divergent component
        # of the wind field
        self.ub, self.vb = self.s.getuv(vortb_spec,div_spec)    # MEAN WINDS
        self.up, self.vp = self.s.getuv(vortp_spec,div_spec)    # PERTURBATION WINDS
        # Use these winds to get the streamfunction (psi) and 
        # velocity potential (chi)
        self.psib,chi = self.s.getpsichi(self.ub,self.vb)       # MEAN STREAMFUNCTION
        self.psip,chi = self.s.getpsichi(self.up,self.vp)       # PERTURBATION STREAMFUNCTION
        # Convert the spectral vorticity to grid
        self.vort_bar = self.s.spectogrd(vortb_spec)            # MEAN RELATIVE VORTICITY
        self.vortp = self.s.spectogrd(vortp_spec)               # PERTURBATION RELATIVE VORTICITY
        
        
        # 3) STORE A COUPLE MORE VARIABLES
        # Map projections for plotting
        self.bmaps = create_basemaps(self.lons, self.lats)
        # Get the vorticity tendency forcing (if any) for integration
        self.forcing = forcing
        
        
    #==== Some simple dimensional functions ==========================================    
    def nlons(self):
        return len(self.lons)
    def nlats(self):
        return len(self.lats)
    
    
    #==== Primary function: model integrator =========================================    
    def integrate(self, make_plots=True):
        """ 
        Integrates the barotropic model using spherical harmonics.
        
        Requires:
        make_plots -> if True, will plot model fields every [NL.output_freq] hours
        """
        # Create a radian grid
        lat_list_r = [x * np.pi/180. for x in self.lats]
        lon_list_r = [x * np.pi/180. for x in self.lons]

        # Meshgrid
        lons,lats = np.meshgrid(self.lons, self.lats)
        lamb, theta = np.meshgrid(lon_list_r, lat_list_r)

        # Need these for derivatives later
        dlamb = np.gradient(lamb)[1]
        dtheta = np.gradient(theta)[0]


        # Plot Initial Conditions
        if make_plots:
            self.plot_figures(0)

        
        # Now loop through the timesteps
        for n in range(NL.ntimes):
  

            # Here we actually compute vorticity tendency
            # Compute tendency with beta as only forcing
            vort_tend_rough = -2. * NL.omega/(NL.Re**2) * d_dlamb(self.psip + self.psib, dlamb) -\
                              Jacobian(self.psip+self.psib, self.vortp+self.vort_bar, theta, dtheta, dlamb)
            
            # Now add any imposed vorticity tendency forcing
            if self.forcing is not None:
                vort_tend_rough += self.forcing

            # Apply hyperdiffusion if requested for smoothing
            if NL.use_hyper:
                vort_tend = add_hyperdiffusion(self.s, self.vortp, vort_tend_rough, self.ntrunc).squeeze()
            else:
                vort_tend = vort_tend_rough

            if n == 0:
                # First step just do forward difference
                # Vorticity at next time is just vort + vort_tend * dt
                vortp_next = self.vortp + vort_tend * NL.dt
            else:
                # Otherwise do leapfrog
                vortp_next = vortp_prev + vort_tend * 2 * NL.dt 


            # Invert this new vort to get the new psi (or rather, uv winds)
            # First go back to spectral space
            vortp_spec = self.s.grdtospec(vortp_next)
            div_spec = np.zeros(np.shape(vortp_spec))  # Divergence is zero in barotropic vorticity

            # Now use the spharm methods to update the u and v grid
            self.up, self.vp = self.s.getuv(vortp_spec, div_spec)
            self.psip, chi = self.s.getpsichi(self.up, self.vp)

            # Change vort_now to vort_prev
            # and if not first step, add Robert filter to dampen out crazy modes
            if n == 0:
                vortp_prev = self.vortp
            else:
                vortp_prev = (1.-2.*NL.r)*self.vortp + NL.r*(vortp_next + vortp_prev)
                
            # Update the vorticity
            self.vortp = self.s.spectogrd(vortp_spec)
                
            # Update the current time  
            cur_fhour = (n+1) * NL.dt / 3600.
            self.curtime = self.start_time + timedelta(hours = cur_fhour)

            # Output every [output_frequ] hours
            if cur_fhour % NL.output_freq == 0 and NL.plot_output:
                # Go from psi to geopotential
                print("Plotting hour", cur_fhour)
                self.plot_figures(int(cur_fhour))
                

    #==== Plotting utilities =========================================================
    def plot_figures(self, n, winds='total', vorts='total', psis='pert', showforcing=True,
                     vortlevs=np.array([-10,-8,-6,-4,-2,2,4,6,8,10])*1e-5,
                     windlevs=np.arange(20,61,4), hgtlevs=np.linspace(-500,500,26),
                     forcelevs=np.array([-15,-12,-9,-6,-3,3,6,9,12,15])*1e-10):
        """
        Make global and regional plots of the flow.
        
        Requires:
        n ------------------> timestep number
        winds, vorts, psis -> are we plotting the 'mean', 'pert', or 'total' field?
        showforcing --------> if True, contour the vorticity tendency forcing
        *levs --------------> contour/fill levels for the respective variables
        """
        # What wind component(s) are we plotting?
        if winds=='pert':   u = self.up; v = self.vp
        elif winds=='mean': u = self.ub; v = self.vb
        else:               u = self.up+self.ub; v = self.vp+self.vb
        # What vorticity component(s) are we plotting?
        if vorts=='pert':   vort = self.vortp
        elif vorts=='mean': vort = self.vort_bar
        else:               vort = self.vortp + self.vort_bar
        # What streamfunction component(s) are we plotting?
        if psis=='pert':   psi = self.psip
        elif psis=='mean': psi = self.psib
        else:              psi = self.psip + self.psib

        # MAKE GLOBAL ZETA & WIND BARB MAP
        fig, ax = plt.subplots(figsize=(10,8))
        fig.subplots_adjust(bottom=0.2, left=0.05, right=0.95)
        
        xx, yy = self.bmaps['global_x'], self.bmaps['global_y']
        cs = ax.contourf(xx, yy, vort, vortlevs, cmap=matplotlib.cm.RdBu_r, extend='both', antialiasing=False)
        self.bmaps['global'].drawcoastlines()
        ax.quiver(xx[::2,::2], yy[::2,::2], u[::2,::2], v[::2,::2])
        # Plot the forcing
        if showforcing and self.forcing is not None:
            ax.contour(xx, yy, self.forcing, forcelevs, linewidths=2, colors='darkorchid')
        ax.set_title('relative vorticity [s$^{-1}$] and winds [m s$^{-1}$] at %03d hours' % n)
        # Colorbar
        cax = fig.add_axes([0.05, 0.12, 0.9, 0.03])
        plt.colorbar(cs, cax=cax, orientation='horizontal')
        plt.savefig('{}/globe_plot_{:03d}.png'.format(NL.figdir,n), bbox_inches='tight')
        plt.close()

        # MAKE REGIONAL HEIGHT & WIND SPEED MAP
        phi = np.divide(psi * NL.omega, NL.g)
        fig, ax = plt.subplots(figsize=(10,6))
        fig.subplots_adjust(bottom=0.2, left=0.05, right=0.95)
        xx, yy = self.bmaps['regional_x'], self.bmaps['regional_y']
        # Calculate wind speed
        wspd = np.sqrt(u**2 + v**2)
        cs = ax.contourf(xx, yy, wspd, windlevs, cmap=plt.cm.viridis, extend='max', antialiasing=False)
        self.bmaps['regional'].drawcoastlines()
        self.bmaps['regional'].drawcountries()
        self.bmaps['regional'].drawstates()
        hgtconts = ax.contour(xx, yy, phi, hgtlevs, colors='k')
        # Plot the forcing
        if showforcing and self.forcing is not None:
            ax.contour(xx, yy, self.forcing, forcelevs, linewidths=2, colors='darkorchid')
        ax.set_title('geopotential height [m] and wind speed [m s$^{-1}$] at %03d hours' % n)
        # Colorbar
        cax = fig.add_axes([0.05, 0.12, 0.9, 0.03])
        plt.colorbar(cs, cax=cax, orientation='horizontal')
        plt.savefig('{}/reg_plot_{:03d}.png'.format(NL.figdir,n), bbox_inches='tight')
        plt.close()
            
            
###########################################################################################################
##### Other Utilities #####################################################################################
###########################################################################################################


def create_basemaps(lons,lats):
    """ Setup global and regional basemaps for eventual plotting """
    print("Creating basemaps for plotting")

    long, latg = np.meshgrid(lons,lats)

    # Set up a global map
    bmap_globe = Basemap(projection='merc',llcrnrlat=-70, urcrnrlat=70,\
                         llcrnrlon=0,urcrnrlon=360,lat_ts=20,resolution='c')
    xg,yg = bmap_globe(long,latg)
   
    # Set up a regional map (currently Pacific and N. America)
    bmap_reg = Basemap(projection='merc',llcrnrlat=0,urcrnrlat=65,llcrnrlon=80, urcrnrlon=290, lat_ts=20,resolution='l')
    xr,yr = bmap_reg(long,latg)

    return {'global' : bmap_globe, 
            'global_x' : xg, 
            'global_y' : yg,
            'regional' : bmap_reg,
            'regional_x' : xr, 
            'regional_y' : yr, 
            }


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
            den = np.complex(1.,0) + DES[m,n] * np.complex(NL.dt,0.)

            new_vort_tend_spec[m,n] = num / den

    # Reshape the new vorticity tendency and convert back to grid
    new_vort_tend_spec = np.reshape(new_vort_tend_spec, (total_length,-1))

    new_vort_tend = s.spectogrd(new_vort_tend_spec)

    return new_vort_tend


def compute_dampening_eddy_sponge(fieldshape):
    """ Computes the eddy sponge by getting the eigenvalues 
    of the Laplacian for each spectral coefficient and 
    multiplying them by a dampening factor nu 
    (specified in the namelist) 
    From Held and Suarez
    """
    
    # Need some arrays
    m_vals = range(fieldshape[0])
    n_vals = range(fieldshape[1])

    spherical_wave = np.zeros(fieldshape)
    eigen_laplacian = np.zeros(fieldshape)
    

    for n in n_vals:
        for m in m_vals:
            fourier_wave = m * NL.fourier_inc
            spherical_wave[m,n] = fourier_wave + n

    # Now for the laplacian
    eigen_laplacian = np.divide(np.multiply(spherical_wave,np.add(spherical_wave,1.)),NL.Re**2)

    # Dampening Eddy Sponge values
    DES = np.multiply(eigen_laplacian, NL.nu) 
    DES_cpx = np.array(DES, dtype=np.complex)

    return DES_cpx

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
    term1 = 1./(NL.Re*np.cos(theta)) * d_dlamb(u,dlamb)
    term2 = 1./(NL.Re*np.cos(theta)) * d_dtheta(v * np.cos(theta)),dtheta
    return term1 + term2  

def vorticity_spher(u,v,theta,dtheta,dlamb):
    """ Computes normal component of vorticity in spherical
    coordinates """
    term1 = 1./(NL.Re*np.cos(theta)) * d_dlamb(v,dlamb)
    term2 = 1./(NL.Re*np.cos(theta)) * d_dtheta(u*np.cos(theta),dtheta)
    return term1 - term2

def wind_stream(psi,theta,dtheta,dlamb):
    """ Compute u and v winds from streamfunction in spherical
    coordinates """
    u = -1./NL.Re * d_dtheta(psi,dtheta)
    v = 1./(NL.Re * np.cos(theta)) * d_dlamb(psi,dlamb)
    return u,v

def Jacobian(A,B,theta,dtheta,dlamb):
    """ Returns the Jacobian of two fields in spherical coordinates """
    term1 = d_dlamb(A,dlamb) * d_dtheta(B,dtheta)
    term2 = d_dlamb(B,dlamb) * d_dtheta(A,dtheta)
    return 1./(NL.Re**2 * np.cos(theta)) * (term1 - term2)

###########################################################################################################

def test_case():
    """
    Runs an example case: extratropical zonal jets with superimposed sinusoidal NH vorticity
    perturbations and a gaussian vorticity tendency forcing.
    """
    from time import time
    
    start = time()
    # 1) LET'S CREATE SOME INITIAL CONDITIONS
    lon_list = list(np.arange(0, 360.1, 2.5))
    lat_list = list(np.arange(-87.5, 88, 2.5))[::-1]
    lamb, theta = np.meshgrid([x * np.pi/180. for x in lon_list], [x * np.pi/180. for x in lat_list])
    # Mean state: zonal extratropical jets
    ubar = 25 * np.cos(theta) - 30 * np.cos(theta)**3 + 300 * np.sin(theta)**2 * np.cos(theta)**6
    vbar = np.zeros(np.shape(ubar))
    # Initial perturbation: sinusoidal vorticity perturbations
    A = 1.5 * 8e-5 # vorticity perturbation amplitude
    m = 4          # zonal wavenumber
    theta0 = np.deg2rad(45)  # center lat = 45 N
    thetaW = np.deg2rad(15)
    vort_pert = 0.5*A*np.cos(theta)*np.exp(-((theta-theta0)/thetaW)**2)*np.cos(m*lamb)
    # Get U' and V' from this vorticity perturbation
    s = spharm.Spharmt(len(lon_list), len(lat_list), gridtype='regular', legfunc='computed', rsphere=NL.Re)
    uprime, vprime = s.getuv(s.grdtospec(vort_pert), np.zeros(np.shape(s.grdtospec(vort_pert))))
    # Full initial conditions dictionary:
    ics = {'u_bar'  : ubar,
           'v_bar'  : vbar,
           'u_prime': uprime,
           'v_prime': vprime,
           'lons'   : lon_list,
           'lats'   : lat_list,
           'start_time' : datetime(2017,1,1,0)}

    # 2) LET'S ALSO FEED IN A GAUSSIAN NH RWS FORCING
    amplitude = 10e-10              # s^-2
    forcing = np.zeros(np.shape(ubar))
    x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 0.5, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )   # GAUSSIAN CURVE
    lat_i = np.where(np.array(lat_list)==35.)[0][0]
    lon_i = np.where(np.array(lon_list)==160.)[0][0]
    forcing[lat_i:lat_i+10, lon_i:lon_i+10] = g*amplitude

    # 3) INTEGRATE!
    model = Model(ics, forcing=forcing)
    model.integrate()
    print('TOTAL INTEGRATION TIME: {:.02f} minutes'.format((time()-start)/60.))

if __name__ == '__main__':
    test_case()