#!/usr/bin/env python
"""
This module contains the variables used to run barotropic_spectral.py
"""
    
# Set parameters
M = None                   # Truncation (if None, defaults to # latitudes)
Re = 6378100.              # Radius of earth (m)
omega = 7.292E-5           # Earth's angular momentum (s^-1)
g = 9.81                   # Gravitational acceleration (m s^-2)
dt = 900.                  # Timestep (seconds)
ntimes = 960               # Number of time steps to integrate
plot_freq = 6              # Frequency of output plots in hours (if 0, no plots are made)
diff_opt = 1               # Hyperdiffusion option (0 = none, 1 = del^4, 2 = DES)
nu = 1E-4                  # Dampening coefficient for hyperdiffusion
fourier_inc = 1            # Fourier increment for computing dampening eddy sponge
r = 0.2                    # Coefficient for Robert Filter
figdir = '/home/disk/p/njweber2/research/subseasonal/barotropic/Barotropic-Python/new_diffusion_test'
