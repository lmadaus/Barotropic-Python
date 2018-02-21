#!/usr/bin/env python
"""
This module contains the variables used to run barotropic_spectral.py
"""
import os
    
# Integration options
dt = 900.                  # Timestep (seconds)
ntimes = 960               # Number of time steps to integrate
plot_freq = 6              # Frequency of output plots in hours (if 0, no plots are made)
M = None                   # Truncation (if None, defaults to # latitudes)
r = 0.2                    # Coefficient for Robert Filter


# I/O parameters
figdir = os.path.join(os.getcwd(), 'figures')  # Figure directory


# Diffusion parameters
diff_opt = 1               # Hyperdiffusion option (0 = none, 1 = del^4, 2 = DES)
k = 2.338e16               # Diffusion coefficient for del^4 hyperdiffusion (diff_opt=1)
nu = 1E-4                  # Dampening coefficient for DES hyperdiffusion (diff_opt=2)
fourier_inc = 1            # Fourier increment for computing dampening eddy sponge (diff_opt=2)


# Constants
Re = 6378100.              # Radius of earth (m)
omega = 7.292E-5           # Earth's angular momentum (s^-1)
g = 9.81                   # Gravitational acceleration (m s^-2)