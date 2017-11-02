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
output_freq = 6            # Frequency of output plots in hours
use_hyper = True           # Whether or not to apply hyperdiffusion
nu = 1E-4                  # Dampening coefficient for hyperdiffusion
plot_output = True         # Whether or not to plot the output
fourier_inc = 1            # Fourier increment for computing dampening eddy sponge
r = 0.2                    # Coefficient for Robert Filter
figdir = '/home/disk/p/njweber2/research/subseasonal/barotropic/Barotropic-Python/figures'
