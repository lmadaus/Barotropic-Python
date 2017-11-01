#!/usr/bin/env python
"""
This module contains the variables used to run barotropic_spectral.py
"""
    
# Set parameters
M = 42
Re = 6378100.              # Radius of earth (m)
omega = 7.292E-5           # Earth's angular momentum (s^-1)
dt = 900.                  # Timestep (seconds)
ntimes = 480               # Number of time steps to integrate
output_freq = 6            # Frequency of output plots in hours
use_hyper = False           # Whether or not to apply hyperdiffusion
damping_order = 1          # Order of dampening
nu = 1E-4                  # Dampening coefficient for hyperdiffusion
restart = None             # Filename of restart file to load
plot_output = True         # Whether or not to plot the output
fourier_inc = 1            #
r = 0.0                    # (was 0.2)
figdir = '/home/disk/p/njweber2/research/subseasonal/barotropic/Barotropic-Python/figures'
