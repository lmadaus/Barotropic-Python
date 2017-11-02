-----------------------------------------------------------------
Barotropic-Python

A simple barotropic model written in Python using the 
spharm package for spherical harmonics.

Currently set up to use idealized initial conditions

Written by Luke Madaus (5/2012) - University of Washington

Restructured by Nick Weber(10/2017) - University of Washington

-----------------------------------------------------------------

Requires:
 PySPHARM -- Python interface to NCAR SPHEREPACK library:
	https://code.google.com/p/pyspharm/

 netCDF4 -- Python interface to netCDF4 library
 matplotlib
 numpy


Based on the Held-Suarez Barotropic model, including hyperdiffusion.
A brief description of their model may be found at:
http://data1.gfdl.noaa.gov/~arl/pubrel/m/atm_dycores/src/atmos_spectral_barotropic/barotropic.pdf

The basic premise of the code is to take 500 hPa u and v winds from the NCEP
reanalysis, extract the non-divergent component, compute vorticity, and
advect along this vorticity using the barotropic vorticity equation.  As this
model uses "real" atmospheric data (which is not barotropic), the results are
rarely stable beyond ~5 days of forecasting.

