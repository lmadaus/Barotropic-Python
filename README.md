Barotropic-Python
=================
-----------------------------------------------------------------

A simple barotropic model written in Python using the ``spharm`` package for spherical harmonics.

Currently set up to use __linearized__ initial conditions

Written by Luke Madaus (5/2012) - University of Washington

Restructured by Nick Weber (10/2017) - University of Washington

-----------------------------------------------------------------

__**Requires**__:

  - PySPHARM -- Python interface to NCAR SPHEREPACK library:  
	https://code.google.com/p/pyspharm/
  - netCDF4 -- Python interface to netCDF4 library  
  - numpy, datetime, matplotlib  


Based on the Held-Suarez Barotropic model, including hyperdiffusion.  
A brief description of their model may be found at:  
http://data1.gfdl.noaa.gov/~arl/pubrel/m/atm_dycores/src/atmos_spectral_barotropic/barotropic.pdf

The basic premise of the code is to take upper-level u and v winds from any dataset  
(forecast or analysis), extract the non-divergent component, compute vorticity, and  
advect along this vorticity using the barotropic vorticity equation. As this model  
uses "real" atmospheric data (which is not barotropic), the results are rarely stable  
beyond ~5 days of forecasting.

-----------------------------------------------------------------

__**Contents**__:

 - **``barotropic_spectral.py``** -- contains the ``Model`` class, which handles the initialization,  
 integration, plotting, and I/O for the barotropic model
 - **``namelist.py``** -- functions as a traditional model namelist, containing the various  
 configuration parameters for the barotropic model
 - **``hyperdiffusion.py``** -- contains functions for applying hyperdiffusion to the vorticity  
 tendecy equation (helps prevent the model from blowing up)

