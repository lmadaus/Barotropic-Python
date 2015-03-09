#!/usr/bin/env python

import os
os.system('rm *.2015.nc')
#os.system('wget ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/pressure/hgt.2012.nc')
os.system('wget ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/pressure/uwnd.2015.nc')
os.system('wget ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/pressure/vwnd.2015.nc')


