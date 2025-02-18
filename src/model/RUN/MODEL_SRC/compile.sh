#!/bin/bash
# set folders
DIR_EXE=.
DIR_SRC=$DIR_EXE/MODEL_SRC
NETCDF=/usr/local

# compile jday
echo "gfortran -o $DIR_EXE/jday $DIR_SRC/jday.f"
gfortran -o $DIR_EXE/jday $DIR_SRC/jday.f

# compile medslik
echo "gfortran -I$NETCDF/include   -L$NETCDF/lib   $DIR_SRC/medslik_II.for -lnetcdf -lnetcdff -o $DIR_EXE/medslik_II.exe"

gfortran -I$NETCDF/include   -L$NETCDF/lib   $DIR_SRC/medslik_II.for -lnetcdf -lnetcdff -fallow-argument-mismatch -o $DIR_EXE/medslik_II.exe

# compile lat_lon
echo "gfortran -o $DIR_EXE/lat_lon.exe $DIR_SRC/lat_lon.for"
gfortran -o $DIR_EXE/lat_lon.exe $DIR_SRC/lat_lon.for