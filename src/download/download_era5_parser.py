import numpy as np
import pdb
import datetime
import os
import cdsapi
import argparse
import subprocess
import time
from glob import glob as gg

# Functions outside this script
from WITOIL_iMagine.src.utils.utils import *

def write_cds(token):
    """
    This function will write the cds token in the home directory.

    If it does not work automatically, please create it mannualy, replacing {token} with your token
    """
    file_path = os.path.expanduser("~/.cdsapirc")
    mode = "x"
    if os.path.exists(file_path):
        mode = "w"
    with open(file_path, mode=mode) as f:
        f.write("url: https://cds.climate.copernicus.eu/api\n")
        f.write(f"key: {token}")
        f.close()

def get_era5(xmin,xmax,ymin,ymax,start_date,end_date,output_path,output_name):
    server = cdsapi.Client()

    days = (end_date-start_date).days + 1

    print(ymin,ymax,xmin,xmax)

    outputs = []

    for i in range(0,days):

        date = start_date + datetime.timedelta(days=i)

        print(date)

        outputname = output_path + f'temp_{str(date.year)+str(date.month).zfill(2)+str(date.day).zfill(2)}.nc'
       
        server.retrieve(
        'reanalysis-era5-single-levels',
        {
          'product_type': ['reanalysis'],
          'data_format': 'netcdf',
          'download_format' : 'unarchived',
          'variable': [
              '10m_u_component_of_wind', '10m_v_component_of_wind',
          ],
          'year' :  [str(date.year)],
          'month':  [str(date.month).zfill(2)],
          'day'  :  [str(date.day).zfill(2)],
          'time' : [
              '00:00', '01:00', '02:00',
              '03:00', '04:00', '05:00',
              '06:00', '07:00', '08:00',
              '09:00', '10:00', '11:00',
              '12:00', '13:00', '14:00',
              '15:00', '16:00', '17:00',
              '18:00', '19:00', '20:00',
              '21:00', '22:00', '23:00',
          ],
          'area': [
              ymax, xmin, ymin,
              xmax
          ],
        },
        outputname)

def process_era5(output_path,output_name):

    met = xr.open_mfdataset('WITOIL_iMagine/data/ERA5/temp*.nc')
    met = Utils.rename_netcdf_variables_mdk3(met)

    met.to_netcdf(output_name)

    #remove the temporary files
    temp_files = gg(os.path.join(output_path, "temp*.nc"))
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == '__main__':

    # Script to download daily ERA-5 files

    parser = argparse.ArgumentParser(description='Download wind fields for a specific area and time window.')
    parser.add_argument('lat_min', type=float, help='Minimum Latitude value')
    parser.add_argument('lon_min', type=float, help='Minimum Longitude value')
    parser.add_argument('lat_max', type=float, help='Maximum Latitude value')
    parser.add_argument('lon_max', type=float, help='Maximum Longitude value')
    parser.add_argument('date_min', type=str, help='Start date in yyyy-mm-dd format')
    parser.add_argument('date_max', type=str, help='End date in yyyy-mm-dd format')
    parser.add_argument('output_path', type=str, default='./', help='Output path (default: current directory)')
    args = parser.parse_args()

    #Set your area of interest
    xmin = float(args.lon_min)
    xmax = float(args.lon_max)
    ymin = float(args.lat_min)
    ymax = float(args.lat_max)

    # Set your period of interest
    start_date=args.date_min
    end_date=args.date_max

    print('********************************************')
    print('PREPARING ERA5 WIND DATA - MEDSLIK II FORMAT')
    print('Start date :' + start_date)
    print('End date :' + end_date)
    print('********************************************')


    get_era5(args.lon_min,args.lon_max,args.lat_min,args.lat_max,start_date,end_date,args.output_path)

    # os.system('cdo -b F64 mergetime ' + out_folder + file1 + ' ' + out_folder + file2 + ' ' + out_folder + 'output.nc')

    # string1 = dDate.strftime('%Y') + '-' + dDate.strftime('%m') + '-' + dDate.strftime('%d') + 'T01:00:00'
    # string2 = fDate.strftime('%Y') + '-' + fDate.strftime('%m') + '-' + fDate.strftime('%d') + 'T00:00:00'

    # os.system('cdo seldate,' + string1 + ',' + string2 + ' ' + out_folder + '/output.nc ' + out_folder + '/' + file1[4::])
    # os.system('ncrename -O -d longitude,lon -d latitude,lat -v longitude,lon -v latitude,lat -v u10,U10M -v v10,V10M ' + out_folder + '/' + file1[4::])

    # os.system('rm ' + out_folder + '/output.nc')

    # os.system('rm ' + out_folder + '/pre_*.nc')