import sys, os, shutil
import numpy as np
import pandas as pd
import xarray as xr
import logging
import geopandas as gpd
import subprocess
from glob import glob
from numpy.typing import NDArray
from datetime import datetime, timedelta, date
import psutil

from WITOIL_iMagine.src.utils.utils import Utils
from WITOIL_iMagine.src.utils.config import Config

import logging

logger = logging.getLogger(__name__)

def print_memory_debug(tag, obj=None):
    process = psutil.Process()
    mem_used = process.memory_info().rss / 1024**2  # MB
    if obj is not None:
        try:
            size = obj.nbytes / 1024**2  # MB
        except AttributeError:
            size = sys.getsizeof(obj) / 1024**2
        print(f"DEBUG [{tag}] - Used RAM: {mem_used:.2f} MB, Object size: {size:.2f} MB")
    else:
        print(f"DEBUG [{tag}] - Used RAM: {mem_used:.2f} MB")

class PreProcessing:
    """
    This class embeds functions for pre-processing, with a user-interface purpose.
    """

    def __init__(self, config: dict, exp_folder: str, domain=list[float]) -> None:
        """
        Class constructor.
        """
        self.exp_folder = exp_folder
        os.makedirs(exp_folder, exist_ok=True)
        # - start_date and end_date: start and end data of the pre-processing
        self.config = config
        day = config["simulation"]["start_datetime"].day
        year = config["simulation"]["start_datetime"].year
        month = config["simulation"]["start_datetime"].month
        self.sim_length = config["simulation"]["sim_length"] 
        start_date = date(year, month, day)
        sim_length_days = int(self.sim_length / 24.0) + 1
        end_date = start_date + timedelta(days=sim_length_days)
        self.start_date = start_date
        self.end_date = end_date
        self.date_list = pd.date_range(start_date, end_date, freq='D')
        self.simname = config["simulation"]["name"]
        self.grid = None
        self.domain = domain

    def create_directories(self):
        os.makedirs(f'{self.exp_folder}/oce_files',exist_ok=True)
        os.makedirs(f'{self.exp_folder}/met_files',exist_ok=True)
        os.makedirs(f'{self.exp_folder}/bnc_files',exist_ok=True)
        os.makedirs(f'{self.exp_folder}/xp_files',exist_ok=True)
        os.makedirs(f'{self.exp_folder}/out_files',exist_ok=True)
        os.makedirs(f'{self.exp_folder}/detections',exist_ok=True)

    def process_currents(self, oce_path: str = None):

        logger.info("Pre processing currents")
        lon_min, lon_max, lat_min, lat_max = self.domain
        # opening all files in the directory and concatenating them automatically through open_mfdataset
        oce_path = f"{self.exp_folder}/oce_files/*.nc"

        if glob(oce_path) == []:
            oce_path = f"{self.exp_folder}/oce_files/*.nc"
        concat = xr.open_mfdataset(oce_path, combine="nested", engine="netcdf4")
        print_memory_debug("process_currents - concat", concat)
        concat = concat.drop_duplicates(dim="time", keep="last")
        if self.config["input_files"]["set_domain"]:
            concat = concat.sel(
                lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max)
            )
        # Interpolating the values in time, transforming it from daily to hourly values
        concat = concat.resample(time="1h").interpolate("linear")
        print_memory_debug("process_currents - after resample", concat)
        Utils.write_mrc(concat, exp_folder=self.exp_folder)
        self._copy_nc_files(oce_path, f"{self.exp_folder}/oce_files/")
        print("DEBUG: process_currents() completed")
    
    def process_winds(self, met_path: str = None):

        logger.info("Pre processing winds")
        lon_min, lon_max, lat_min, lat_max = self.domain
        # opening all files in the directory and concatenating them automatically through open_mfdataset
        met_path = f"{self.exp_folder}/met_files/*.nc"

        if glob(met_path) == []:
            met_path = f"{self.exp_folder}/met_files/*.nc"
        concat = xr.open_mfdataset(met_path, combine="nested", engine="netcdf4")
        print_memory_debug("process_winds - concat", concat)
        concat = concat.drop_duplicates(dim="time", keep="first")
        # Interpolating the values in time, transforming it from daily to hourly values
        concat = concat.resample(time="1h").interpolate("linear")
        print_memory_debug("process_winds - after resample", concat)
        # Handle longitude and latitude
        concat["lon"] = xr.where(
            concat["lon"] > 180, concat["lon"] - 360, concat["lon"]
        )
        concat = concat.sortby("lat")
        concat = concat.sortby("lon")
        # Iterating at each hour to generate the .eri files
        for date in self.date_list:
            # Call write eri function located in medslik.utils file
            Utils.write_eri(concat, date, exp_folder=self.exp_folder)
        self._copy_nc_files(met_path, f"{self.exp_folder}/met_files/")
        print("DEBUG: process_winds() completed")

    def _copy_nc_files(self, src_files: str, dst_dir: str) -> None:
        # Use glob to find all .nc files in the source directory
        nc_files = glob(src_files)
        # Loop through each file
        for src_file in nc_files:
            # Get the destination file path
            dst_file = os.path.join(dst_dir, os.path.basename(src_file))
            # Check if the destination file already exists
            if not os.path.exists(dst_file):
                # Copy the file if it doesn't exist
                shutil.copy(src_file, dst_file)

    def process_bathymetry(self,gebco):

        grid = self.grid
        
        if gebco is None or not gebco:
            bnc_path = f"{self.exp_folder}/bnc_files/*.nc"
        else:
            bnc_path = gebco

        # Use open_mfdataset with the netcdf4 engine (instead of open_dataset)
        gebco = xr.open_mfdataset(bnc_path, engine="netcdf4")
        
        # --- NEW: Rename variables if they are 'latitude'/'longitude'
        # Rename variables if they exist in the dataset
        if (("latitude" in gebco.coords or "latitude" in gebco.data_vars) and 
            ("longitude" in gebco.coords or "longitude" in gebco.data_vars)):
            gebco = gebco.rename({"latitude": "lat", "longitude": "lon"})

        # interpolation on medslik grid
        med = gebco.interp(lon=grid.lon.values.tolist(),lat=grid.lat.values.tolist())
        print_memory_debug("process_bathymetry - interpolated gebco", med)
        print("DEBUG: bathymetry memory usage (MB):", med.nbytes / 1e6)
        print("DEBUG: GEBCO dimensions:", gebco.sizes)
        print("DEBUG: interpolated med dimensions:", med.sizes)
        #converting from begative depths to positive
        med['elevation'] = med.elevation *-1
        #filling nan to -9999 as expected by medslik
        med = med.fillna(9999)	

        # Convert bathymetry to MDK-II
        mdk_z=[]

        llat,llon = len(med.lat),len(med.lon)
        
        for i in reversed(range(0,llat)):
            for j in range(0,llon):
                rec = med.isel(lat=i,lon=j)
                mdk_z.append(rec.elevation.values.max())

        mdk_z = np.array(mdk_z)
        land_mask = np.where(mdk_z <= 0)
        mdk_z[land_mask]=9999
        print_memory_debug("process_bathymetry - mdk_z array", mdk_z)


        BathFile=open(f'{self.exp_folder}/bnc_files/dtm.bath', "w")
        BathFile.write("MEDSLIK-II compatible bathymetry file. Degraded resolution based on GEBCO 30''\n")
        BathFile.write("%-7.4f %-7.4f %-7.4f %-7.4f \n" % (np.min(grid.lon.values),np.max(grid.lon.values),np.min(grid.lat.values),np.max(grid.lat.values)))
        BathFile.write("%d %d \n" % (llon,llat))
        np.savetxt(BathFile,mdk_z,fmt="%04.0f")
        BathFile.close()

    def process_coastline(self,gshhs):
        
        grid = self.grid

        if 'nav_lat' in grid and 'nav_lon' in grid:
            grid = grid.rename({'nav_lat': 'lat', 'nav_lon': 'lon'})

        # try:
        #     grid = grid.rename({'nav_lat':'lat','nav_lon':'lon'})
        # except:
        #     pass

        # 1 degree buffer to collect more coastline
        buffer = 1
        #defining minimum and maximum of coordinates box search 
        xmin = grid.lon.min() - buffer
        xmax = grid.lon.max() + buffer
        ymin = grid.lat.min() - buffer
        ymax = grid.lat.max() + buffer

        shp = gpd.read_file(gshhs)
        print_memory_debug("process_coastline - gshhs read", shp)

        # Cropping to a smaller area
        shp = shp.cx[xmin:xmax, ymin:ymax]

        #shp with linestring instead of polygons
        shp['geometry'] = shp.geometry.boundary

        # Cropping the selected linestrings to the same bounding box
        shp = shp.clip_by_rect(xmin.values.max(),ymin.values.max(), xmax.values.max(),ymax.values.max())

        # Removing empty geometries
        shp = shp[~shp.is_empty]

        #Transforming it back again to geodataframe
        shp = gpd.GeoDataFrame(geometry = shp)

        #removing any multiline strings left on the shapefile
        shp = shp.explode(index_parts=True)

        # writes the first line of the .map file. It should contain the # of "isles"
        CoastFile=open(f'{self.exp_folder}/bnc_files/dtm.map','w')
        CoastFile.write("%-4.0f \n" % (len(shp)))
        iTargetSites=[]
        iTargetSite=np.array([0.,0.,0.,0.])

        for i,polygon in shp.iterrows():

            # extracts the island
            pol = polygon.geometry

            # Extract the exterior coordinates of the polygon
            # exterior_coords = list(pol.exterior.coords)
            exterior_coords = list(pol.coords)
            
            #prints the length of the island
            CoastFile.write("%-4.0f %1.0f \n" % (len(exterior_coords),0))
            #prints data related to the island
            for segment in exterior_coords:
                CoastFile.write("%-8.5f %-6.5f \n" % (segment[0], segment[1]))

    def process_medslik_memmory_array(self):
        #ocean
        my_o = xr.open_mfdataset(f'{self.exp_folder}/oce_files/*nc')

        #wind
        # my_w = xr.open_dataset(glob(f'{self.exp_folder}/{self.simname}/met_files/*nc')[0]).isel(time=0)['U10M'].values.shape
        
        #obtaining the maximum array
        nmax = np.max([np.max(my_o.isel(time=0).uo.values.shape)])

        # modify medslik_ii
        print('...medslik_ii.for...')

        med_for = f'{self.exp_folder}/xp_files/medslik_II.for'

        # Refactored: Copying the template file
        template_path = 'WITOIL_iMagine/src/templates/medslik_II_template.for'
        shutil.copy(template_path, med_for)

        # Replacing NMAX in medslik Fortran with a Python function
        Utils.search_and_replace(med_for, 'NMAX', str(nmax))
        print("DEBUG: medslik_II.for ready")

    def configuration_parameters(self):

        source_file = 'WITOIL_iMagine/src/templates/config2.txt'
        destination_file = f'{self.exp_folder}/xp_files/config2.txt'

        # Refactored: Copy the file
        shutil.copy(source_file, destination_file)

    def common_grid(self):

        '''
        function to crop files based on this presented common grid.

        Default uses current grid for the area of interest
        '''

        lon_min, lon_max, lat_min, lat_max = self.domain
        grid = glob(f"{self.exp_folder}/oce_files/*.nc")[0]

        grid = xr.open_dataset(grid)

        self.grid = grid

    def write_config_files(self,
        spill_dictionary=None,
        use_slk_contour=False,
        separate_slicks=False,
        s_num=None,
    ):
        print("DEBUG: write_config_files() called")
        print_memory_debug("before write_config_files")
        print(f"       simname={spill_dictionary['simname']}, separate_slicks={separate_slicks}, s_num={s_num}")
        # obtaining the variables
        simname = spill_dictionary["simname"]
        dt_sim = spill_dictionary["dt_sim"]
        sim_length = spill_dictionary["sim_length"]
        longitude = spill_dictionary["longitude"]
        latitude = spill_dictionary["latitude"]
        spill_duration = spill_dictionary["spill_duration"]
        spill_rate = spill_dictionary["spill_rate"]
        oil_api = spill_dictionary["oil_api"]
        if separate_slicks:
            number_slick = spill_dictionary["number_slick"]
        else:
            number_slick = 1

        # # modify config_1.txt
        print("...config1.txt...")
        # Iterating through slicks or doing for single simulation
        if separate_slicks == False:
            config_file = f"WITOIL_iMagine/cases/{simname}/xp_files/config1.txt"
        else:
            config_file = f"WITOIL_iMagine/cases/{simname}/xp_files/slick{s_num+1}/config1.txt"
        # Refactored: Copy the template file
        print(f"DEBUG: Copying config1 template to {config_file}")
        source_file = "WITOIL_iMagine/src/templates/config1_template_0.txt"
        shutil.copy(source_file, config_file)
        # adding spill Name - Add slick number if separate slicks
        if separate_slicks == False:
            print("DEBUG: Replacing run name")
            Utils.search_and_replace(config_file, "RUNNAME", simname)
        else:
            Utils.search_and_replace(config_file, "RUNNAME", simname + f"_slick{s_num+1}")
        # adding spill date and hour information
        print("DEBUG: Replacing datetime info")
        Utils.search_and_replace(config_file, "DD", f"{dt_sim.day:02d}")
        Utils.search_and_replace(config_file, "MM", f"{dt_sim.month:02d}")
        Utils.search_and_replace(config_file, "YY", f"{dt_sim.year-2000:02d}")
        Utils.search_and_replace(config_file, "c_Hour", f"{dt_sim.hour:02d}")
        Utils.search_and_replace(config_file, "c_minute", f"{dt_sim.minute:02d}")
        # adding simulation length
        print("DEBUG: Replacing simulation length")
        Utils.search_and_replace(config_file, "SIMLENGTH", f"{sim_length:04d}")
        #  adding spill coordinates - dd for degrees and mm for minutes
        # Latitude
        dd = int(latitude)
        mm = (float(latitude) - dd) * 60
        print("DEBUG: Replacing coordinates")
        Utils.search_and_replace(config_file, "LATd", f"{dd:02d}")
        Utils.search_and_replace(config_file, "LATm", f"{mm:.3f}")
        # Longitude
        dd = int(longitude)
        mm = (float(longitude) - dd) * 60
        Utils.search_and_replace(config_file, "LONd", f"{dd:02d}")
        Utils.search_and_replace(config_file, "LONm", f"{mm:.3f}")
        # spill duration
        print("DEBUG: Replacing spill duration and rate")
        Utils.search_and_replace(config_file, "SDUR", f"{spill_duration:04d}")
        # spill volume
        Utils.search_and_replace(config_file, "SRATE", f"{spill_rate:08.2f}")
        # oil characteristics
        print("DEBUG: Replacing oil type")
        Utils.search_and_replace(config_file, "APIOT", f"{oil_api}")
        # number of slicks
        Utils.search_and_replace(config_file, "N_SLICK", f"{number_slick}")
        # slick countour
        if use_slk_contour == True:
            slik = "YES"
            if separate_slicks == False:
                with open(f"{self.exp_folder}/{simname}/xp_files/slick_countour.txt", "r") as file1:
                    content = file1.read()
                with open(config_file, "a") as file2:
                    # Append the contents of the first file to config file
                    file2.write(content)
            else:
                with open(
                    f"{self.exp_folder}/{simname}/xp_files/slick{s_num+1}/slick_countour.txt", "r"
                ) as file1:
                    content = file1.read()
                with open(config_file, "a") as file2:
                    # Append the contents of the first file to config file
                    file2.write(content)
        else:
            slik = "NO"
        # Writing that will use slick countor
        Utils.search_and_replace(config_file, "SSLICK", f"{slik}")
        print("DEBUG: write_config_files() finished")

if __name__ == "__main__":
    
    pass
