from netCDF4 import Dataset
import sys, os
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import datetime

# geo libs
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS

# numerical and plotting libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# system libs
import time
import subprocess
import threading
from glob import glob
import multiprocessing


class Utils:
    """
    This class embeds functions with utility purposes,
    which can be used across MEDSLIK-II software.
    """

    @staticmethod
    def compute_domain(
        sim_length: float,
        spill_lat_deg: float,
        spill_lat_min: float,
        spill_lon_deg: float,
        spill_lon_min: float,
        scale_factor: float = 0.0,
    ) -> list[float]:
        """
        Compute domain to be preprocessed based on simulation length and spill position.
        """
        max_sim_length_allowed = 120.0  # max 5 days
        if sim_length > max_sim_length_allowed:
            sim_length = max_sim_length_allowed
        # reference velocity = 1.5 nautical miles/hr
        ref_vel = 1.5
        # Distance in nautical miles
        dist_nm = ref_vel * sim_length
        # Conversion to latitude minutes
        dist_lat = dist_nm / 60.0
        # Conversion to longitude minutes
        dist_lon = dist_lat / np.cos(np.deg2rad(spill_lat_deg))
        # Find grid corners
        xE = (spill_lon_deg + 1 * scale_factor) + (spill_lon_min / 60 + dist_lon)
        xW = (spill_lon_deg - 1 * scale_factor) + (spill_lon_min / 60 - dist_lon)
        yN = (spill_lat_deg + 1 * scale_factor) + (spill_lat_min / 60 + dist_lat)
        yS = (spill_lat_deg - 1 * scale_factor) + (spill_lat_min / 60 - dist_lat)
        grid_corners = [xW, xE, yS, yN]
        return grid_corners

    @staticmethod
    def read_txt_config1(config: str, *keys: list[str]) -> tuple[float]:
        """
        Read config file
        """
        df = pd.read_csv(
            config, delimiter="=", index_col=None, header=None, usecols=None
        )
        values = list()
        for key in keys:
            for val in df.values:
                if val[0] == key:
                    values.append(float(val[1][:7]))
                    break
        return tuple(values)

    @staticmethod
    def set_product(product_name: str) -> dict[NDArray]:
        """
        Set input and output reference hours based on product name.
        """
        if product_name == "mercator_daily":
            output_ref_hours = {"start": 24, "end": 47}
            input_ref_hours = [12, 36, 60]
        elif product_name == "era5":
            output_ref_hours = {"start": 0, "end": 23}
            input_ref_hours = [0, 6, 12, 18, 24]
        else:
            raise ValueError("Invalid product name")
            # Compute
        output_hours = np.arange(output_ref_hours["start"], output_ref_hours["end"] + 1)
        input_hours = np.array(input_ref_hours)
        dict_hours = {"irh": input_hours, "orh": output_hours}
        return dict_hours

    @staticmethod
    def check_land(lon, lat, gshhg_path: str):
        """
        This script receives a lon and lat value and  checks if the position is within land or sea
        It uses a shapefile of world boundaries currently based in geopandas database

        In case a position is on land:
            Script returns 0, since it will not be possible to simulate oil spill on land

        Otherwise returns 1
        """

        world = gpd.read_file(gshhg_path)
        point = Point(lon, lat)
        is_within_land = world.geometry.contains(point).any()
        if is_within_land:
            sea = 0
        else:
            sea = 1
        return sea

    @staticmethod
    def validate_date(date):
        """
        Convert a date in string format to datetime, checking if the date provided is valid

        It also checks if date is in the future, blocking the user to procced
        """

        try:
            dt = pd.to_datetime(date)
            if dt > datetime.datetime.today():
                dt = "Date provided is in the future. No data will be available"
            if (datetime.datetime.today() - dt) < datetime.timedelta(days=5):
                raise Warning('ERA5 data might not be available in case you need downloads')
        except:
            raise ValueError('Date not provided in correct format')
        return dt

    @staticmethod
    def search_and_replace(file_path, search_word, replace_word):
        with open(file_path, "r") as file:
            file_contents = file.read()
            updated_contents = file_contents.replace(search_word, replace_word)
        with open(file_path, "w") as file:
            file.write(updated_contents)

    @staticmethod
    def write_cds(key):
        with open("~.cdsapirc_test") as f:
            f.write("url: https://cds.climate.copernicus.eu/api/v2\n")
            f.write(f"key: {key}\n")
            f.write("verify: 0")
            f.close()

    @staticmethod
    def write_mrc(ds, exp_folder=None):
        """
        This function write the currents and sea surface temperature files for Medslik-II.

        Data has to be passed in hourly format and in netcdf format
        """

        # iterating at each hour to generate the .mrc files
        for i in range(0, len(ds.time)):
            # select the i time instant
            rec = ds.isel(time=i)
            # get the datetime values from that instant
            try:
                dt = pd.to_datetime(rec.time.values)
            except:
                try:
                    dt = datetime.datetime.strptime(
                        str(rec.time.values), "%Y-%m-%d %H:%M:%S"
                    )
                except:
                    raise ValueError(
                        "Datetime from the dataset in on an unknown format"
                    )
            # transforms it from xarray datset to pandas dataframe to facilitate the processes of adjusting values
            df = rec.to_dataframe().reset_index()
            df = df.fillna(0)
            df = df.drop(["time"], axis=1)
            # pivoting it in order to create the same pattern in .mrc files
            df = df.pivot(
                index=["lat", "lon"], columns="depth", values=["thetao", "uo", "vo"]
            ).reset_index()
            # join colum information to
            df.columns = [pair[0] + str(pair[1]).split(".")[0] for pair in df.columns]
            # dropping temperature columns
            df = df.drop(["thetao10", "thetao30", "thetao120"], axis=1)
            # sort first by latitude and then by longitude
            df = df.sort_values(["lon", "lat"])
            df.columns = [
                "lat",
                "lon",
                "SST",
                "u_srf",
                "u_10m",
                "u_30m",
                "u_120m",
                "v_srf",
                "v_10m",
                "v_30m",
                "v_120m",
            ]
            df = df[
                [
                    "lat",
                    "lon",
                    "SST",
                    "u_srf",
                    "v_srf",
                    "u_10m",
                    "v_10m",
                    "u_30m",
                    "v_30m",
                    "u_120m",
                    "v_120m",
                ]
            ]
            # making sure that .mrc files in 0 hour are written as 24
            # this code also makes sure that the file is written correctly even in the transition of months
            if dt.hour == 0:
                hour = 24
                day = (dt - datetime.timedelta(hours=1)).day
                month = (dt - datetime.timedelta(hours=1)).month
            else:
                hour = dt.hour
                day = dt.day
                month = dt.month
            # writing the current files
            with open(
                f"{exp_folder}/oce_files/merc{dt.year-2000:02d}{dt.month:02d}{day:02d}{hour:02d}.mrc",
                "w",
            ) as f:
                f.write(
                    f"Ocean forecast data for {day:02d}/{dt.month:02d}/{dt.year} {hour:02d}:00\n"
                )
                f.write("Subregion of the Global Ocean:\n")
                f.write(
                    f"{df.lon.min():02.2f}  {df.lon.max():02.2f}  {df.lat.min():02.2f} {df.lat.max():02.2f}   {len(rec.lon)}   {len(rec.lat)}   Geog. limits\n"
                )
                f.write(f"{len(df)}   0.0\n")
                f.write(
                    "lat        lon        SST        u_srf      v_srf      u_10m      v_10m       u_30m      v_30m      u_120m     v_120m\n"
                )
                for index, row in df.iterrows():
                    f.write(
                        f"{row['lat']:<10.4f}    {row['lon']:<10.4f}    {row['SST']:<10.4f}     {row['u_srf']:<10.4f}    {row['v_srf']:<10.4f}     {row['u_10m']:<10.4f}    {row['v_10m']:<10.4f}     {row['u_30m']:<10.4f}    {row['v_30m']:<10.4f}     {row['u_120m']:<10.4f}    {row['v_120m']:<10.4f}\n"
                    )
        print("Sea State variables written")
        print("DEBUG: Finished writing all .mrc files")


    @staticmethod
    def write_eri(ds, date, exp_folder=None):
        """
        This function write the wind velocity as 10 m files for Medslik-II.

        Data has to be passed in hourly format and in netcdf format
        """

        # iterating at each hour to generate the .eri files
        try:
            date1 = f"{date.year}-{date.month:02d}-{date.day:02d} 00:00"
            date2 = f"{date.year}-{date.month:02d}-{date.day:02d} 23:00"
            met = ds.sel(time=slice(date1, date2))
            # getting date from the netcdf
            try:
                dt = pd.to_datetime(met.time[0].values)
            except:
                dt = datetime.datetime.strptime(
                    str(met.time[0].values), "%Y-%m-%d %H:%M:%S"
                )
            df = met.to_dataframe().reset_index()
            df = df.fillna(9999)
            df = df.pivot(
                index=["lat", "lon"], columns="time", values=["U10M", "V10M"]
            ).reset_index()
            df.columns = [pair[0] + str(pair[1]) for pair in df.columns]

            df = df.rename({"lonNaT": "lon", "latNaT": "lat"}, axis=1)
            df = df.sort_values(["lon", "lat"], ascending=[True, False])

            # writing the wind files
            with open(
                f"{exp_folder}/met_files/erai{dt.year-2000:02d}{dt.month:02d}{dt.day:02d}.eri",
                "w",
            ) as file:
                file.write(
                    f" 10m winds forecast data for {dt.day:02d}/{dt.month:02d}/{dt.year}\n"
                )
                file.write(" Subregion of the Global Ocean with limits:\n")
                file.write(
                    f"  {df.lon.min():02.5f}  {df.lon.max():02.5f}  {df.lat.min():02.5f}  {df.lat.max():02.5f}   {len(met.lon)}   {len(met.lat)}   Geog. limits\n"
                )
                file.write(f"   {len(df)}   0.0\n")
                file.write(
                    "lat        lon        u00        v00        u01       v01      u02      v02     u03     v03     u04     v04     u05     v05     u06     v06\n"
                )
                for index, row in df.iterrows():
                    file.write(f"   {row['lat']: .4f}   {row['lon']: .4f}")

                    for h in range(1, 25):
                        try:
                            file.write(
                                f"    {row.iloc[1+h]: .4f}    {row.iloc[25+h]: .4f}"
                            )
                        except:
                            file.write(f"    {0: .4f}    {0: .4f}")

                        if h == 24:
                            file.write("\n")
        except:
            print(f"{date} has no files in met directory")
            pass

    @staticmethod
    def process_mrc(i, concat, exp_folder=None):
        rec = concat.isel(time=i)
        # get the datetime values from that instant
        try:
            dt = pd.to_datetime(rec.time.values)
        except:
            try:
                dt = datetime.datetime.strptime(
                    str(rec.time.values), "%Y-%m-%d %H:%M:%S"
                )
            except:
                raise ValueError("Datetime from the dataset in on an unknown format")
        # transforms it from xarray datset to pandas dataframe to facilitate the processes of adjusting values
        df = rec.to_dataframe().reset_index()
        df = df.fillna(0)
        df = df.drop(["time"], axis=1)
        # pivoting it in order to create the same pattern in .mrc files
        df = df.pivot(
            index=["lat", "lon"], columns="depth", values=["thetao", "uo", "vo"]
        ).reset_index()
        # join colum information to
        df.columns = [pair[0] + str(pair[1]).split(".")[0] for pair in df.columns]
        # dropping temperature columns
        df = df.drop(["thetao10", "thetao30", "thetao120"], axis=1)
        # sort first by latitude and then by longitude
        df = df.sort_values(["lon", "lat"])
        df.columns = [
            "lat",
            "lon",
            "SST",
            "u_srf",
            "u_10m",
            "u_30m",
            "u_120m",
            "v_srf",
            "v_10m",
            "v_30m",
            "v_120m",
        ]
        df = df[
            [
                "lat",
                "lon",
                "SST",
                "u_srf",
                "v_srf",
                "u_10m",
                "v_10m",
                "u_30m",
                "v_30m",
                "u_120m",
                "v_120m",
            ]
        ]
        # making sure that .mrc files in 0 hour are written as 24
        # this code also makes sure that the file is written correctly even in the transition of months
        if dt.hour == 0:
            hour = 24
            day = (dt - datetime.timedelta(hours=1)).day
            month = (dt - datetime.timedelta(hours=1)).month
        else:
            hour = dt.hour
            day = dt.day
            month = dt.month

        # writing the current files
        with open(
            f"{exp_folder}/oce_files/merc{dt.year-2000:02d}{month:02d}{day:02d}{hour:02d}.mrc",
            "w",
        ) as f:
            f.write(
                f"Ocean forecast data for {day:02d}/{month:02d}/{dt.year} {hour:02d}:00\n"
            )
            f.write("Subregion of the Global Ocean:\n")
            f.write(
                f"{df.lon.min():02.2f}  {df.lon.max():02.2f}  {df.lat.min():02.2f} {df.lat.max():02.2f}   {len(rec.lon)}   {len(rec.lat)}   Geog. limits\n"
            )
            f.write(f"{len(df)}   0.0\n")
            f.write(
                "lat        lon        SST        u_srf      v_srf      u_10m      v_10m       u_30m      v_30m      u_120m     v_120m\n"
            )
            for index, row in df.iterrows():
                f.write(
                    f"{row['lat']:<10.4f}    {row['lon']:<10.4f}    {row['SST']:<10.4f}     {row['u_srf']:<10.4f}    {row['v_srf']:<10.4f}     {row['u_10m']:<10.4f}    {row['v_10m']:<10.4f}     {row['u_30m']:<10.4f}    {row['v_30m']:<10.4f}     {row['u_120m']:<10.4f}    {row['v_120m']:<10.4f}\n"
                )

    @staticmethod
    def parallel_processing_mrc(concat, exp_folder=None):
        """
        Define a function to process multiple time steps in parallel
        """

        num_time_steps = len(concat.time)
        pool = multiprocessing.Pool()  # Create a pool of workers
        results = [
            pool.apply_async(Utils.process_mrc, args=(i, concat, exp_folder))
            for i in range(num_time_steps)
        ]
        pool.close()  # Close the pool, no more tasks can be submitted
        pool.join()  # Wait for all worker processes to finish

    @staticmethod
    def run_process_gebco(gebco, grid, output_dir):
        script_name = "scripts/pre_processing/preproc_gebco_mdk2.py"
        # Run the external Python script as a subprocess
        subprocess.run([f"{sys.executable}", script_name, gebco, grid, output_dir]) # nosec

    @staticmethod
    def run_process_gshhs(gshhs, grid, output_dir):
        script_name = "scripts/pre_processing/preproc_gshhs_mdk2.py"
        # Run the external Python script as a subprocess
        subprocess.run([f"{sys.executable}", script_name, gshhs, grid, output_dir]) # nosec

    def rename_netcdf_variables_mdk3(ds):
        """
        Dictionary containing names with the possibility to rename
        """
        variables_to_rename = {
            "depthu": "depth",
            "depthv": "depth",
            "deptht": "depth",
            "nav_lat": "lat",
            "nav_lon": "lon",
            "y": "lat",
            "x": "lon",
            "latitude": "lat",
            "longitude": "lon",
            "votemper": "thetao",
            "mesh2d_tem1": "thetao",
            "vozocrtx": "uo",
            "mesh2d_ucx": "uo",
            "vomecrty": "vo",
            "mesh2d_ucy": "vo",
            "u10": "U10M",
            "mesh2d_windx": "U10M",
            "v10": "V10M",
            "mesh2d_windy": "V10M",
            "valid_time":"time","time_counter": "time",
        }
        # Rename variables only if they exist in the dataset
        for old_name, new_name in variables_to_rename.items():
            if old_name in ds.variables:
                ds = ds.rename({old_name: new_name})

        return ds
    
    @staticmethod
    def oil_volume_shapefile(config, dens=0.922, thick=0.00001):
        """
        This method considers that the simulation will start from a shapefile, providing an area.

        Thefore, volume of oil will be calculated from it, since it is difficult to estimate without the proper knowledge.

        density and thickness are defined as standard, but could be modified for different oil slick behaviour or oil characteristic.

        """

        obs = gpd.read_file(config["input_files"]["shapefile"]["shape_path"])

        ######## OBTAINING THE VOLUME FOR THE GIVEN SHAPEFILE ########

        # Get the centroid from all the slicks in the observation or modeled results
        centroid_lon = obs.centroid.x.mean()
        centroid_lat = obs.centroid.y.mean()

        # Calculate the UTM zone
        utm_zone = int((centroid_lon + 180) / 6) + 1

        # Define the UTM CRS based on the determined zone
        utm_crs = CRS.from_epsg(32600 + utm_zone)  # EPSG code for UTM zones

        # Reproject the GeoDataFrame to UTM
        gdf_utm = obs.to_crs(utm_crs)

        # Calculate the area (in square meters) of the reprojected shapefile
        area = gdf_utm.area.sum()

        # Obtain the volume from the calculated area, density and thickness.
        volume = np.round(area * dens * thick, 2)

        return volume


if __name__ == "__main__":
    path = "config1.txt"
    try:
        values = Utils.read_txt_config1(path, "sim_length", "SIM_NAME", "lat_degree")
    except ValueError:
        pass
    values = Utils.read_txt_config1(path, "sim_length", "lat_degree")
    print(values)
    print(Utils.set_product("era5"))
    print(Utils.set_product("mercator_daily"))
    result = Utils.compute_domain(120, 33, 41, 35, 10)
    result2 = Utils.compute_domain(480, 33, 41, 35, 10)

    if result2 != result:
        raise AssertionError(f"Validation failed: result2 ({result2}) != result ({result})")

    print(result)
