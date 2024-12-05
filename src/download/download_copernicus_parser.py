import copernicusmarine
import argparse
import os
import subprocess
import datetime
import pandas as pd
import xarray as xr

# Functions outside this script
from WITOIL_iMagine.src.utils.utils import *


def download_copernicus(
    min_lat,
    max_lat,
    min_lon,
    max_lon,
    min_depth,
    max_depth,
    start_time,
    end_time,
    region,
    output_path,
    output_name,
    user,
    password,
):

    if region == "global":

        if end_time < pd.to_datetime("2022-01-01"):
            if end_time > pd.to_datetime("2021-06-30"):
                dataset_id = "cmems_mod_glo_phy_myint_0.083deg_P1D-m"
            else:
                dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
            output_name = output_name.format("reanalysis")
        else:
            dataset_id = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
            output_name = output_name.format("analysis")

        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=["uo", "vo", "thetao"],
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=start_time,
            end_datetime=end_time,
            minimum_depth=min_depth,
            maximum_depth=max_depth,
            output_filename="temp.nc",
            output_directory=output_path,
            username=user,
            password=password,
            force_download=True,
        )

        # Transform to medslik standards
        ds = xr.open_mfdataset(f"{output_path}temp.nc")

        # Rename variables only if they exist in the dataset
        ds = Utils.rename_netcdf_variables_mdk3(ds)

        # Selecting only 4 layers
        ds = ds.sel(depth=[0, 10, 30, 120], method="nearest")
        # Modifying labels to simplfy drop in temperature columns
        ds["depth"] = [0, 10, 30, 120]

        # Selecting only the relavent variables
        ds = ds[["uo", "vo", "thetao"]]

        # saves the daily current or temperature netcdf in the case dir
        ds.to_netcdf(output_name)

        # remove the temporary files
        temp_file = os.path.join(output_path, "temp.nc")
        if os.path.exists(temp_file):
            os.remove(temp_file)

    else:

        if end_time < pd.to_datetime("2021-11-01"):
            dataset_id_curr = "med-cmcc-cur-rean-h"
            dataset_id_temp = "med-cmcc-tem-rean-d"
            output_name = output_name.format("reanalysis")
        else:
            dataset_id_curr = "cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m"
            dataset_id_temp = "cmems_mod_med_phy-tem_anfc_4.2km-2D_PT1H-m"
            output_name = output_name.format("analysis")

        files = []
        for dataset in [dataset_id_curr, dataset_id_temp]:

            if "cur" in dataset:
                copernicusmarine.subset(
                    dataset_id=dataset_id_curr,
                    variables=["uo", "vo"],
                    minimum_longitude=min_lon,
                    maximum_longitude=max_lon,
                    minimum_latitude=min_lat,
                    maximum_latitude=max_lat,
                    start_datetime=start_time,
                    end_datetime=end_time,
                    minimum_depth=min_depth,
                    maximum_depth=max_depth,
                    output_filename="curr.nc",
                    output_directory=output_path,
                    username=user,
                    password=password,
                    force_download=True,
                )

                files.append(output_path + "curr.nc")
            else:
                copernicusmarine.subset(
                    dataset_id=dataset_id_temp,
                    variables=["thetao"],
                    minimum_longitude=min_lon,
                    maximum_longitude=max_lon,
                    minimum_latitude=min_lat,
                    maximum_latitude=max_lat,
                    start_datetime=start_time,
                    end_datetime=end_time,
                    output_filename="temp.nc",
                    output_directory=output_path,
                    username=user,
                    password=password,
                    force_download=True,
                )

                files.append(output_path + "temp.nc")

        # Transform to medslik standards
        ds = xr.open_mfdataset(files)

        # Rename variables only if they exist in the dataset
        ds = Utils.rename_netcdf_variables_mdk3(ds)

        # Selecting only 4 layers
        try:
            ds = ds.sel(depth=[0, 10, 30, 120], method="nearest")
            # Modifying labels to simplfy drop in temperature columns
            ds["depth"] = [0, 10, 30, 120]
        except:
            ds = ds.expand_dims(dim={"depth": [0, 10, 30, 120]})

        # Selecting only the relavent variables
        ds = ds[["uo", "vo", "thetao"]]

        # saves the daily current or temperature netcdf in the case dir
        ds.to_netcdf(output_name)

        # remove the temporary files
        temp_files = [os.path.join(output_path, "curr.nc"), os.path.join(output_path, "temp.nc")]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
