import subprocess
import pandas as pd
from . import *
from logging import Logger


def data_download_medslik(
    config: dict,
    domain: list[float],
    root_directory: str,
    logger: Logger,
) -> None:
    """
    Download METOCE datasets.
    """
    lon_min, lon_max, lat_min, lat_max = domain
    copernicus_user = config["download"]["copernicus_user"]
    copernicus_pass = config["download"]["copernicus_password"]
    date = pd.to_datetime(config["initial_conditions"]["start_datetime"])
    identifier = str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
    inidate = date - pd.Timedelta(hours=1)
    enddate = date + pd.Timedelta(hours=config["simulation_setup"]["sim_length"] + 24)
    if (
        30.37 < np.mean([lat_min, lat_max]) < 45.7
        and -17.25 < np.mean([lon_min, lon_max]) < 36
    ):
        down = "local"
    else:
        down = "global"
    if config["download"]["download_curr"]:
        output_path = "input_data/ocean/COPERNICUS/"
        output_name = output_path + "Copernicus{}_{}_{}_mdk.nc".format(
            "{}", identifier, config["simulation_setup"]["name"]
        )
        logger.info("Downloading CMEMS currents")
        download_copernicus(
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            0,
            120,
            inidate,
            enddate,
            down,
            output_path=output_path,
            output_name=output_name,
            user=copernicus_user,
            password=copernicus_pass,
        )
        subprocess.run(
            [
                f'cp {output_path}*{identifier}*{config["simulation_setup"]["name"]}*.nc {root_directory}/oce_files/'
            ],
            shell=True,
        )
        subprocess.run(
            [f'rm {output_path}*{identifier}*{config["simulation_setup"]["name"]}*.nc'],
            shell=True,
        )
    if config["download"]["download_wind"]:

        # ensuring .cdsapirc is created in the home directory
        write_cds(config["download"]["cds_token"])

        output_path = "input_data/atmosphere/ERA5/"
        output_name = output_path + "era5_winds10_{}_{}_mdk.nc".format(
            identifier, config["simulation_setup"]["name"]
        )
        logger.info("Downloading ERA5 reanalysis winds")
        get_era5(
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            inidate,
            enddate,
            output_path=output_path,
            output_name=output_name,
        )
        process_era5(output_path=output_path, output_name=output_name)
        subprocess.run(
            [
                f'cp {output_path}*{identifier}*{config["simulation_setup"]["name"]}*.nc {root_directory}/met_files/'
            ],
            shell=True,
        )
        subprocess.run(
            [f'rm {output_path}*{identifier}*{config["simulation_setup"]["name"]}*.nc'],
            shell=True,
        )


if __name__ == "__main__":
    pass
