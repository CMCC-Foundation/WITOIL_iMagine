# ------------------------------------------------
# MEDSLIK-II oil spill fate and transport model
# ------------------------------------------------
import sys, os
import shutil
import logging
import warnings
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from glob import glob as gg

# Import medslik modules
from src.utils import Utils, Config, read_oilbase
from src.download import *
from src.preprocessing import PreProcessing
from src.postprocessing import PostProcessing
from src.plot import MedslikIIPlot

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler with overwrite mode ('w')
file_handler = logging.FileHandler("medslik_run.log", mode="w")
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


class MedslikII:
    """
    This class embeds the MAIN code of medslik-II software.
    """

    def __init__(self, config: dict) -> None:
        """
        Class constructor given config file path.
        """
        self.config = config
        self.medslik_directory = os.path.dirname(os.path.abspath(__file__))
        # Create experiment directories
        self.root_directory = os.path.join(
            self.config["simulation"]["experiment_path"], config["simulation"]["name"]
        )
        os.makedirs(self.root_directory, exist_ok=True)
        self.out_directory = os.path.join(self.root_directory, "out_files")
        os.makedirs(self.out_directory, exist_ok=True)
        self.out_figures = os.path.join(self.out_directory, "figures")
        os.makedirs(self.out_figures, exist_ok=True)
        self.xp_directory = os.path.join(self.root_directory, "xp_files")
        os.makedirs(self.xp_directory, exist_ok=True)
        spill_lat = np.array(self.config["simulation"]["spill_lat"])
        self.n_spill_points = np.shape(spill_lat)[0]
        # Domain of the simulation will be defined under what the user set in config files
        if config["input_files"]["set_domain"] == True:
            logger.info("User defined domain")
            lat_min, lat_max = (
                config["input_files"]["lat"][0],
                config["input_files"]["lat"][1],
            )
            lon_min, lon_max = (
                config["input_files"]["lon"][0],
                config["input_files"]["lon"][1],
            )
        #Domain is based on a delta degrees
        else:
            logger.info(f"Domain defined around simulation point, \
                using {config['input_files']['delta'][0]} degrees")
            latitude = config["simulation"]["spill_lat"][0]
            longitude = config["simulation"]["spill_lon"][0]

            lat_min, lat_max = (
                latitude - config["input_files"]["delta"][0],
                latitude + config["input_files"]["delta"][0],
            )
            lon_min, lon_max = (
                longitude - config["input_files"]["delta"][0],
                longitude + config["input_files"]["delta"][0],
            )
        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max
        self.apply_aging_effects()
        self.initial_checking()

    def apply_aging_effects(self) -> None:
        """
        Consider aging when setting
        the simulation length and the start datetime.
        """
        oilspill = self.config["simulation"]
        age = oilspill["slick_age"]
        if self.n_spill_points > 1:
            oilspill["sim_length"] += age
            oilspill["start_datetime"] -= pd.Timedelta(hours=age)
            oilspill["spill_duration"] = 0
        self.config["simulation"] = oilspill

    def initial_checking(self):
        """
        Check if any issue might derive from configuration.
        """
        # checking if the coordinates are on land
        lat = self.config["simulation"]["spill_lat"]
        lon = self.config["simulation"]["spill_lon"]
        # function to check if the spill location was put into land
        coastline_path = self.config["input_files"]["dtm"]["coastline_path"]
        sea = Utils.check_land(lon, lat, coastline_path)
        if sea == 0:
            raise ValueError(
                "Your coordinates lie within land. Please check your values again"
            )
        # checking dates
        dt = Utils.validate_date(self.config["simulation"]["start_datetime"])
        logger.info("No major issues found on dates and oil spill coordinates")

        # checking if starting from area spill
        if self.config["input_files"]["shapefile"]["shape_path"]:
            shapefile_path = self.config["input_files"]["shapefile"]["shape_path"]
            # if simulation starts from shapefile, the volume will be disconsidered
            if os.path.exists(shapefile_path):
                logger.info(
                    f"Simulation initial conditions area spill are provided on \
                        {self.config['input_files']['shapefile']['shape_path']}. \
                        Spill rate from config files will not be considered"
                )
                volume = Utils.oil_volume_shapefile(self.config)

                # Correcting volume on the config object
                self.config["simulation"]["spill_rate"] = volume

    @staticmethod
    def data_download_medslik(
        config: dict, domain: list[float], root_directory: str
    ) -> None:
        """
        Download METOCE datasets.
        """
        lon_min, lon_max, lat_min, lat_max = domain
        copernicus_user = config["download"]["copernicus_user"]
        copernicus_pass = config["download"]["copernicus_password"]

        date = pd.to_datetime(config["simulation"]["start_datetime"])

        identifier = str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)

        inidate = date - pd.Timedelta(hours=1)
        enddate = date + pd.Timedelta(hours=config["simulation"]["sim_length"] + 24)

        if 30.37 < np.mean([lat_min,lat_max]) < 45.7 and -17.25 < np.mean([lon_min,lon_max]) < 36:
            down = "local"
        else:
            down = "global"

        if config["download"]["download_curr"]:

            output_path = "data/COPERNICUS/"
            output_name = output_path + "Copernicus{}_{}_{}_mdk.nc".format(
                "{}", identifier, config["simulation"]["name"]
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

            source_files = gg(f"{output_path}*{identifier}*{config['simulation']['name']}*.nc")
            destination = os.path.join(root_directory, "oce_files")
            for file in source_files:
                shutil.copy(file, destination)

            for file in gg(f"{output_path}*{identifier}*{config['simulation']['name']}*.nc"):
                os.remove(file)

        if config["download"]["download_wind"]:
            # ensuring .cdsapirc is created in the home directory
            write_cds(config["download"]["cds_token"])

            output_path = "data/ERA5/"
            output_name = output_path + "era5_winds10_{}_{}_mdk.nc".format(
                identifier, config["simulation"]["name"]
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

            source_files = gg(f"{output_path}*{identifier}*{config['simulation']['name']}*.nc")
            destination = os.path.join(root_directory, "met_files")
            for file in source_files:
                shutil.copy(file, destination)

            for file in gg(f"{output_path}*{identifier}*{config['simulation']['name']}*.nc"):
                os.remove(file)

    def run_preproc(config: dict, exp_folder: str, lon_min, lon_max, lat_min, lat_max):
        """
        Run preprocessing.
        """

        domain = [lon_min, lon_max, lat_min, lat_max]
        preproc = PreProcessing(config=config, exp_folder=exp_folder, domain=domain)
        # Create folders
        preproc.create_directories()

        # download data if needed
        if config["download"]["download_data"] == True:
            MedslikII.data_download_medslik(config, domain, exp_folder)

        if config["run_options"]["preprocessing"]:

            if config["run_options"]["preprocessing_metoce"]:
                oce_path = config["input_files"]["metoce"]["oce_data_path"]
                met_path = config["input_files"]["metoce"]["met_data_path"]
                if oce_path == "":
                    oce_path = None
                if met_path == "":
                    met_path = None
                #create Medslik-II current file inputs
                preproc.process_currents(oce_path=f"{exp_folder}/oce_files/")
                #create Medslik-II wind file inputs
                preproc.process_winds(met_path=f"{exp_folder}/met_files/")

            if config["run_options"]["preprocessing_dtm"]:
               #use the same grid on currents to crop bathymetry
                preproc.common_grid()
                #create Medslik-II bathymetry file inputs
                preproc.process_bathymetry(
                    config["input_files"]["dtm"]["bathymetry_path"]
                )
                #create Medslik-II coastline file inputs
                preproc.process_coastline(
                    config["input_files"]["dtm"]["coastline_path"]
                )
            
            if config["input_files"]["shapefile"]["shape_path"]:
                    shapefile_path = config["input_files"]["shapefile"]["shape_path"]
                    if os.path.exists(shapefile_path):
                        # using an area spill identified on a shapefile to generate initial conditions
                        preproc.process_initial_shapefile()

            spill_dictionary = {}
            if main.n_spill_points > 1:
                logger.info(
                    f"Starting to write {main.n_spill_points} events of oil spill"
                )
                for i, dur in enumerate(config["spill_rate"]):
                    # obtaining the variables
                    spill_dictionary["simname"] = preproc.simname
                    spill_dictionary["dt_sim"] = main.config["simulation"][
                        "start_datetime"
                    ]
                    spill_dictionary["sim_length"] = preproc.sim_length
                    spill_dictionary["longitude"] = main.config["simulation"][
                        "spill_lon"
                    ][i]
                    spill_dictionary["latitude"] = main.config["simulation"][
                        "spill_lat"
                    ][i]
                    spill_dictionary["spill_duration"] = int(
                        main.config["simulation"]["spill_duration"][i]
                    )
                    spill_dictionary["spill_rate"] = main.config["simulation"][
                        "spill_rate"
                    ][i]
                    spill_dictionary["oil_api"] = main.config["simulation"]["oil"][i]
                    spill_dictionary["number_slick"] = 1
                    preproc.write_config_files(
                        spill_dictionary, separate_slicks=True, s_num=i
                    )

            else:
                logger.info("Writing single slick event")
                spill_dictionary["simname"] = preproc.simname
                spill_dictionary["dt_sim"] = main.config["simulation"]["start_datetime"]
                spill_dictionary["sim_length"] = int(preproc.sim_length)
                spill_dictionary["longitude"] = main.config["simulation"]["spill_lon"][
                    0
                ]
                spill_dictionary["latitude"] = main.config["simulation"]["spill_lat"][0]
                spill_dictionary["spill_duration"] = int(
                    main.config["simulation"]["spill_duration"][0]
                )
                spill_dictionary["spill_rate"] = main.config["simulation"][
                    "spill_rate"
                ][0]
                spill_dictionary["oil_api"] = main.config["simulation"]["oil"][0]
                preproc.write_config_files(spill_dictionary, separate_slicks=False)

            logger.info("Modfying medslik_II.for")
            preproc.process_medslik_memmory_array()
            logger.info("Medslik-II simulation parameters")
            if config["simulation"]["advanced_parameters"] == False:
                logger.info("Using custom advanced parameters")
                preproc.configuration_parameters()
            else:
                logger.info("Using standard parameters")

    def run_medslik_sim(self, simdir, simname, separate_slicks=False):

        # model directory. Could be changed, but will remain fixed for the time being.
        model_dir = "src/model/"

        day = self.config["simulation"]["start_datetime"].day
        year = self.config["simulation"]["start_datetime"].year
        month = self.config["simulation"]["start_datetime"].month
        hour = self.config["simulation"]["start_datetime"].hour
        minute = self.config["simulation"]["start_datetime"].minute

        output_dir = f"{model_dir}OUT/MDK_SIM_{year}_{month:02d}_{day:02d}_{hour:02d}{minute:02d}_{simname}/."

        # Remove old outputs (equivalent to `rm -rf`)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        if not separate_slicks:
            # Copy METOCEAN files
            oce_files = gg(f"{simdir}{simname}/oce_files/*.mrc")
            met_files = gg(f"{simdir}{simname}/met_files/*.eri")
            bnc_files = gg(f"{simdir}{simname}/bnc_files/*")
            xp_files = {
                "medslik_II.for": os.path.join(simdir, simname, "xp_files", "medslik_II.for"),
                "config2.txt": os.path.join(simdir, simname, "xp_files", "config2.txt"),
                "config1.txt": os.path.join(simdir, simname, "xp_files", "config1.txt"),
            }

            # Copy METOCEAN, MET, and BNC files
            for file in oce_files:
                shutil.copy(file, os.path.join(model_dir, "RUN", "TEMP", "OCE"))
            for file in met_files:
                shutil.copy(file, os.path.join(model_dir, "RUN", "TEMP", "MET"))
            for file in bnc_files:
                shutil.copy(file, os.path.join(model_dir, "DTM_INP"))

            # Copy other required files
            for dest, src in xp_files.items():
                shutil.copy(src, os.path.join(model_dir, "RUN", dest if "config" in dest else "MODEL_SRC"))

            # Compile and start running (replacing `cd` with `cwd`)
            compile_script_path = os.path.abspath(os.path.join(model_dir, "RUN", "MODEL_SRC", "compile.sh"))
            run_script_path = os.path.abspath(os.path.join(model_dir, "RUN", "RUN.sh"))
            if not os.access(compile_script_path, os.X_OK):
                os.chmod(compile_script_path, 0o755)

            if not os.access(run_script_path, os.X_OK):
                os.chmod(run_script_path, 0o755) 
            subprocess.run([compile_script_path], check=True, cwd=os.path.join(model_dir, "RUN"))
            subprocess.run([run_script_path], check=True, cwd=os.path.join(model_dir, "RUN"))

        else:
            # Handle separate slicks
            slicks = gg(f"{simdir}{simname}/xp_files/*/")
            for i, slick_dir in enumerate(slicks):
                # Copy METOCEAN, MET, and BNC files
                oce_files = gg(f"{simdir}{simname}/oce_files/*.mrc")
                met_files = gg(f"{simdir}{simname}/met_files/*.eri")
                bnc_files = gg(f"{simdir}{simname}/bnc_files/*")
                config1_path = os.path.join(simdir, simname, f"xp_files/slick{i + 1}/config1.txt")

                for file in oce_files:
                    shutil.copy(file, os.path.join(model_dir, "RUN", "TEMP", "OCE"))
                for file in met_files:
                    shutil.copy(file, os.path.join(model_dir, "RUN", "TEMP", "MET"))
                for file in bnc_files:
                    shutil.copy(file, os.path.join(model_dir, "DTM_INP"))
                shutil.copy(config1_path, os.path.join(model_dir, "RUN", "config1.txt"))

                # Compile and start running
                compile_script_path = os.path.abspath(os.path.join(model_dir, "RUN", "MODEL_SRC", "compile.sh"))
                run_script_path = os.path.abspath(os.path.join(model_dir, "RUN", "RUN.sh"))
                if not os.access(compile_script_path, os.X_OK):
                    os.chmod(compile_script_path, 0o755)

                if not os.access(run_script_path, os.X_OK):
                    os.chmod(run_script_path, 0o755) 
                subprocess.run([compile_script_path], check=True, cwd=os.path.join(model_dir, "RUN"))
                subprocess.run([run_script_path], check=True, cwd=os.path.join(model_dir, "RUN"))

        # Copy output files (replacing `cp -r`)
        output_dest = os.path.join(simdir, simname, "out_files")
        if os.path.exists(output_dir):
            shutil.copytree(output_dir, output_dest, dirs_exist_ok=True)

        # Remove temporary MET and OCE files (equivalent to `rm -rf`)
        temp_met = os.path.join(output_dest, "MET")
        temp_oce = os.path.join(output_dest, "OCE")
        if os.path.exists(temp_met):
            shutil.rmtree(temp_met)
        if os.path.exists(temp_oce):
            shutil.rmtree(temp_oce)


if __name__ == "__main__":
    # Logging first info
    logger.info("Starting Medslik-II oil spill simulation")

    # Config as argument
    parser = ArgumentParser(
        description="Medslik-II # oil spill fate and transport model"
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to configuration file",
        required=False,
        default=None,
    )
    args = parser.parse_args()
    config_path = args.config

    if config_path is None:
        config_path = os.path.join("config.toml")

    logger.info("Defining the main object")
    config = Config(config_path).config_dict

    main = MedslikII(config)

    shutil.copy(config_path, os.path.join(main.xp_directory, "config.toml"))

    # performing initial checking
    main.initial_checking()

# Run preprocessing
    logger.info("Starting pre processing ... ")
    preproc = MedslikII.run_preproc(
        main.config,
        main.root_directory,
        main.lon_min,
        main.lon_max,
        main.lat_min,
        main.lat_max,
    )
    logger.info("End of pre processing ...")

    # Run model
    if main.config["run_options"]["run_model"]:
        logger.info("Running Medslik-II simulation")
        main.run_medslik_sim(
            "cases/", main.config["simulation"]["name"], separate_slicks=False
        )

    #performing post processing
    if main.config["run_options"]["postprocessing"]:
        multiple_slick = main.config["simulation"]["multiple_slick"]
        PostProcessing.create_concentration_dataset(
            lon_min=main.lon_min,
            lon_max=main.lon_max,
            lat_min=main.lat_min,
            lat_max=main.lat_max,
            filepath=main.out_directory,
            multiple_slick=multiple_slick,
        )

    #plotting the results
    if main.config["plot_options"]["plotting"]:
        mplot = MedslikIIPlot(main)
        mplot.plot_matplotlib(main.lon_min, main.lon_max, main.lat_min, main.lat_max)
        mplot.plot_mass_balance()


    # if main.config["plot_options"]["plotting"]:
    #     mplot = MedslikIIPlot(main)
    #     mplot.plot_matplotlib(main.lon_min,main.lon_max,main.lat_min,main.lat_max)
    #     try:
    #         mplot.plot_mass_balance()
    #     except:
    #         pass

    shutil.copy("medslik_run.log", f"{main.out_directory}medslik_run.log")

    if config_path is None:
        shutil.copy("config.toml", f"{main.out_directory}config.toml")
