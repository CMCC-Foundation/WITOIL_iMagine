# ------------------------------------------------
# MEDSLIK-II oil spill fate and transport model
# ------------------------------------------------
import sys, os
import shutil
import logging
import warnings
import subprocess
import logging
import importlib.util
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser

# Import medslik modules
try:
    from .utils import Utils, Config, read_oilbase
    from .download import *
    from .preprocessing import PreProcessing
    from .postprocessing import PostProcessing
    from .plot import MedslikIIPlot
except (ImportError, ModuleNotFoundError):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import Utils, Config, read_oilbase
    from src.download import *
    from src.preprocessing import PreProcessing
    from src.postprocessing import PostProcessing
    from src.plot import MedslikIIPlot


# Import pyngl if present
package_spec = importlib.util.find_spec("Ngl")
if package_spec is not None:
    import Ngl

    _has_pyngl = True
else:
    _has_pyngl = False


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create a file handler with overwrite mode ('w')
# Get the current time
current_time = datetime.datetime.now()
# Format the time (e.g., Year-Month-Day_Hour-Minute-Second)
time_str = current_time.strftime("%H%M%S_%f")[:-3]
# Create the log file name with the timestamp
main_logfile_name = f"medslik_run_{time_str}.log"
if not logger.handlers:
    file_handler = logging.FileHandler(main_logfile_name, mode="w")
    file_handler.setLevel(logging.DEBUG)
    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(file_handler)


class MedslikII:
    """
    This class embeds the MAIN code of medslik-II software.
    """

    def __init__(
        self, config: dict, my_logger: logging.Logger, land_check: bool = True
    ) -> None:
        """
        Class constructor given config file path.
        """
        self.config = config
        self.my_logger = my_logger
        self.medslik_directory = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        # Create experiment directories
        self.exp_directory = os.path.join(
            self.config["simulation_setup"]["experiment_path"],
            config["simulation_setup"]["name"],
        )
        os.makedirs(self.exp_directory, exist_ok=True)
        self.out_directory = os.path.join(self.exp_directory, "out_files")
        os.makedirs(self.out_directory, exist_ok=True)
        self.out_figures = os.path.join(self.out_directory, "figures")
        os.makedirs(self.out_figures, exist_ok=True)
        self.xp_directory = os.path.join(self.exp_directory, "xp_files")
        os.makedirs(self.xp_directory, exist_ok=True)
        os.makedirs(f"{self.exp_directory}/oce_files", exist_ok=True)
        os.makedirs(f"{self.exp_directory}/met_files", exist_ok=True)
        os.makedirs(f"{self.exp_directory}/bnc_files", exist_ok=True)
        # Path to preprocessed data
        self.preproc_path = self.exp_directory
        try:
            preproc_path = self.config["input_files"]["preproc_path"]
            if preproc_path:
                if os.path.exists(preproc_path):
                    self.preproc_path = preproc_path
        except KeyError:
            pass
        # Sea Over Land extrapolaton method
        try:
            sol_method = self.config["input_files"]["sol_extrap_method"]
            self.sol_method = sol_method
        except KeyError:
            self.sol_method = "gradient"  # default
        # Spill coordinates
        self.spill_lat = (
            config["initial_conditions"]["spill_lat_deg"]
            + config["initial_conditions"]["spill_lat_min"] / 60.0
        )
        self.spill_lon = (
            config["initial_conditions"]["spill_lon_deg"]
            + config["initial_conditions"]["spill_lon_min"] / 60.0
        )
        self.__set_domain()
        self.__initial_checking(land_check)

    def __set_domain(self) -> None:
        """
        Set simulation domain.
        """
        config = self.config
        # Domain of the simulation will be defined under what the user set in config files
        if config["input_files"]["set_domain"] == True:
            self.my_logger.info("User defined domain")
            lat_min, lat_max = (
                config["input_files"]["latitude"][0],
                config["input_files"]["latitude"][1],
            )
            lon_min, lon_max = (
                config["input_files"]["longitude"][0],
                config["input_files"]["longitude"][1],
            )
        # Domain is based on a delta degrees
        else:
            self.my_logger.info(
                f"Domain defined around simulation point, \
                using {config['input_files']['delta']} degrees"
            )
            latitude = self.spill_lat
            longitude = self.spill_lon
            lat_min, lat_max = (
                latitude - config["input_files"]["delta"] / 2.0,
                latitude + config["input_files"]["delta"] / 2.0,
            )
            lon_min, lon_max = (
                longitude - config["input_files"]["delta"] / 2.0,
                longitude + config["input_files"]["delta"] / 2.0,
            )
        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max

    def __initial_checking(self, land_check=True):
        """
        Check if any issue might derive from configuration.
        """
        # checking if the coordinates are on land
        lat = self.spill_lat
        lon = self.spill_lon
        # function to check if the spill location was put into land
        coastline_path = self.config["input_files"]["dtm"]["coastline_path"]
        if land_check:
            if self.config["run_options"]["run_model"]:
                sea = Utils.check_land(lon, lat, coastline_path)
                if sea == 0:
                    raise ValueError(
                        "Your coordinates lie within land. Please check your values again"
                    )
        # checking dates
        dt = Utils.validate_date(self.config["initial_conditions"]["start_datetime"])
        self.config["initial_conditions"]["start_datetime"] = dt
        self.my_logger.info(
            f"Date {dt}; No major issues found on dates and oil spill coordinates"
        )
        # checking if starting from area spill
        if self.config["input_files"]["shapefile"]["shape_path"]:
            shapefile_path = self.config["input_files"]["shapefile"]["shape_path"]
            # if simulation starts from shapefile, the volume will be disconsidered
            if os.path.exists(shapefile_path):
                self.my_logger.info(
                    f"Simulation initial conditions area spill are provided on \
                        {self.config['input_files']['shapefile']['shape_path']}. \
                        Spill rate from config files will not be considered"
                )
                volume = Utils.oil_volume_shapefile(self.config)
                # Correcting volume on the config object
                self.config["initial_conditions"]["spill_rate"] = volume

    def toml_to_parameters(self, path_to_toml: str, txt_file: str) -> None:
        """
        Write parameters.txt from parameters.toml file.
        """
        toml_file = Config(path_to_toml).config_dict
        # Ensure 'computational_parameters' is created
        toml_file.setdefault("computational_parameters", {})["time_step"] = int(
            self.config["simulation_setup"]["time_step"]
        )
        f = open(txt_file, mode="w")
        for key, keyval in toml_file.items():
            f.write(key + "\n")
            for key, val in keyval.items():
                if type(val) == list:
                    string_val = " ".join(map(str, val))
                else:
                    string_val = str(val)
                if string_val == "True":
                    string_val = "1"
                if string_val == "False":
                    string_val = "0"
                f.write(string_val + "\n")
        f.close()

    def config_to_input(
        self, config: dict, oil_file_path: str, path_to_inp_file: str
    ) -> None:
        """
        Convert config dictionary into medslik.inp file.
        """
        current_path = os.path.dirname(os.path.abspath(__file__))
        template_file = os.path.join(current_path, "model", "oilspill.inp")
        simulation = config["simulation_setup"]
        init = config["initial_conditions"]
        # Checks if spill starts from area spill or point source
        if init["contour_slick"]:
            isat = 1
        else:
            isat = 0

        replace_dict = {
            "restart_hr": "0",
            "restart_min": "0",
            "day": init["start_datetime"].day,
            "month": init["start_datetime"].month,
            "year": init["start_datetime"].year,
            "hour": init["start_datetime"].hour,
            "minutes": f"{init['start_datetime'].minute:02d}",
            "isat": isat,
            "spl_dur": int(init["spill_duration"]),
            "sim_length": int(simulation["sim_length"]),
            "spl_lat": self.spill_lat,
            "spl_lon": self.spill_lon,
            "splrate": init["spill_rate"],
        }
        with open(template_file, "r") as file:
            file_contents = file.read()
        for key, val in replace_dict.items():
            file_contents = file_contents.replace(key, str(val))
        file_contents = file_contents.replace("[", "")
        file_contents = file_contents.replace("]", "")
        with open(path_to_inp_file, "w") as file:
            file.write(file_contents)
        # Add oil contents
        with open(oil_file_path, "r") as file:
            oil_contents = file.read()
        with open(path_to_inp_file, "a") as file:
            file.write(oil_contents)
        # Add forecast days dates
        with open(path_to_inp_file, "a") as file:
            n_days = int(simulation["sim_length"] / 24.0) + 1
            file.write(str(n_days) + "\n")
            date_array = np.arange(
                init["start_datetime"],
                init["start_datetime"] + pd.Timedelta(days=n_days),
                pd.Timedelta(days=1),
            ).astype(datetime.datetime)
            for date in date_array:
                # Convert to 'yyMMdd' format
                formatted_date = date.strftime("%y%m%d")
                file.write(formatted_date)
                file.write("\n")

    def run_medslik_sim(self):
        """
        Run Medslik-II simulation.
        """
        # model directory. Could be changed, but will remain fixed for the time being.
        current_path = os.path.dirname(os.path.abspath(__file__))
        exec_path = os.path.join(current_path, "model", "bin", "simulation.exe")
        # Compile and start running
        subprocess.run(
            [f"{exec_path} {self.exp_directory} {self.preproc_path}"],
            shell=True,
            check=True,
        )

    def run_preproc(self, preproc: PreProcessing):
        """
        Run preprocessing.
        """
        config = preproc.config
        domain = preproc.domain
        exp_folder = preproc.exp_folder
        oce_path = config["input_files"]["metoce"]["oce_data_path"]
        met_path = config["input_files"]["metoce"]["met_data_path"]
        if oce_path == "":
            oce_path = None
        if met_path == "":
            met_path = None
        preproc_metoce = config["run_options"]["preprocessing_metoce"]
        preproc_dtm = config["run_options"]["preprocessing_dtm"]
        if preproc_metoce or preproc_dtm:
            self.preproc_path = self.exp_directory
        # download data if needed
        if config["download"]["download_data"] == True:
            self.my_logger.info("Downloading data")
            data_download_medslik(config, domain, exp_folder, self.my_logger)
        if preproc_metoce:
            self.my_logger.info("MET/OCE preprocessing")
            # create Medslik-II current file inputs
            preproc.process_currents(oce_path=oce_path, sol_method=self.sol_method)
            # create Medslik-II wind file inputs
            preproc.process_winds(met_path=met_path)
        if preproc_dtm:
            self.my_logger.info("DTM preprocessing")
            # use the same grid on currents to crop bathymetry
            preproc.common_grid(oce_path)
            # create Medslik-II bathymetry file inputs
            preproc.process_bathymetry(
                config["input_files"]["dtm"]["bathymetry_path"],
                sol_method=self.sol_method,
            )
            # create Medslik-II coastline file inputs
            preproc.process_coastline(config["input_files"]["dtm"]["coastline_path"])


if __name__ == "__main__":
    print("Running Medslik-II software")
    print(f"For further information, please check the log file: {logger.handlers}")
    # Logging first info
    logger.info("Starting Medslik-II oil spill simulation")
    exec_start_time = datetime.datetime.now()
    logger.info(f"Execution starting at {exec_start_time}")

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
    # Read config file
    config_path = args.config
    if config_path is None:
        config_path = os.path.join("config.toml")
    config_dict = Config(config_path).config_dict

    # Create object
    main = MedslikII(config_dict, logger, land_check=True)

    # Copy config file
    shutil.copy(config_path, os.path.join(main.xp_directory, "config.toml"))

    # Run preprocessing
    domain = [main.lon_min, main.lon_max, main.lat_min, main.lat_max]
    preprocessing = PreProcessing(
        config=main.config,
        exp_folder=main.exp_directory,
        domain=domain,
        my_logger=logger,
    )
    main.run_preproc(preproc=preprocessing)

    # create initial condition area oil spill euther from vertices or shapefile
    if main.config["initial_conditions"]["contour_slick"]:
        shapefile_path = main.config["input_files"]["shapefile"]["shape_path"]
        if not shapefile_path:
            # passing the area vertices to generate the polygon
            preprocessing.write_initial_input(
                main.config["initial_conditions"]["slick_vertices"]
            )
        elif shapefile_path == "none":
            # passing the area vertices to generate the polygon
            preprocessing.write_initial_input(
                main.config["initial_conditions"]["slick_vertices"]
            )
        else:
            if os.path.exists(shapefile_path):
                # using an area spill identified on a shapefile to generate initial conditions
                preprocessing.process_initial_shapefile()
            else:
                raise FileNotFoundError(f"Shapefile {shapefile_path} not found")

    # Log preproc exec time
    preproc_end_time = datetime.datetime.now()
    logger.info(
        f"Preprocessing execution time = {pd.Timedelta(preproc_end_time - exec_start_time)}"
    )
    # Set this as simulation end time in case model is not run
    simulation_end_time = preproc_end_time

    # Write input files to simulation and run simulation
    if main.config["run_options"]["run_model"]:
        # Create oil file
        oil_path = os.path.join(
            main.medslik_directory, "input_data", "oil", "oilbase.csv"
        )
        logger.info(f"Reading oil from database {oil_path}")
        oil_value = main.config["initial_conditions"]["oil_api"]
        oil_option = "api"
        oil_out_path = read_oilbase(oil_option, oil_value, oil_path, main.xp_directory)
        logger.info(f"Created oil file {oil_out_path}")
        # Create input file
        oilspill_input_file = os.path.join(main.xp_directory, "oilspill.inp")
        main.config_to_input(main.config, oil_out_path, oilspill_input_file)
        logger.info(f"Created input file {oilspill_input_file}")
        # Create parameters file
        parameters_file = os.path.join(main.xp_directory, "parameters.txt")
        toml_parameters = main.config["simulation_setup"]["advanced_parameters_path"]
        logger.info(f"Reading parameters file {toml_parameters}")
        main.toml_to_parameters(toml_parameters, parameters_file)
        shutil.copy(toml_parameters, os.path.join(main.xp_directory, "parameters.toml"))
        logger.info(f"Created parameters file {parameters_file}")
        # Run model
        logger.info("Running Medslik-II simulation")
        main.run_medslik_sim()
        # Log simulation exec time
        simulation_end_time = datetime.datetime.now()
        logger.info(
            f"Simulation execution time = {pd.Timedelta(simulation_end_time - preproc_end_time)}"
        )

    # performing post processing
    if main.config["run_options"]["postprocessing"]:
        logger.info("Running postprocessing")
        try:
            resolution = main.config["simulation_setup"]["oil_tracer_grid_step"]
        except KeyError:
            resolution = 150.0
        PostProcessing.create_concentration_dataset(
            lon_min=main.lon_min,
            lon_max=main.lon_max,
            lat_min=main.lat_min,
            lat_max=main.lat_max,
            filepath=main.out_directory,
            multiple_slick=False,
            resolution=resolution,
        )
        # Log postproc exec time
        postproc_end_time = datetime.datetime.now()
        logger.info(
            f"Postprocessing execution time = {pd.Timedelta(postproc_end_time - simulation_end_time)}"
        )

    # plotting the results
    if main.config["plot_options"]["plotting"]:
        logger.info("Plotting results")
        try:
            plot_step = main.config["plot_options"]["plot_step_hrs"]
        except KeyError:
            plot_step = 1
        try:
            crange = main.config["plot_options"]["concentration_range"]
        except KeyError:
            crange = None
        if crange == []:
            crange = None
        # Plot object
        mplot = MedslikIIPlot(
            main.config,
            main.spill_lat,
            main.spill_lon,
            main.exp_directory,
            main.out_directory,
            main.out_figures,
        )
        try:
            plot_curr_method = main.config["plot_options"]["currents_plotting_method"]
        except KeyError:
            plot_curr_method = "curly"
        try:
            plot_extrap_currents = main.config["plot_options"]["plot_extrap_currents"]
        except KeyError:
            plot_extrap_currents = False
        # Concentration plot
        if _has_pyngl and main.config["plot_options"]["pyngl"]:
            mplot.plot_pyngl(plot_step=plot_step, crange=crange)
        else:
            mplot.plot_matplotlib(
                plot_step=plot_step,
                crange=crange,
                plot_curr_method=plot_curr_method,
                plot_extrap_currents=plot_extrap_currents,
            )
        try:
            mplot.create_gif()
        except Exception as e:
            print(f"An error occurred while creating the oil trajectory GIF: {e}")
            pass
        # mass balance plot
        mplot.plot_mass_balance()
        # Plotting beached oil points
        mplot.plot_beached_oil(plot_step=plot_step)
        try:
            mplot.create_beached_oil_gif()
        except Exception as e:
            print(f"An error occurred while creating the beached oil GIF: {e}")
            pass

    # Log execution time
    exec_end_time = datetime.datetime.now()
    total_exec_time = pd.Timedelta(exec_end_time - exec_start_time)
    logger.info(f"Execution ending at {exec_end_time}")
    logger.info(f"Total execution time = {total_exec_time}")
    shutil.move(main_logfile_name, os.path.join(main.exp_directory, main_logfile_name))
    print(f"END - Total execution time = {total_exec_time}")
