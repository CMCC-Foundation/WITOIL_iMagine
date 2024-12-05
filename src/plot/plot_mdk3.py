import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
import os, sys
import subprocess
import warnings
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import json
from shapely.geometry import box, Polygon


class MedslikIIPlot:
    """
    This class embeds functions for plotting.
    """

    def __init__(
        self,
        config: object,
    ) -> None:
        """
        Class constructor.
        """
        self.config = config.config
        self.root_directory = config.root_directory
        self.out_directory = config.out_directory
        self.out_figures = config.out_figures
        os.makedirs(self.out_figures, exist_ok=True)
        self.concentration_path = os.path.join(
            self.out_directory, "oil_concentration.nc"
        )

    def plot_matplotlib(self,lon_min,lon_max,lat_min,lat_max):

        # read coasline
        land = gpd.read_file(self.config["input_files"]["dtm"]["coastline_path"])

        # read output netcdf in concentration
        ds_particles = xr.open_dataset(self.concentration_path)

        # simulation initial and end dates
        inidate = pd.to_datetime(
            self.config["simulation"]["start_datetime"]
        ) + pd.Timedelta(hours=1.0)
        enddate = pd.to_datetime(
            inidate + pd.Timedelta(hours=self.config["simulation"]["sim_length"])
        )

        # opening currents netcdf
        curr = xr.open_mfdataset(f"{self.root_directory}/oce_files/*.nc")

        # ensuring date index is correct
        try:
            curr["time"] = curr.indexes["time"].to_datetimeindex()
        except (KeyError, AttributeError):
            pass

        curr = curr.resample(time="1h").interpolate("linear")
        # selecting simulation date
        curr = curr.sel(time=slice(inidate, enddate))
        curr = curr.isel(depth=0)

        # defining plot boundaries
        # lon_min, lon_max = self.config["plot_options"]["plot_lon"]
        # lat_min, lat_max = self.config["plot_options"]["plot_lat"]

        # cropping coastline to area of interest
        rec = land.cx[lon_min:lon_max, lat_min:lat_max]

        # selecting simulation domain
        curr = curr.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

        # loop for ploting
        for t in range(0, len(ds_particles.time)):

            # select the iteration timestep
            ds_p = ds_particles.isel(time=t)
            plot_curr = curr.isel(time=t)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_facecolor("#ADD8E6")

            # Ploting coastline
            rec.plot(ax=ax, color="#FFFDD0", edgecolor="black", zorder=1000, aspect=1)

            # ploting concentration (conversion to tons/km^2)
            ds_c1 = xr.where(
                ds_p.concentration * 1000 > 0.001, ds_p.concentration * 1000, np.nan
            )
            ds_c1.plot(ax=ax, cbar_kwargs={"label": r"Concentration (tons $km^{-2}$)"})

            # plotting current vectors
            ax.quiver(
                plot_curr.lon.values,
                plot_curr.lat.values,
                plot_curr.uo.values,
                plot_curr.vo.values,
                scale=10,
                width=0.0025,
                color="gray",
            )

            # plotting release point or center of mass
            ax.plot(
                self.config["simulation"]["spill_lon"],
                self.config["simulation"]["spill_lat"],
                marker="x",
                color="black",
            )

            plt.xlim(lon_min, lon_max)
            plt.ylim(lat_min, lat_max)

            plt.xlabel("Longitude (°E)")
            plt.ylabel("Latitude (°N)")

            plt.title(f"Oil Surface Concentration\n{t+1} hour(s) after oil release")

            plt.grid()

            plt.savefig(
                self.out_figures
                + f"/surf_oil_concentration_{self.config['simulation']['name']}_{t+1:03d}.png",
                dpi=200,
            )

            plt.close()

    def plot_mass_balance(self):

        path = self.out_directory
        filename = path + "/medslik.fte"

        header = pd.read_csv(
            filename, sep=r"\s\s+", skiprows=6, engine="python"
        ).columns.values
        df = pd.read_csv(filename, sep=r" +", skiprows=8, header=None, engine="python")
        df.columns = header

        df = df[["time", "%evap", "%srftot", "%disp", "%cstfxd", "%csttot"]]

        df.index = df.time
        df.drop("time", axis=1, inplace=True)

        df["%cstfxd"] = df["%csttot"] - df["%cstfxd"]

        # renaming columns
        df.columns = [
            "Percentage of Oil Evaporated",
            "Percentage of Oil on the Sea Surface",
            "Percentage of Oil Dispersed in the Water Column",
            "Percentage of Oil on the Coast but Potentially Releasable",
            "Total Percentage of Oil on the Coast",
        ]

        styles = ["-", "-", "-", "--", "-"]

        df.plot(style=styles, legend=True).legend(loc="upper right")

        plt.xlabel("Time (h)")
        plt.ylabel("Percentage (%)")
        plt.title("Mass Balance")

        plt.savefig(
            self.out_figures + f"/massbalance_{self.config['simulation']['name']}.png",
            dpi=100,
        )

        plt.close()

    def create_gif(self):
        path = self.out_figures
        subprocess.run(
            [
                f"magick -delay 20 -loop 0 {path}/*surf_oil_*.png \
                    {self.out_figures}/oil_concentration_{self.config['simulation']['name']}.gif"
            ],
            # shell=True,
        )

    def plot_pyngl(
        self,
        plot_step: int = 1,
    ) -> None:
        """
        Plotting with pyngl.
        """
        config = self.config
        current_folder = os.path.dirname(os.path.abspath(__file__))
        path_to_plotspill = 'WITOIL_iMagine/src/plot/plotngl.py'
        root_directory = self.root_directory
        spill_lon = config["simulation"]["spill_lon"][0]
        spill_lat = config["simulation"]["spill_lat"][0]
        start_datetime = str(pd.Timestamp(config["simulation"]["start_datetime"]))
        start_datetime = start_datetime.replace(" ", "T")
        sim_length = config["simulation"]["sim_length"]
        plot_lon = config["plot_options"]["plot_lon"]
        plot_lat = config["plot_options"]["plot_lat"]
        subprocess.run(
            [
                f"python {path_to_plotspill} {root_directory} {plot_step} \
                                {spill_lon} {spill_lat} {start_datetime} \
                                {sim_length} {plot_lon[0]} {plot_lon[1]} \
                                {plot_lat[0]} {plot_lat[1]}"
            ],
            # shell=True,
            check=True,
        )


if __name__ == "__main__":
    pass
