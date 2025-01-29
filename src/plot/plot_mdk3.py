# Standard Library Imports
import xarray as xr
import glob
import json
import os
import subprocess
import sys
import warnings

# Third-Party Imports
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.interpolate import interp2d
from shapely.geometry import LineString, Point, Polygon, box
from datetime import datetime, timedelta
from matplotlib.lines import Line2D
from ..preprocessing.landextrap import LandExtrap

# Configure pandas and matplotlib options
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("agg")

# Relative Imports
try:
    from .curved_quivers import velovect
except (ImportError, ModuleNotFoundError):
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from src.plot.curved_quivers import velovect


class MedslikIIPlot:
    """
    This class embeds functions for plotting.
    """

    def __init__(
        self,
        config: dict,
        spill_lat: float,
        spill_lon: float,
        exp_directory: str,
        out_directory: str,
        out_figures: str,
    ) -> None:
        """
        Class constructor.
        """
        self.config = config
        self.spill_lat = spill_lat
        self.spill_lon = spill_lon
        self.exp_directory = exp_directory
        self.out_directory = out_directory
        self.bnc_directory = os.path.join(exp_directory, "bnc_files")
        self.out_figures = out_figures
        self.beaching_figures = os.path.join(out_figures, "beached_oil")
        os.makedirs(self.out_figures, exist_ok=True)
        os.makedirs(self.beaching_figures, exist_ok=True)
        self.concentration_path = os.path.join(
            self.out_directory, "oil_concentration.nc"
        )
        self.spill_properties_path = os.path.join(
            self.out_directory, "spill_properties.nc"
        )
        self.__envdata_path()

    def __envdata_path(self) -> None:
        """
        Set path to Environmental Data.
        """
        config = self.config
        # Winds data path (default)
        wind_path = os.path.join(self.exp_directory, "met_files", "*.nc")
        # Currents data path (default)
        curr_path = os.path.join(self.exp_directory, "oce_files", "*.nc")
        metoce = config["input_files"]["metoce"]
        if os.path.exists(metoce["met_data_path"]):
            wind_path = os.path.join(metoce["met_data_path"], "*.nc")
        if os.path.exists(metoce["oce_data_path"]):
            curr_path = os.path.join(metoce["oce_data_path"], "*.nc")
        self.wind_path = wind_path
        self.curr_path = curr_path

    def plot_matplotlib(
        self,
        plot_step: int = 1,
        crange: list[float] = None,
        plot_extrap_currents=False,
        plot_curr_method="curly",
    ):

        # Simulation initial and end dates
        inidate = pd.to_datetime(
            self.config["initial_conditions"]["start_datetime"]
        ) + pd.Timedelta(hours=1.0)
        enddate = pd.to_datetime(
            inidate
            + pd.Timedelta(hours=int(self.config["simulation_setup"]["sim_length"]))
        )

        ### Define plot boundaries ###
        lon_min, lon_max = self.config["plot_options"]["plot_lon"]
        lat_min, lat_max = self.config["plot_options"]["plot_lat"]

        # Read coastline
        land = gpd.read_file(self.config["input_files"]["dtm"]["coastline_path"])

        # Ensure CRS for coastline data
        if land.crs is None:
            print(
                "plotting -\
                Assigning EPSG:4326 CRS to the land dataset \
                (assuming geographic coordinates)."
            )
            land = land.set_crs("EPSG:4326")

        if land.crs != "EPSG:4326":
            land = land.to_crs("EPSG:4326")

        # Read output NetCDF for concentration
        ds_particles = xr.open_dataset(self.concentration_path)

        # Opening currents NetCDF
        curr = xr.open_mfdataset(self.curr_path)

        # Read wind data
        wind = xr.open_mfdataset(self.wind_path).transpose("time", "lon", "lat")

        # gravity center
        lon_gravity_center = ds_particles["lon_gravity_center"].values
        lat_gravity_center = ds_particles["lat_gravity_center"].values

        # Ensure date index is correct
        try:
            curr["time"] = curr.indexes["time"].to_datetimeindex()
            wind["time"] = wind.indexes["time"].to_datetimeindex()
        except (KeyError, AttributeError):
            pass

        # Resample current data to hourly intervals and interpolate
        curr = curr.resample(time="1h").interpolate("linear")

        # selecting simulation date
        curr = curr.sel(time=slice(inidate, enddate))

        # selecting surface current
        curr = curr.isel(depth=0)

        # Resample wind data to hourly intervals and interpolate
        wind = wind.resample(time="1h").interpolate("linear")

        # Select wind data within the simulation time period
        wind = wind.sel(time=slice(inidate, enddate))

        ### Subset the current data to the plot boundaries ###
        subset_curr = curr.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

        ### Calculate Maximum Current Magnitude for the Simulation ###
        current_magnitude = np.sqrt(subset_curr["uo"] ** 2 + subset_curr["vo"] ** 2)
        max_current_magnitude = current_magnitude.max().values

        """Uncomment to check on currents scale quiver
            If needed you can adjust the quiver in ax3.quiver (scale)"""
        # Get the corresponding time and location for the maximum current
        # max_index = np.unravel_index(np.nanargmax(current_magnitude.values), current_magnitude.shape)
        # max_lat = subset_curr.lat.values[max_index[1]]
        # max_lon = subset_curr.lon.values[max_index[2]]
        # max_time = subset_curr.time.values[max_index[0]]
        # print(
        #   f"plotting -\
        #     The maximum current value is {max_current_magnitude:.4f} m/s \
        #     Registered on the following time and position: {max_time}; Lat: {max_lat}, Lon: {max_lon}."
        # )

        ### Calculate Maximum Concentration for the Whole Dataset ###
        if crange is None:
            min_concentration = 0.0
            max_concentration = (ds_particles.concentration * 1000).max().values
            crange = [min_concentration, max_concentration]
        else:
            min_concentration = crange[0]
            max_concentration = crange[1]

        # Define discrete levels for the colorbar (to make it more distinct)
        levels = np.linspace(
            min_concentration, max_concentration, 11
        )  # Adjust number of levels as needed

        # Extend levels to add an extra section
        extended_levels = np.append(levels, crange[1] + 1)
        # Format the colorbar levels and ticks
        try:
            magnitude = int(
                np.floor(np.log10(abs(max_concentration)))
            )  # Order of magnitude
        except OverflowError:
            print("plotting - no significative beached oil")
            return
        if magnitude >= 2:
            n_digits = 0
            format_string = "%.0f"
        elif magnitude < 2:
            n_digits = (-magnitude) + 1
            format_string = "%." + str(n_digits) + "f"
        extended_levels = (extended_levels // 10 ** (magnitude - 1)) * 10 ** (
            magnitude - 1
        )
        # Clip the coastline data to the bounding box
        bounding_box = Polygon(
            [
                (lon_min, lat_min),
                (lon_min, lat_max),
                (lon_max, lat_max),
                (lon_max, lat_min),
            ]
        )
        rec = gpd.clip(land, bounding_box)

        # Loop through the time series for plotting
        for t in range(plot_step - 1, len(ds_particles.time), plot_step):

            # Specify the target latitude and longitude
            target_lon = lon_gravity_center[t]  # Replace with your target latitude
            target_lat = lat_gravity_center[t]  # Replace with your target longitude

            # From wind dataset find nearest lon/lat points to the gravity center
            nearest_lon = wind.lon.sel(lon=target_lon, method="nearest")
            nearest_lat = wind.lat.sel(lat=target_lat, method="nearest")

            # Using nearest available lat/lon
            subset_wind = wind.sel(lon=nearest_lon, lat=nearest_lat)

            # Make sure that lon/lat are preserved as dimensions
            subset_wind = subset_wind.expand_dims(["lat", "lon"])

            ### Calculate Maximum Wind Magnitude for the Simulation ###
            wind_magnitude = np.sqrt(
                subset_wind["U10M"] ** 2 + subset_wind["V10M"] ** 2
            )
            max_wind_magnitude = wind_magnitude.max().values

            """Uncomment to check on wind scale quiver
                If needed you can adjust the quiver in ax5.quiver (scale)"""
            # Get the corresponding time and location for the maximum wind
            # max_w_index = np.unravel_index(np.nanargmax(wind_magnitude.values), wind_magnitude.shape)
            # max_lon = target_lon
            # max_lat = target_lat
            # max_time = subset_wind.time.values[np.nanargmax(wind_magnitude.values)]
            # max_time = subset_curr.time.values[max_w_index[0]]
            # print(
            #     f"plotting -\
            #     The maximum wind value is {max_wind_magnitude:.4f} m/s \
            #     Registered on the following time and position: {max_time}; Lat: {max_lat}, Lon: {max_lon}."
            # )

            # Select the iteration timestep, ensure lat/lon dimensions are retained
            ds_p = ds_particles.isel(time=t)
            plot_curr = subset_curr.isel(time=t)
            plot_wind = subset_wind.isel(time=t)

            # Sea Over Land plotting
            if plot_extrap_currents:
                try:
                    extrap_method = self.config["input_files"]["sol_extrap_method"]
                    try:
                        sol_iterations = self.config["input_files"][
                            "sol_extrap_iterations"
                        ]
                        LandExtrap.iterations = sol_iterations
                    except KeyError:
                        sol_iterations = LandExtrap.iterations
                    if extrap_method == "gradient":
                        print(
                            f"WARNING: Extrapolated currents can not be plotted using '{extrap_method}' method."
                        )
                    else:
                        print(
                            f"Extrapolated currents will be plotted using method '{extrap_method}' and {sol_iterations} iterations."
                        )
                        plot_curr["uo"].values = LandExtrap.extrap3d(
                            plot_curr["uo"].values, method=extrap_method
                        )
                        plot_curr["vo"].values = LandExtrap.extrap3d(
                            plot_curr["vo"].values, method=extrap_method
                        )
                except KeyError:
                    plot_extrap_currents = False
                    print(
                        "WARNING: No Sea Over Land extrapolation method provided. \
                            Extrapolated currents will not be plotted."
                    )
            # Ensure that the release coordinates fall within the dataset bounds
            spill_lon = self.spill_lon
            spill_lat = self.spill_lat

            # Handle list case, extract first element if they are lists
            if isinstance(spill_lon, list):
                spill_lon = spill_lon[0]
            if isinstance(spill_lat, list):
                spill_lat = spill_lat[0]

            # Ensure that the release coordinates fall within the dataset bounds
            if not (
                lon_min <= spill_lon <= lon_max and lat_min <= spill_lat <= lat_max
            ):
                print(
                    f"plotting - Release point {spill_lon}, {spill_lat} is out of domain."
                )
                continue

            # Extract U10M and V10M values at the nearest position
            u_wind_raw = plot_wind.sel(lon=nearest_lon, lat=nearest_lat).U10M.values
            v_wind_raw = plot_wind.sel(lon=nearest_lon, lat=nearest_lat).V10M.values

            # Check if the data is NaN
            if np.isnan(u_wind_raw) or np.isnan(v_wind_raw):
                print(
                    f"plotting - NaN wind data found at timestep {t}. Skipping wind interpolation."
                )
                continue  # Skip interpolation if raw data is invalid

            # Create figure and GridSpec layout
            fig = plt.figure(figsize=(10, 8))
            gs = plt.GridSpec(
                5, 3, height_ratios=[20, 0.1, 0.8, 1, 2], width_ratios=[1, 1, 1]
            )

            # Big plot on top
            ax1 = fig.add_subplot(gs[0, :])
            # ax1.set_facecolor("#ADD8E6")
            ax1.set_facecolor("white")

            # Plot coastline using GeoPandas
            rec.plot(ax=ax1, color="#FFFDD0", edgecolor="black", zorder=1000, aspect=1)
            # rec.plot(ax=ax1, color="#efeeda", edgecolor="black", zorder=1000, aspect=1)
            # rec.plot(ax=ax1, color="white", edgecolor="black", zorder=1000, aspect=1)

            # Define the custom colormap --- PyNGL
            colors = [
                (0, 0, 1),  # Blue
                (0, 1, 1),  # Aqua
                (0, 1, 0),  # Green
                (1, 1, 0),  # Yellow
                (1, 0.5, 0),  # Orange
                (1, 0, 0),  # Red
                (0.5, 0, 0.5),  # Violet
            ]

            # Add final color (violet) for the extended section
            colors.append((0.5, 0, 0.5))  # Violet for the final section
            cmap_name = "custom_BlAqGrYeOrReVi"
            custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=200)

            # Plot oil concentration (conversion to tons/km^2)
            ds_c1 = xr.where(
                ds_p.concentration * 1000 > 0.001, ds_p.concentration * 1000, np.nan
            )
            # *****************************************************************
            # PLOT OBSERVATIONS
            # Read the .txt file using space as delimiter
            # df = pd.read_csv(
            #     "../../testcase/lebanon/observations/observation_0607230835.txt",
            #     delim_whitespace=True,
            #     header=None,
            #     names=["lat", "lon"],
            #     skiprows=3,
            # )
            # # Create geometry from lat/lon using Shapely's Point
            # geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
            # # Step 4: Create GeoDataFrame
            # gdf = gpd.GeoDataFrame(df, geometry=geometry)
            # # Step 5: Plot using GeoPandas
            # gdf.plot(
            #     ax=ax1,
            #     marker="o",
            #     color="gray",
            #     markersize=3,
            #     alpha=0.2,
            # )
            # # Create custom legend
            # legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Observations')]

            # # Add the legend
            # ax1.legend(handles=legend_elements, loc='upper left')
            # *****************************************************************

            c = ds_c1.plot(
                ax=ax1,
                levels=extended_levels,
                vmin=min_concentration,
                vmax=max_concentration,
                add_colorbar=False,
                cmap=custom_cmap,
                extend="max",
            )

            if plot_curr_method == "curly":
                """START - PLOTTING CURRENTS WITH CURLY VECTORS --- SOLUTION: https://github.com/Deltares/dfm_tools/blob/829e76f48ebc42460aae118cc190147a595a5f26/dfm_tools/modplot.py"""
                # regularly spaced grid spanning the domain of x and y
                xi = np.linspace(
                    plot_curr.lon.values.min(),
                    plot_curr.lon.values.max(),
                    plot_curr.lon.values.shape[0],
                )
                yi = np.linspace(
                    plot_curr.lat.values.min(),
                    plot_curr.lat.values.max(),
                    plot_curr.lat.values.shape[0],
                )

                # bicubic interpolation
                x, y = np.meshgrid(xi, yi, indexing="xy", sparse=False)

                # Vector components
                u = plot_curr.uo.values
                v = plot_curr.vo.values

                velovect(
                    ax1,
                    x,
                    y,
                    u,
                    v,
                    density=3,  # Adjust for density of vectors
                    linewidth=0.6,  # Adjust for vector line width
                    color="black",  # Single color or array for multicolor
                    arrowsize=0.6,  # Adjust arrow size
                    grains=16,
                    # integration_direction='forward',
                    broken_streamlines=True,  # Allow streamlines to break
                    zorder=1500,
                )
                """ END - PLOTTING CURRENTS WITH CURLY VECTORS"""
            elif plot_curr_method == "quiver":
                """START - PLOTTING CURRENTS WITH QUIVERS"""
                # Quiver plot adjustments for the big plot ###
                ax1.quiver(
                    plot_curr.lon.values,
                    plot_curr.lat.values,
                    plot_curr.uo.values,
                    plot_curr.vo.values,
                    scale=7,
                    # scale=5,                              # Adjusted scale for the current vectors
                    width=0.0025,  # Line thickness
                    headwidth=3,  # Adjust head width
                    headlength=4,  # Adjust head length
                    headaxislength=3.5,  # Length of the axis part of the arrow
                    color="black",
                    zorder=1500,
                )
                """END - PLOTTING CURRENTS WITH QUIVERS"""
            else:
                raise ValueError(
                    f"Plotting currents with method '{plot_curr_method}' is not supported. \
                                 Supported methods are 'curly', 'quiver'."
                )

            """START - PLOTTING CURRENTS WITH STREAMLINES"""
            # # regularly spaced grid spanning the domain of x and y
            # xi = np.linspace(
            #     plot_curr.lon.values.min(),
            #     plot_curr.lon.values.max(),
            #     plot_curr.lon.values.shape[0],
            # )
            # yi = np.linspace(
            #     plot_curr.lat.values.min(),
            #     plot_curr.lat.values.max(),
            #     plot_curr.lat.values.shape[0],
            # )

            # # bicubic interpolation
            # x, y = np.meshgrid(xi, yi, indexing="xy", sparse=False)

            # # plotting current vectors
            # ax1.streamplot(
            #     x,
            #     y,
            #     plot_curr.uo.values,
            #     plot_curr.vo.values,
            #     color="black",
            #     linewidth=0.5,
            #     arrowsize=1.5,
            # )
            """ END - PLOTTING CURRENTS WITH STREAMLINES"""

            # Plot the release point
            ax1.plot(spill_lon, spill_lat, marker="+", color="black", markersize=14)

            # Add wind vector (quiver) at the gravity center
            ax1.quiver(
                target_lon,  # Plot at center of gravity
                target_lat,  # Plot at center of gravity
                u_wind_raw,  # Use wind from gravity center
                v_wind_raw,  # Use wind from gravity center
                scale=50,  # Adjust the scale of the wind vector
                color="red",  # "#1f77b4"
                zorder=2000,
                width=0.003,  # Width of the arrow
                headwidth=3,
                headlength=4,
            )

            # Convert timedelta64[ns] to absolute datetime
            current_time = (
                inidate + pd.to_timedelta(ds_particles.time.values[t]) - pd.Timedelta(hours=1)
            ).strftime("%Y-%m-%d %H:%M")

            # Determine longitude and latitude labels based on boundaries
            lon_label = "Longitude (°E)" if lon_min >= 0 else "Longitude (°W)"
            lat_label = "Latitude (°N)" if lat_min >= 0 else "Latitude (°S)"

            # Set plot limits, labels, and title for the big plot
            # ax1.set_title(f"Oil Surface Concentration\n{t+1} hour(s) after oil release", fontsize=14, pad=20)
            ax1.set_title(
                f"Surface Oil Concentration\n{current_time}", fontsize=18, pad=20
            )
            ax1.set_xlabel(lon_label, fontsize=12)
            ax1.set_ylabel(lat_label, fontsize=12)
            ax1.set_xlim(lon_min, lon_max)
            ax1.set_ylim(lat_min, lat_max)
            plt.grid()

            # Blank space row (above colorbar)
            ax_blank1 = fig.add_subplot(gs[1, :])
            ax_blank1.set_visible(False)

            # Colorbar row
            cbar_ax = fig.add_subplot(gs[2, :])
            cbar = plt.colorbar(
                c, cax=cbar_ax, orientation="horizontal", ticks=extended_levels[:-1]
            )
            cbar.set_label(r"tons km$^{-2}$", fontsize=14)

            # Format the colorbar ticks
            cbar.ax.xaxis.set_major_formatter(FormatStrFormatter(format_string))

            # Blank space row (below colorbar)
            ax_blank2 = fig.add_subplot(gs[3, :])
            ax_blank2.set_visible(False)

            ### Left Small Plot (Quiver Current Scale) ###
            ax3 = fig.add_subplot(gs[4, 0])

            ### Quiver plot adjustments for currents scale plot ###
            ax3.quiver(
                0.05,
                0.5,
                max_current_magnitude,
                0,  # Shift quiver to the left using transformation
                # scale=3.5,
                scale=1.1,
                width=0.008,
                headwidth=4,
                headlength=5,
                headaxislength=5,
                color="black",
                transform=ax3.transAxes,  # Ensure the position is relative to the axis
            )

            # Adjust axis limits to ensure quiver is on the left
            ax3.set_xlim(
                0, 1
            )  # Use normalized axis limits to force the quiver to stay in a fixed position
            ax3.set_ylim(-0.5, 0.5)  # Keep vertical limit

            # Align the title and xlabel to the left using loc='left'
            ax3.set_title("Ocean Currents", loc="left", fontsize=13)
            ax3.set_xlabel(
                f"{max_current_magnitude:.1f} m s$^{{-1}}$", loc="left", fontsize=13
            )

            # Remove ticks and grid from the small plot
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_frame_on(False)

            # Bottom row with other subplots (middle one transparent)
            ax4 = fig.add_subplot(gs[4, 1])
            ax4.set_visible(False)

            # Wind Reference Quiver plot
            ax5 = fig.add_subplot(gs[4, 2])

            # Quiver plot adjustments for wind scale plot
            ax5.quiver(
                0.05,
                0.5,
                max_wind_magnitude,
                0,  # Shift quiver to the left using transformation
                # scale=20,
                scale=17,
                width=0.008,
                headwidth=4,
                headlength=5,
                headaxislength=5,
                color="red",
                transform=ax5.transAxes,  # Ensure the position is relative to the axis
            )

            # Adjust axis limits to ensure quiver is on the left
            ax5.set_xlim(
                0, 1
            )  # Use normalized axis limits to force the quiver to stay in a fixed position
            ax5.set_ylim(-0.5, 0.5)  # Keep vertical limit

            # Align the title and xlabel to the left using loc='left'
            ax5.set_title("Wind Vectors", loc="left", fontsize=13)
            ax5.set_xlabel(
                f"{max_wind_magnitude:.1f} m s$^{{-1}}$", loc="left", fontsize=13
            )

            # Remove ticks and grid from the small plot
            ax5.set_xticks([])
            ax5.set_yticks([])
            ax5.set_frame_on(False)

            # Save the figure for each time step
            plt.savefig(
                self.out_figures
                + f"/surf_oil_concentration_{self.config['simulation_setup']['name']}_{t+1:03d}.png",
                dpi=200,
            )

            plt.close()

    def plot_mass_balance(self):

        path = self.out_directory
        filename = path + "/oil_fate.txt"

        header = pd.read_csv(
            filename, sep=r"\s\s+", skiprows=6, engine="python"
        ).columns.values
        df = pd.read_csv(filename, sep=r" +", skiprows=7, header=None, engine="python")
        df.columns = header

        df = df[["time", "%srftot", "%evap", "%disp", "%cstfxd"]]
        df.index = df.time
        df.drop("time", axis=1, inplace=True)
        # "total oil on coast" is converted into "free oil on coast"
        # renaming columns
        df.columns = [
            "Oil on the Sea Surface",
            "Oil Evaporated",
            "Oil Dispersed in the Water Column",
            "Oil Fixed on Coast",
        ]

        styles = ["-", "-", "-", "-"]
        # plt.figure(figsize=(10, 10))
        df.plot(style=styles, legend=True)
        # Move the legend to the bottom center, below the x-axis
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize="small"
        )  # ncol=3 for multi-column legend
        # Adjust the layout to make space for the legend
        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
        )  # Adjust the bottom boundary for the legend
        # Labels
        plt.xlabel("Time (hrs)")
        plt.grid(True)
        plt.ylim(0, 100)
        plt.xlim(
            0,
        )
        plt.ylabel("Percentage (%)")
        plt.title("Mass Balance")
        plt.savefig(
            self.out_figures
            + f"/massbalance_{self.config['simulation_setup']['name']}.png",
            dpi=100,
            bbox_inches="tight",
        )

        plt.close()

    def create_gif(self):
        path = self.out_figures
        subprocess.run(
            [
                f"magick -delay 20 -loop 0 {path}/*surf_oil_*.png \
                    {self.out_figures}/oil_concentration_{self.config['simulation_setup']['name']}.gif"
            ],
            shell=True,
        )

    def plot_beached_oil(self, plot_step: int = 1):
        """
        Generate plots of beached oil for each time step and save impacted segments.
        """
        # Load the .nc dataset
        output = xr.open_dataset(self.spill_properties_path)

        # Oil density
        oil_density = output.non_evaporative_volume.oil_density  # Density in kg/m³

        # Initialize a list to store the data for each time step
        output_data = []

        # Step 1: Determine the maximum beached oil mass across all segments at the last time step
        rec = output.isel(time=-1)  # Select the last time step
        # Convert the xarray dataset to a pandas DataFrame
        df = rec.to_dataframe()
        # Drop duplicates based on the 'status' column.
        # Particles with same status have the same value of seeped oil.
        unique_status = df.drop_duplicates(subset="particle_status")
        # Convert back to an xarray dataset
        rec = unique_status.to_xarray()

        geom = []
        mass = []

        # Filter beached particles at the last time step and collect coordinates and mass in tons
        for p in range(len(rec.non_evaporative_volume)):
            if rec.particle_status[p].values < 0:
                volume_m3 = rec.isel(parcel_id=p).seeped_vol.values  # Volume in m³
                mass_tons = (volume_m3 * oil_density) / 1000  # Convert to tons

                geom.append(
                    Point(
                        rec.isel(parcel_id=p).longitude.values,
                        rec.isel(parcel_id=p).latitude.values,
                    )
                )
                mass.append(mass_tons)

        # If no beached particles are found, print a message and exit the method
        if not mass:
            print("plotting - No beached oil detected.")
            return

        # Create GeoDataFrame with beached particles and their mass in tons at the last time step
        shp = gpd.GeoDataFrame(data=mass, columns=["oil_mass"], geometry=geom, crs=4326)

        # Read and prepare the coastline
        coastline = gpd.read_file(self.config["input_files"]["dtm"]["coastline_path"])
        xmin, xmax = self.config["plot_options"]["plot_lon"]
        ymin, ymax = self.config["plot_options"]["plot_lat"]

        # Crop and split the coastline into segments
        lines = coastline.cx[xmin:xmax, ymin:ymax]
        lines["geometry"] = lines.geometry.boundary
        lines = lines.clip_by_rect(xmin, ymin, xmax, ymax)
        lines = lines[~lines.is_empty].explode(index_parts=True)

        # Separate each LineString into individual segments
        segments = []
        for line in lines.geometry:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                segment = LineString([coords[i], coords[i + 1]])
                segments.append(segment)

        # Create a GeoDataFrame with individual coastline segments
        lines = gpd.GeoDataFrame(geometry=segments, crs="EPSG:4326").reset_index()
        lines["id"] = lines.index

        # Determine the UTM zone based on the centroid of the area
        centroid_lon = (xmin + xmax) / 2
        centroid_lat = (ymin + ymax) / 2
        utm_zone = int((centroid_lon + 180) / 6) + 1

        # Use 32600 series for the northern hemisphere and 32700 series for the southern hemisphere
        utm_crs = CRS.from_epsg(
            32600 + utm_zone if centroid_lat >= 0 else 32700 + utm_zone
        )

        # Reproject coastline segments to UTM
        lines_utm = lines.to_crs(utm_crs)

        # Calculate the length of each segment in kilometers
        lines_utm["length_km"] = lines_utm.length / 1000  # Convert meters to kilometers
        lines_utm = lines_utm.reset_index(drop=True)
        lines_utm["id"] = lines_utm.index

        # Reproject beached particles to UTM for distance calculations
        shp_utm = shp.to_crs(utm_crs)

        # Associate particles with the closest coastline segment
        join = gpd.sjoin_nearest(shp_utm, lines_utm, how="left")

        # Aggregate oil mass per segment in tons
        try:
            oil_mass_per_segment = (
                join.groupby("index_right")["oil_mass"].sum().reset_index()
            )
        except KeyError:
            oil_mass_per_segment = (
                join.groupby("index_right0")["oil_mass"].sum().reset_index()
            )
        oil_mass_per_segment.columns = ["segment_id", "total_oil_mass"]

        # Join oil mass data back with the UTM coastline segments to calculate tons per km
        segments_with_mass = lines_utm.merge(
            oil_mass_per_segment, left_on="id", right_on="segment_id"
        )
        segments_with_mass["mass_per_km"] = (
            segments_with_mass["total_oil_mass"] / segments_with_mass["length_km"]
        )

        # Determine the maximum mass per km to use for colorbar levels
        max_mass_per_km = segments_with_mass["mass_per_km"].max() * 2 / 3
        levels = np.linspace(
            0, max_mass_per_km, 10
        )  # Define discrete levels based on max mass per km
        # Format the colorbar ticks
        try:
            magnitude = int(
                np.floor(np.log10(abs(max_mass_per_km)))
            )  # Order of magnitude
        except OverflowError:
            print("plotting - no significative beached oil")
            return
        if magnitude >= 2:
            n_digits = 0
            format_string = "%.0f"
        elif magnitude <= 0:
            n_digits = (-magnitude) + 2
            format_string = "%." + str(n_digits) + "f"
        else:
            format_string = "%.1f"
        levels = (levels // 10 ** (magnitude - 1)) * 10 ** (magnitude - 1)
        # Step 2: Loop through each time step to plot
        for t in range(plot_step - 1, len(output.time), plot_step):
            # Select data for the current time step
            rec = output.isel(time=t)
            # Convert the xarray dataset to a pandas DataFrame
            df = rec.to_dataframe()
            # Drop duplicates based on the 'status' column.
            # Particles with same status have the same value of seeped oil.
            unique_status = df.drop_duplicates(subset="particle_status")
            # Convert back to an xarray dataset
            rec = unique_status.to_xarray()

            # Lists to store geometry and oil mass of beached particles
            geom = []
            mass = []

            # Filter beached particles and collect coordinates and mass in tons
            for p in range(len(rec.non_evaporative_volume)):
                if rec.particle_status[p].values < 0:
                    volume_m3 = rec.isel(parcel_id=p).seeped_vol.values
                    mass_tons = (volume_m3 * oil_density) / 1000  # Convert to tons

                    geom.append(
                        Point(
                            rec.isel(parcel_id=p).longitude.values,
                            rec.isel(parcel_id=p).latitude.values,
                        )
                    )
                    mass.append(mass_tons)

            # Create GeoDataFrame with beached particles and their mass in tons
            shp = gpd.GeoDataFrame(
                data=mass, columns=["oil_mass"], geometry=geom, crs="EPSG:4326"
            )
            shp_utm = shp.to_crs(utm_crs)

            # Skip to the next time step if there are no beached particles
            if shp.empty:
                print(
                    f"plotting - No beached particles at time step {t+1}, skipping plot."
                )
                continue

            # Associate particles with closest coastline segment
            join = gpd.sjoin_nearest(shp_utm, lines_utm, how="left")

            # Aggregate oil mass per segment
            try:
                oil_mass_per_segment = (
                    join.groupby("index_right")["oil_mass"].sum().reset_index()
                )
            except KeyError:
                oil_mass_per_segment = (
                    join.groupby("index_right0")["oil_mass"].sum().reset_index()
                )
            oil_mass_per_segment.columns = ["segment_id", "total_oil_mass"]

            # Merge with the UTM coastline segments to calculate tons per km for plotting
            segments_with_mass = lines_utm.merge(
                oil_mass_per_segment, left_on="id", right_on="segment_id", how="left"
            )
            segments_with_mass["mass_per_km"] = (
                segments_with_mass["total_oil_mass"] / segments_with_mass["length_km"]
            )
            segments_with_mass = segments_with_mass.to_crs("EPSG:4326")
            # Extract the information for each segment at the current time step
            for _, segment in segments_with_mass.dropna(
                subset=["mass_per_km"]
            ).iterrows():
                output_data.append(
                    {
                        "time_step": t + 1,
                        "segment_id": segment["segment_id"],
                        "segment_length_km": segment["length_km"],
                        "oil_concentration_tons_per_km": segment["mass_per_km"],
                        "initial_latitude": segment.geometry.coords[0][1],
                        "initial_longitude": segment.geometry.coords[0][0],
                        "final_latitude": segment.geometry.coords[-1][1],
                        "final_longitude": segment.geometry.coords[-1][0],
                    }
                )

            # Set up the figure and simplified GridSpec layout
            fig = plt.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(3, 1, height_ratios=[16, 1, 1])

            # Main plot area
            ax_main = fig.add_subplot(gs[0, 0])

            # Crop and plot the coastline within the area of interest
            coastline = coastline.cx[xmin:xmax, ymin:ymax]
            coastline.plot(
                ax=ax_main, color="lightgray", edgecolor="black", label="Coastline"
            )

            # Define colormap and norm for colorbar based on calculated levels
            cmap = LinearSegmentedColormap.from_list(
                "custom_colormap", ["green", "yellow", "red", "purple"]
            )
            norm = BoundaryNorm(levels, cmap.N)

            # Plot impacted segments with color representing mass per km
            segments_with_mass = segments_with_mass.dropna(subset=["mass_per_km"])
            shp_concentration = gpd.sjoin_nearest(
                shp, segments_with_mass, how="left", distance_col="distance"
            )
            shp_concentration.plot(
                ax=ax_main,
                column="mass_per_km",
                cmap=cmap,
                markersize=10,
                legend=False,
                norm=norm,
                label="Beached Oil",
            )

            # Set plot boundaries and labels
            ax_main.set_xlim([xmin, xmax])
            ax_main.set_ylim([ymin, ymax])
            ax_main.set_xlabel("Longitude (°E)")
            ax_main.set_ylabel("Latitude (°N)")
            ax_main.set_title(
                f"Beached Oil Fixed on the Coast\n{t+1} hour(s) after oil release"
            )

            # Add a grid for better readability
            ax_main.grid(True)

            # Invisible spacer row below the main plot
            ax_blank = fig.add_subplot(gs[1, 0])
            ax_blank.set_visible(False)

            # Colorbar row
            cbar_ax = fig.add_subplot(gs[2, 0])
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []  # Dummy array for the colorbar
            cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="max")
            cbar.set_label(r"Concentration (tons km$^{-1}$)", fontsize=10)
            # Format the colorbar ticks to show 4 decimal places
            cbar.ax.xaxis.set_major_formatter(FormatStrFormatter(format_string))

            # Save plot with specific naming convention
            plt.savefig(
                os.path.join(
                    self.beaching_figures,
                    f"beached_oil_concentration_{t+1:03d}.png",
                ),
                dpi=200,
            )
            plt.close(fig)

        # Convert to DataFrame and save as CSV after all time steps are processed
        df_segments = pd.DataFrame(output_data)
        df_segments.to_csv(
            os.path.join(
                self.out_directory,
                f"beached_oil_fixed.csv",
            ),
            index=False,
        )

    def create_beached_oil_gif(self):
        """
        Creates a GIF from the beached oil plots saved as PNG files.
        """
        path = self.beaching_figures
        gif_name = f"beached_oil_{self.config['simulation_setup']['name']}.gif"
        output_gif_path = os.path.join(self.beaching_figures, gif_name)

        # Check if any beached oil images exist
        image_files = glob.glob(f"{path}/beached_oil_concentration_*.png")
        if not image_files:
            # No images found, exit the method quietly
            return

        # Use ImageMagick to create the GIF from beached oil images
        subprocess.run(
            [
                f"magick -delay 20 -loop 0 {path}/beached_oil_concentration_*.png {output_gif_path}"
            ],
            shell=True,
        )

    def plot_oil_difference(
        self,
        path_to_comparison: str,
        plot_step: int = 1,
    ):

        # Read output NetCDF for concentration
        ds_particles_1 = xr.open_dataset(self.concentration_path)
        ds_particles_2 = xr.open_dataset(path_to_comparison)

        # Load datasets
        ds1 = ds_particles_1.load()
        ds2 = ds_particles_2.load()

        # Check if lat and lon coordinates differ
        if not (
            np.array_equal(ds1.lat.values, ds2.lat.values)
            and np.array_equal(ds1.lon.values, ds2.lon.values)
        ):
            print(
                "Plotting - Plot oil difference: Lat/Lon grids are different. Interpolating ds2 to match ds1 grid."
            )
            # Interpolate ds2 onto the grid of ds1
            ds2_interp = ds2.interp(lat=ds1.lat, lon=ds1.lon)
        else:
            print(
                "Plotting - Plot oil difference: Lat/Lon grids are identical. No interpolation needed."
            )
            ds2_interp = ds2  # No interpolation required

        # Determine the shared time steps
        common_times = np.intersect1d(ds1["time"].values, ds2_interp["time"].values)

        # Subset datasets to the common time steps
        ds1 = ds1.sel(time=common_times)
        ds2_interp = ds2_interp.sel(time=common_times)

        # Define grid cell area in square meters (dynamic based on user config)
        grid_step = self.config["simulation_setup"]["oil_tracer_grid_step"]
        grid_cell_area = grid_step * grid_step  # Dynamic grid area in m²

        # Calculate the mass of oil per grid cell for both datasets
        # Multiply the concentration variable by the grid cell area
        ds1["mass"] = ds1["concentration"] * grid_cell_area  # Mass in kg
        ds2_interp["mass"] = ds2_interp["concentration"] * grid_cell_area  # Mass in kg

        # Convert to tons for easier comparison
        ds1["mass_in_tons"] = ds1["mass"] / 1000  # Convert kg to tons
        ds2_interp["mass_in_tons"] = ds2_interp["mass"] / 1000  # Convert kg to tons

        # Sum mass over all grid cells (lat and lon) for each time step
        ds1_total_mass = ds1["mass_in_tons"].sum(
            dim=["lat", "lon"]
        )  # Total mass per time step (kg)
        ds2_total_mass = ds2_interp["mass_in_tons"].sum(
            dim=["lat", "lon"]
        )  # Total mass per time step (kg)

        # Calculate spilled mass for each time step
        spill_rate = self.config["initial_conditions"][
            "spill_rate"
        ]  # Spill rate in tons/hour
        sim_length = self.config["simulation_setup"]["sim_length"]
        spill_duration = self.config["initial_conditions"]["spill_duration"]
        spill_hours = min(
            sim_length, spill_duration
        )  # Effective spill duration in hours
        spill_hours = int(spill_hours)

        # Extract time steps from ds1
        time_steps = ds1["time"].values

        # Calculate spilled mass for the effective spill duration
        spilled_mass = [spill_rate * (i + 1) for i in range(int(spill_hours))]

        # Pad spilled_mass to match the length of time_steps
        if len(time_steps) > spill_hours:
            spilled_mass.extend([None] * (len(time_steps) - int(spill_hours)))

        # Create a DataFrame for comparison
        data = {
            "time_step": time_steps,
            "ds1_total_mass_tons": ds1_total_mass.values,
            "ds2_total_mass_tons": ds2_total_mass.values,
            "spilled_mass_tons": spilled_mass,
        }

        df = pd.DataFrame(data)

        # Save DataFrame to CSV
        df.to_csv(self.out_figures + f"/oil_mass_comparison.csv", index=False)

        ### PLOT TOTAL OIL MASS DIFFERENCE

        plt.figure(figsize=(10, 6))

        # # Convert time_steps to hours as integers or strings
        time_steps = [t / np.timedelta64(1, "h") for t in ds1["time"].values]

        # Plot each column as a line
        plt.plot(
            time_steps,
            df["ds1_total_mass_tons"],
            label="V3.0",
            linestyle="-",
            marker="o",
        )
        plt.plot(
            time_steps,
            df["ds2_total_mass_tons"],
            label="V2.01",
            linestyle="-",
            marker="x",
        )
        plt.plot(
            time_steps,
            df["spilled_mass_tons"],
            label="'Gross' spilled mass",
            linestyle="--",
            marker="^",
        )

        # Add title and labels
        plt.title("Oil Mass Difference", fontsize=16)
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Mass (tons)", fontsize=12)

        # Add legend
        plt.legend(fontsize=12)

        plt.xticks(time_steps[::plot_step])

        # Add grid for better readability
        plt.grid(visible=True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plt.savefig(
            self.out_figures + f"/oil_mass_comparison.png",
            dpi=300,
        )
        plt.close()

        """OIL DIFFERENCE MASS PER GRID CELL
        REMEMBER TO ACTIVATE DIFFERENCES ON PLOT (LEGENDS)"""

        # Compute the absolute difference
        diff_concentration = abs(ds1.concentration - ds2_interp.concentration)

        diff_concentration = diff_concentration.where(diff_concentration != 0)

        # Locate the maximum difference across all timesteps
        max_difference = diff_concentration.max().values

        # Define colorbar range
        min_concentration = 0.0
        max_concentration = max_difference
        levels = np.linspace(min_concentration, max_concentration, 10)

        """END - OIL DIFFERENCE MASS PER GRID CELL"""

        """OIL DIFFERENCE MASS PERCENTAGE PER GRID CELL
        REMEMBER TO ACTIVATE DIFFERENCES ON PLOT (LEGENDS)"""
        # Compute the percentage difference relative to ds1.concentration and prevent errors from divisions by 0
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     diff_concentration = (abs(ds1.mass - ds2_interp.mass) / ds1.mass) * 100

        # # Mask division by zero (avoid NaN and Inf in colorbar limits)
        # diff_concentration = diff_concentration.where(np.isfinite(diff_concentration))

        # # Mask areas where the difference is 0
        # diff_concentration = diff_concentration.where(diff_concentration != 0)

        # # Locate the maximum percentage difference across all timesteps
        # max_difference = diff_concentration.max().values

        # # Define colorbar range
        # min_concentration = 0.0
        # max_concentration = max_difference
        # levels = np.linspace(min_concentration, max_concentration, 10)

        """END - OIL DIFFERENCE MASS PERCENTAGE PER GRID CELL"""

        # Simulation initial and end dates
        inidate = pd.to_datetime(
            self.config["initial_conditions"]["start_datetime"]
        ) + pd.Timedelta(hours=1.0)
        enddate = pd.to_datetime(
            inidate + pd.Timedelta(hours=self.config["simulation_setup"]["sim_length"])
        )

        ### Define plot boundaries ###
        lon_min, lon_max = self.config["plot_options"]["plot_lon"]
        lat_min, lat_max = self.config["plot_options"]["plot_lat"]

        # Read coastline
        land = gpd.read_file(self.config["input_files"]["dtm"]["coastline_path"])

        # Ensure CRS for coastline data
        if land.crs is None:
            print(
                "plotting -\
                  Assigning EPSG:4326 CRS to the land dataset \
                  (assuming geographic coordinates)."
            )
            land = land.set_crs("EPSG:4326")

        if land.crs != "EPSG:4326":
            land = land.to_crs("EPSG:4326")

        # Opening currents NetCDF
        curr = xr.open_mfdataset(self.curr_path)

        # Ensure date index is correct
        try:
            curr["time"] = curr.indexes["time"].to_datetimeindex()
        except (KeyError, AttributeError):
            pass

        # Resample current data to hourly intervals and interpolate
        curr = curr.resample(time="1h").interpolate("linear")

        # selecting simulation date
        curr = curr.sel(time=slice(inidate, enddate))

        # selecting surface current
        curr = curr.isel(depth=0)

        ### Subset the current data to the plot boundaries ###
        subset_curr = curr.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

        # Clip the coastline data to the bounding box
        bounding_box = Polygon(
            [
                (lon_min, lat_min),
                (lon_min, lat_max),
                (lon_max, lat_max),
                (lon_max, lat_min),
            ]
        )
        rec = gpd.clip(land, bounding_box)

        # Loop through the time series for plotting
        for t in range(plot_step - 1, len(diff_concentration.time), plot_step):

            # Select the iteration timestep
            ds_p = diff_concentration.isel(time=t)
            plot_curr = subset_curr.isel(time=t)

            # Ensure that the release coordinates fall within the dataset bounds
            spill_lon = self.spill_lon
            spill_lat = self.spill_lat

            # Handle list case, extract first element if they are lists
            if isinstance(spill_lon, list):
                spill_lon = spill_lon[0]
            if isinstance(spill_lat, list):
                spill_lat = spill_lat[0]

            # Ensure that the release coordinates fall within the dataset bounds
            if not (
                lon_min <= spill_lon <= lon_max and lat_min <= spill_lat <= lat_max
            ):
                print(
                    f"plotting -Release point {spill_lon}, {spill_lat} is out of bounds."
                )
                continue

            # Create figure and GridSpec layout
            fig = plt.figure(figsize=(10, 8))
            gs = plt.GridSpec(
                5, 3, height_ratios=[16, 1, 1, 1, 2], width_ratios=[1, 1, 1]
            )

            # Big plot on top
            ax1 = fig.add_subplot(gs[0, :])
            # ax1.set_facecolor("#ADD8E6")
            ax1.set_facecolor("white")

            # Plot coastline using GeoPandas
            rec.plot(ax=ax1, color="#FFFDD0", edgecolor="black", zorder=1000, aspect=1)
            # rec.plot(ax=ax1, color="#efeeda", edgecolor="black", zorder=1000, aspect=1)
            # rec.plot(ax=ax1, color="white", edgecolor="black", zorder=1000, aspect=1)

            # Plot oil concentration
            c = ds_p.plot(
                ax=ax1,
                levels=levels,
                vmin=min_concentration,
                vmax=max_concentration,
                add_colorbar=False,
            )

            """START - PLOTTING CURRENTS WITH STREAMLINES"""
            # regularly spaced grid spanning the domain of x and y
            xi = np.linspace(
                plot_curr.lon.values.min(),
                plot_curr.lon.values.max(),
                plot_curr.lon.values.shape[0],
            )
            yi = np.linspace(
                plot_curr.lat.values.min(),
                plot_curr.lat.values.max(),
                plot_curr.lat.values.shape[0],
            )

            # bicubic interpolation
            x, y = np.meshgrid(xi, yi, indexing="xy", sparse=False)

            # plotting current vectors
            ax1.streamplot(
                x,
                y,
                plot_curr.uo.values,
                plot_curr.vo.values,
                color="black",
                linewidth=0.5,
                arrowsize=1.5,
            )
            """ END - PLOTTING CURRENTS WITH STREAMLINES"""

            # Plot the release point
            ax1.plot(
                spill_lon,
                spill_lat,
                marker="x",
                color="black",
            )

            # Set plot limits, labels, and title for the big plot
            ax1.set_title(
                f"Oil Surface Concentration Difference\n{t+1} hour(s) after oil release",
                fontsize=14,
                pad=20,
            )
            ax1.set_xlabel("Longitude (°E)", fontsize=10)
            ax1.set_ylabel("Latitude (°N)", fontsize=10)
            ax1.set_xlim(lon_min, lon_max)
            ax1.set_ylim(lat_min, lat_max)
            plt.grid()

            # Blank space row (above colorbar)
            ax_blank1 = fig.add_subplot(gs[1, :])
            ax_blank1.set_visible(False)

            # Colorbar row
            cbar_ax = fig.add_subplot(gs[2, :])
            cbar = plt.colorbar(c, cax=cbar_ax, orientation="horizontal")

            """LEGEND CHANGE - CONCENTRATION OR PERCENTAGE"""
            ### Colobar Legend
            cbar.set_label(
                r"Concentration Difference (tons)", fontsize=10
            )  ## Concentration
            # cbar.set_label(r"Concentration Difference (%)", fontsize=10)  ## Percentage
            ### Format the colorbar ticks to show only 1 decimal place
            cbar.ax.xaxis.set_major_formatter(
                FormatStrFormatter("%.3f")
            )  ## Concentration
            # cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f')) ## Percentage

            # Blank space row (below colorbar)
            ax_blank2 = fig.add_subplot(gs[3, :])
            ax_blank2.set_visible(False)

            ### Left Small Plot (Quiver Current Scale) ###
            ax3 = fig.add_subplot(gs[4, 0])

            # Remove all visual elements from the axis
            ax3.set_xlim(0, 1)
            ax3.set_ylim(-0.5, 0.5)

            # Remove title and xlabel
            ax3.set_title("")
            ax3.set_xlabel("")

            # Remove ticks and grid
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.grid(False)

            # Hide the frame
            ax3.set_frame_on(False)
            ax3.set_facecolor("white")

            # Bottom row with other subplots (middle one transparent)
            ax4 = fig.add_subplot(gs[4, 1])
            ax4.set_visible(False)

            # Wind Reference Quiver plot
            ax5 = fig.add_subplot(gs[4, 2])

            # Remove all visual elements from the axis
            ax5.set_xlim(0, 1)
            ax5.set_ylim(-0.5, 0.5)

            # Remove title and xlabel
            ax5.set_title("")
            ax5.set_xlabel("")

            # Remove ticks and grid
            ax5.set_xticks([])
            ax5.set_yticks([])
            ax5.grid(False)

            # Hide the frame
            ax5.set_frame_on(False)
            ax5.set_facecolor("white")

            # Save the figure for each time step
            plt.savefig(
                self.out_figures
                + f"/surf_oil_concentration_diff_{self.config['simulation_setup']['name']}_{t+1:03d}.png",
                dpi=200,
            )

            plt.close()

    def create_surf_oil_concentration_diff_gif(self):
        """
        Creates a GIF from the surface oil concentration difference plots saved as PNG files.
        """
        path = self.out_figures
        gif_name = (
            f"surf_oil_concentration_diff_{self.config['simulation_setup']['name']}.gif"
        )
        output_gif_path = os.path.join(self.out_figures, gif_name)

        # Check if any surface oil concentration difference images exist
        image_files = glob.glob(
            f"{path}/surf_oil_concentration_diff_{self.config['simulation_setup']['name']}_*.png"
        )
        if not image_files:
            # No images found, exit the method quietly
            return

        # Use ImageMagick to create the GIF from the surface oil concentration difference images
        subprocess.run(
            [
                f"magick -delay 35 -loop 0 {path}/surf_oil_concentration_diff_{self.config['simulation_setup']['name']}_*.png {output_gif_path}"
            ],
            shell=True,
        )

    def plot_pyngl(
        self,
        plot_step: int = 1,
        crange: list[float] = None,
    ) -> None:
        """
        Plotting with pyngl.
        """
        config = self.config
        current_folder = os.path.dirname(os.path.abspath(__file__))
        path_to_plotspill = os.path.join(current_folder, "plotngl.py")
        exp_directory = self.exp_directory
        spill_lon = self.spill_lon
        spill_lat = self.spill_lat
        start_datetime = str(
            pd.Timestamp(config["initial_conditions"]["start_datetime"])
        )
        start_datetime = start_datetime.replace(" ", "T")
        sim_length = config["simulation_setup"]["sim_length"]
        plot_lon = config["plot_options"]["plot_lon"]
        plot_lat = config["plot_options"]["plot_lat"]
        curr_path = self.curr_path.replace("*.nc", "")
        wind_path = self.wind_path.replace("*.nc", "")
        if crange is None:
            crange = []
            crange.append(0.0)
            crange.append(0.0)
        subprocess.run(
            [
                f"python {path_to_plotspill} {exp_directory} {plot_step} \
                                {spill_lon} {spill_lat} {start_datetime} \
                                {sim_length} {plot_lon[0]} {plot_lon[1]} \
                                {plot_lat[0]} {plot_lat[1]} {crange[0]} {crange[1]} \
                                {curr_path} {wind_path}"
            ],
            shell=True,
            check=True,
        )


if __name__ == "__main__":
    try:
        from ..utils.config import Config
    except (ImportError, ModuleNotFoundError):
        sys.path.append(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        from src.utils.config import Config
    # variables
    config_path = (
        "/Users/francesco/shared/Medslik-II/testcase/lebanon/config_lebanon.toml"
    )
    path_to_comparison = "/Users/francesco/shared/MEDSLIK/MEDSLIK_FIXED_SEED_COMPARISON_v201_VS_v300/lebanon/v201/oil_concentration_150m.nc"
    plot_step = 24
    # SETUP
    config_dict = Config(config_path).config_dict
    spill_lat = (
        config_dict["initial_conditions"]["spill_lat_deg"]
        + config_dict["initial_conditions"]["spill_lat_min"] / 60.0
    )
    spill_lon = (
        config_dict["initial_conditions"]["spill_lon_deg"]
        + config_dict["initial_conditions"]["spill_lon_min"] / 60.0
    )
    exp_directory = os.path.join(
        config_dict["simulation_setup"]["experiment_path"],
        config_dict["simulation_setup"]["name"],
    )
    out_directory = os.path.join(exp_directory, "out_files")
    out_figures = os.path.join(out_directory, "figures_comparison")
    # Oil difference plot
    mplot = MedslikIIPlot(
        config_dict, spill_lat, spill_lon, exp_directory, out_directory, out_figures
    )
    mplot.plot_oil_difference(path_to_comparison, plot_step=plot_step)
    try:
        mplot.create_surf_oil_concentration_diff_gif()
    except Exception as e:
        print(f"An error occurred while creating the concentration difference GIF: {e}")
        pass
