import os, sys
import math
import warnings
import numpy as np
import xarray as xr
import pandas as pd
from numpy.typing import NDArray
from datetime import datetime, timedelta
from xarray import DataArray
import Ngl


class PlotNgl:
    """
    This class is for plotting Medslik-II simulation result using NCAR graphic library.
    """

    def __init__(
        self,
        outpath: str,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        concentration_range: list[float] = None,
    ) -> None:
        """
        Class constructor.
        """
        self.outpath = outpath
        # Check if path exists, otherwise create folder
        plot_folder, plot_name = os.path.split(outpath)
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        self.boundaries = [lon_min, lon_max, lat_min, lat_max]
        self.concentration_range = concentration_range

    def __call__(
        self,
        lon: NDArray,
        lat: NDArray,
        surf_concentration: NDArray,
        currents: list[DataArray] = None,
        winds: list[NDArray] = None,
        spill_time: datetime = None,
        plot_title: str = "Surface Oil Concentration",
        plot_step: int = 1,
        release_coords: list[float] = None,
    ) -> None:
        """
        Call functions for plotting results.
        """
        self.release_coords = release_coords
        if currents is not None:
            self.lon_curr = currents[0].lon.values
            self.lat_curr = currents[0].lat.values
            u_curr, v_curr = currents[0].values, currents[1].values
            u_curr[np.where(np.isnan(u_curr))] = 0.0
            v_curr[np.where(np.isnan(v_curr))] = 0.0
        else:
            self.lon_curr = lon
            self.lat_curr = lat
            u_curr = np.zeros_like(surf_concentration) * np.nan
            v_curr = np.zeros_like(surf_concentration) * np.nan
        if winds is not None:
            u_wind, v_wind = winds
            u_wind[np.where(np.isnan(u_wind))] = 0.0
            v_wind[np.where(np.isnan(v_wind))] = 0.0
        else:
            u_wind = np.zeros_like(surf_concentration) * np.nan
            v_wind = np.zeros_like(surf_concentration) * np.nan
        if self.concentration_range is not None:
            self.min_concentration = self.concentration_range[0]
            self.max_concentration = self.concentration_range[1]
        else:
            self.min_concentration = np.nanmin(surf_concentration)
            self.max_concentration = np.nanmax(surf_concentration)
        # Loop for plotting
        if len(surf_concentration.shape) == 3:
            n_tsteps = surf_concentration.shape[0]
            for time_index in range(n_tsteps):
                if spill_time is None:
                    time = None
                else:
                    dt = timedelta(hours=time_index * plot_step)
                    time = spill_time + dt
                self.make_plot(
                    f"{self.outpath}_{((time_index+1) * plot_step):02d}",
                    lon,
                    lat,
                    surf_concentration[time_index].T,
                    u_curr[time_index].T,
                    v_curr[time_index].T,
                    u_wind[time_index].T,
                    v_wind[time_index].T,
                    time=time,
                    plot_title=plot_title,
                )
        elif len(surf_concentration.shape) == 2:
            self.make_plot(
                f"{self.outpath}",
                lon,
                lat,
                surf_concentration.T,
                u_curr.T,
                v_curr.T,
                u_wind.T,
                v_wind.T,
                time=spill_time,
                plot_title=plot_title,
            )
        else:
            raise ValueError("Concentration array has invalid number of dimensions.")

    def make_plot(
        self,
        file_out_name: str,
        oil_lon: NDArray,
        oil_lat: NDArray,
        concentration: NDArray,
        u_curr: NDArray,
        v_curr: NDArray,
        u_wind: NDArray,
        v_wind: NDArray,
        time: datetime = None,
        plot_title: str = "Surface Oil Concentration",
        plot_wind: bool = True,
    ) -> None:
        """
        Plot concentration and environmental fields using ngl library.
        """
        # Open the WorKStation and set white colormap for contour fill
        rlist = Ngl.Resources()
        rlist.wkColorMap = "BlAqGrYeOrReVi200"  # cmap
        wks_type = "png"
        wks = Ngl.open_wks(wks_type, file_out_name, rlist)
        cmap = Ngl.retrieve_colormap(wks)
        Ngl.destroy(wks)
        if len(cmap) > 2:
            cmap[0] = [1.0, 1.0, 1.0]  # White
            cmap[1] = [1.0, 1.0, 1.0]  # White
            cmap[2] = [1.0, 1.0, 1.0]  # White
        rlist.wkColorMap = cmap
        Ngl.set_values(wks, rlist)
        wks = Ngl.open_wks(wks_type, file_out_name, rlist)
        # Set up resources and Draw vector fields and contours
        # Slick
        slick_rsrc = self.__slick_resources(oil_lon, oil_lat)
        contour_slick = Ngl.contour(wks, concentration, slick_rsrc)
        # Currents
        try:
            lon_oce = self.lon_oce
            lat_oce = self.lat_oce
            currents_rsrc = self.__current_resources(lon_oce, lat_oce)
        except AttributeError:
            currents_rsrc = self.__current_resources(oil_lon, oil_lat)
        vector_curr = Ngl.vector(wks, u_curr, v_curr, currents_rsrc)
        # Winds
        wind_rsrc = self.__wind_resources(oil_lon, oil_lat)
        vector_wind = Ngl.vector(wks, u_wind, v_wind, wind_rsrc)
        # MAP SETUP
        # Define map boundaries
        if self.boundaries is not None:
            lon_start, lon_end, lat_start, lat_end = self.boundaries
        else:
            lon_start = Ngl.get_float(contour_slick.sffield, "sfXCActualStartF") - 0.5
            lon_end = Ngl.get_float(contour_slick.sffield, "sfXCActualEndF") + 0.5
            lat_start = Ngl.get_float(contour_slick.sffield, "sfYCActualStartF") - 0.5
            lat_end = Ngl.get_float(contour_slick.sffield, "sfYCActualEndF") + 0.5
        if time is not None:
            plot_title = plot_title + "~C~" + datetime.strftime(time, "%Y-%m-%d %H:%M")
        else:
            plot_title = plot_title
        # Define resources and plot
        map_rsrc = self.__map_resources(
            plot_title, [lon_start, lon_end, lat_start, lat_end]
        )
        map_plot = Ngl.map(wks, map_rsrc)  # Draw a plot of map
        # Add a black cross marker with a label
        marker_rsrc = Ngl.Resources()
        marker_rsrc.gsMarkerIndex = (
            2  # Cross marker index (you can adjust if necessary)
        )
        marker_rsrc.gsMarkerSizeF = 0.02  # Adjust size as needed
        marker_rsrc.gsMarkerColor = "Black"  # Marker color
        marker_rsrc.gsMarkerThicknessF = 5.0  # thicker marker outlines
        # Longitude and Latitude for the marker
        if self.release_coords is not None:
            marker_lon = self.release_coords[0]
            marker_lat = self.release_coords[1]
            # Plot the marker
            Ngl.add_polymarker(wks, map_plot, marker_lon, marker_lat, marker_rsrc)
        # OVERLAY FIELDS ON THE MAP PLOT and DRAW
        Ngl.overlay(map_plot, contour_slick)
        Ngl.overlay(map_plot, vector_curr)
        if plot_wind:
            Ngl.overlay(map_plot, vector_wind)
        Ngl.maximize_plot(wks, map_plot)  # Maximize size of plot in frame.
        Ngl.draw(map_plot)
        Ngl.frame(wks)
        Ngl.destroy(wks)

    def __map_resources(
        self, plot_title: str, boundaries: list[float]
    ) -> Ngl.Resources:
        """
        Create and customize ngl resources for geographical map.
        """
        mpres = Ngl.Resources()  # map resources
        mpres.nglDraw = False
        mpres.nglFrame = False
        mpres.mpProjection = "CylindricalEquidistant"
        mpres.mpDataBaseVersion = "HighRes"
        mpres.mpLimitMode = "LatLon"
        lon_start, lon_end, lat_start, lat_end = boundaries
        mpres.mpMinLonF = lon_start
        mpres.mpMaxLonF = lon_end
        mpres.mpMinLatF = lat_start
        mpres.mpMaxLatF = lat_end
        mpres.tmXBMode = "Automatic"
        mpres.tmXBFormat = "f"  # X-axis (longitude) labels as decimal
        mpres.tmYLFormat = "f"  # Y-axis (latitude) labels as decimal
        # Turn on tickmark labels
        mpres.tmXBOn = True
        mpres.tmYLOn = True
        mpres.mpPerimOn = True  # Turn on map perimeter.
        mpres.mpGridAndLimbOn = True
        mpres.mpPerimDrawOrder = "PostDraw"
        mpres.mpFillDrawOrder = "PostDraw"
        mpres.mpOutlineBoundarySets = "GeophysicalAndUSStates"
        mpres.mpGeophysicalLineThicknessF = 5.0  # thickness of outlines
        mpres.mpFillOn = True
        # Fill land and inland water and leave ocean transparent
        mpres.mpFillColors = ["background", "transparent", "transparent", "transparent"]
        mpres.pmTitleDisplayMode = "Always"  # Turn on map title.
        mpres.pmTickMarkDisplayMode = "Always"
        mpres.pmLabelBarOrthogonalPosF = -0.05
        # TITLE resources
        mpres.tiMainString = plot_title
        mpres.tiMainFontHeightF = 0.020
        mpres.tiMainFont = "Helvetica-bold"
        mpres.tiMainOffsetYF = 0.025
        return mpres

    def __wind_resources(self, lon: NDArray, lat: NDArray) -> Ngl.Resources:
        """
        Create and customize ngl resources for winds.
        """
        wd_vres = Ngl.Resources()  # Wind vector resources
        wd_vres.nglDraw = False
        wd_vres.nglFrame = False
        # WIND VECTOR FIELD SETUP
        wd_vres.vfXArray = lon
        wd_vres.vfYArray = lat
        wd_vres.vcGlyphStyle = "LineArrow"
        wd_vres.vcMonoLineArrowColor = True
        wd_vres.vcLineArrowColor = "green"
        wd_vres.vcRefLengthF = 0.05
        wd_vres.vcRefMagnitudeF = 2
        wd_vres.vcPositionMode = "ArrowTail"
        wd_vres.vcLineArrowThicknessF = 4.0  # Quadruple the thickness.
        # wd_vres.vcRefAnnoOrthogonalPosF   =  0.159 # Move reference annotation up
        wd_vres.vcRefAnnoParallelPosF = 0.59  # and over to left.
        wd_vres.vcRefAnnoString1 = "2 m s~S~-1~N~"
        wd_vres.vcRefAnnoString2 = "Wind Vectors"
        wd_vres.vfMissingUValueV = np.nan
        wd_vres.vfMissingVValueV = np.nan
        return wd_vres

    def __current_resources(self, lon: NDArray, lat: NDArray) -> Ngl.Resources:
        """
        Create and customize ngl resources for currents.
        """
        c_vcres = Ngl.Resources()  # ocean currents vector resources
        c_vcres.nglDraw = False
        c_vcres.nglFrame = False
        c_vcres.vfXCStartV = float(self.lon_curr[0])  # Define X/Y axes range
        c_vcres.vfXCEndV = float(self.lon_curr[-1])  # for vector plot.
        c_vcres.vfYCStartV = float(self.lat_curr[0])
        c_vcres.vfYCEndV = float(self.lat_curr[-1])
        # CURRENTS VECTOR FIELD SETUP
        c_vcres.vcGlyphStyle = "CurlyVector"
        c_vcres.vcMonoLineArrowColor = True
        c_vcres.vcLineArrowColor = "black"
        c_vcres.vcRefLengthF = 0.05
        c_vcres.vcRefMagnitudeF = 0.2
        c_vcres.vcPositionMode = "ArrowTail"
        c_vcres.vcLineArrowThicknessF = 2.0  # Double the thickness.
        #    c_vcres.vcRefAnnoOrthogonalPosF = -0.179 # Move reference annotation up
        c_vcres.vcRefAnnoParallelPosF = 0.19  # and over to left.
        c_vcres.vcRefAnnoString1 = "0.2 m s~S~-1~N~"
        c_vcres.vcRefAnnoString2 = "Ocean Currents"
        c_vcres.vfMissingUValueV = np.nan
        c_vcres.vfMissingVValueV = np.nan
        return c_vcres

    def __slick_resources(self, lon: NDArray, lat: NDArray) -> Ngl.Resources:
        """
        Create and customize ngl resources for currents.

        args: max_concentration = maximum concentration
        """
        cfres = Ngl.Resources()  # Contour line/fill resources
        cfres.nglDraw = False
        cfres.nglFrame = False
        cfres.sfXCStartV = float(lon[0])  # Define X/Y axes range
        cfres.sfXCEndV = float(lon[-1])  # for contour plot.
        cfres.sfYCStartV = float(lat[0])
        cfres.sfYCEndV = float(lat[-1])
        # SCALAR FIELD SETUP
        cfres.cnFillOn = True  # Turn on contour fill.
        cfres.cnLinesOn = False  # Turn on contour lines.
        cfres.cnLineLabelsOn = False  # Turn off contour line labels.
        cfres.lbOrientation = "Horizontal"  # horizontal labelbar
        cfres.lbLabelFontHeightF = 0.012  # Decrease font size.
        cfres.pmLabelBarOrthogonalPosF = -0.05  # Move labelbar up.
        cfres.cnLineDashPatterns = 3  # dashed contour lines
        cfres.cnLineThicknessF = 3.0  # triple thick contour lines
        cfres.cnInfoLabelOrthogonalPosF = -0.15  # Move info label up.
        cfres.cnFillMode = "RasterFill"  # or "AreaFill" or "CellFill" or "RasterFill"
        cfres.cnRasterSmoothingOn = True
        cfres.cnLevelSelectionMode = "ExplicitLevels"  # Define your own contour levels
        min_concentration = self.min_concentration
        max_concentration = self.max_concentration
        concentration_order_of_magnitude = math.floor(math.log10(max_concentration))
        if concentration_order_of_magnitude < 1:
            n_levels = 11
            n_decimals = (-concentration_order_of_magnitude) + 1
        else:
            n_levels = 11
            n_decimals = 0
        contour_levels = np.round(
            np.linspace(
                min_concentration,
                max_concentration,
                n_levels,
            ),
            decimals=n_decimals,
        )
        if contour_levels[0] == 0.0:
            contour_levels[0] = 10 ** (-n_decimals - 1)  # contour_levels[1:]
        # Set explicit contour levels
        cfres.cnLevelSelectionMode = "ExplicitLevels"
        cfres.cnLevels = list(contour_levels)
        # Additional customization
        cfres.cnLineLabelPlacementMode = "Constant"
        cfres.lbOrientation = "Horizontal"
        cfres.lbTitleString = "tons km~S~-2~N~"
        cfres.lbTitleFontHeightF = 0.018
        cfres.lbLabelFontHeightF = 0.014
        cfres.lbTitleOffsetF = -0.27
        cfres.lbBoxMinorExtentF = 0.15
        cfres.pmLabelBarOrthogonalPosF = -0.01
        cfres.sfMissingValueV = np.nan
        return cfres


if __name__ == "__main__":
    """
    Plot results using pyngl
    """
    # Arguments
    root_directory = str(sys.argv[1])
    plot_step = int(sys.argv[2])
    spill_lon = float(sys.argv[3])
    spill_lat = float(sys.argv[4])
    start_datetime = str(sys.argv[5])
    sim_length = float(sys.argv[6])
    lon_min = float(sys.argv[7])
    lon_max = float(sys.argv[8])
    lat_min = float(sys.argv[9])
    lat_max = float(sys.argv[10])
    concentration_range = []
    concentration_range.append(float(sys.argv[11]))
    concentration_range.append(float(sys.argv[12]))
    if concentration_range[0] == concentration_range[1] == 0.0:
        concentration_range = None
    oce_files = os.path.join(str(sys.argv[13]), "*.nc")
    met_files = os.path.join(str(sys.argv[14]), "*.nc")
    # Release coordinates
    release_coords = [spill_lon, spill_lat]
    # read output netcdf in concentration
    concentration_path = os.path.join(
        root_directory, "out_files", "oil_concentration.nc"
    )
    ds_oil = xr.open_dataset(concentration_path)
    n_time = len(ds_oil.time)
    ds_oil = ds_oil.isel(
        time=slice(plot_step - 1, n_time + 1, plot_step)
    )  # slice in time
    # simulation initial and end dates
    inidate = pd.to_datetime(start_datetime) + pd.Timedelta(hours=plot_step)
    enddate = pd.to_datetime(inidate + pd.Timedelta(hours=sim_length))
    # opening currents netcdf
    curr = xr.open_mfdataset(oce_files)
    wind = xr.open_mfdataset(met_files)
    # transpose dataset
    curr = curr.transpose("time", "lon", "lat", ...)
    wind = wind.transpose("time", "lon", "lat", ...)
    ds_oil = ds_oil.transpose("time", "lon", "lat")
    # interpolate wind on concentration grid
    wind = wind.interp(lon=ds_oil.lon.values.tolist(), lat=ds_oil.lat.values.tolist())
    # ensuring date index is correct
    try:
        curr["time"] = curr.indexes["time"].to_datetimeindex()
        wind["time"] = wind.indexes["time"].to_datetimeindex()
    except (KeyError, AttributeError):
        pass
    # Time interpolation
    curr = curr.resample(time="h").interpolate("linear")
    wind = wind.resample(time="h").interpolate("linear")
    # selecting simulation date
    curr = curr.sel(time=slice(inidate, enddate, plot_step))
    wind = wind.sel(time=slice(inidate, enddate, plot_step))
    curr = curr.isel(depth=0)
    # selecting simulation domain
    curr = curr.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    wind = wind.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    ds_oil = ds_oil.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    # gravity center
    lon_gravity_center = ds_oil["lon_gravity_center"].values
    lat_gravity_center = ds_oil["lat_gravity_center"].values
    # Interpolate wind over slick center of gravity
    wind_u10m = np.zeros_like(wind.U10M.values)
    wind_u_values = wind.U10M.values
    wind_v10m = np.zeros_like(wind.V10M.values)
    wind_v_values = wind.V10M.values
    lat = ds_oil.lat.values  # Get latitude values
    lon = ds_oil.lon.values  # Get longitude values
    for t in range(len(ds_oil.time)):
        # Step 1: Specify the target latitude and longitude
        target_lon = lon_gravity_center[t]  # Replace with your target latitude
        target_lat = lat_gravity_center[t]  # Replace with your target longitude

        # Step 2: Calculate the distances
        # Using numpy to compute the squared distance
        lat_diff = np.abs(lat - target_lat)
        lon_diff = np.abs(lon - target_lon)

        # Create a meshgrid of the differences
        lat_grid, lon_grid = np.meshgrid(lat_diff, lon_diff, indexing="ij")
        distances = np.sqrt(lat_grid**2 + lon_grid**2)

        # Step 3: Find the index of the nearest grid point
        min_index = np.unravel_index(np.argmin(distances), distances.shape)
        # Step 4: Get the nearest value
        wind_u10m[t, min_index[1], min_index[0]] = wind_u_values[
            t, min_index[1], min_index[0]
        ]
        wind_v10m[t, min_index[1], min_index[0]] = wind_v_values[
            t, min_index[1], min_index[0]
        ]
    # Plot
    plot_path = os.path.join(
        root_directory, "out_files", "figures", "surf_oil_concentration"
    )
    pyngl_plot = PlotNgl(
        plot_path,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        concentration_range=concentration_range,
    )
    # Conversion to tons/km^2
    surf_concentration = ds_oil["concentration"].values * 1000
    pyngl_plot(
        lon=ds_oil.lon.values,
        lat=ds_oil.lat.values,
        surf_concentration=surf_concentration,
        currents=[curr.uo, curr.vo],
        winds=[wind_u10m, wind_v10m],
        spill_time=inidate,
        plot_title="Surface Oil Concentration",
        plot_step=plot_step,
        release_coords=release_coords,
    )
