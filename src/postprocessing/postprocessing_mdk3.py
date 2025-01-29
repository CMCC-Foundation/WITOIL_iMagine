import numpy as np
import pandas as pd
import xarray as xr
import os
from glob import glob


class PostProcessing:
    """
    This class is used to apply postprocessing.
    """

    @staticmethod
    def create_concentration_dataset(
        lon_min: float = None,
        lon_max: float = None,
        lat_min: float = None,
        lat_max: float = None,
        resolution: float = 150.0,
        multiple_slick: bool = False,
        filepath: str = "",
    ):
        """
        This script creates a gridded netcdf from point output netcdf from Medslik-II

        The file is usually called as spill_properties.nc

        In order to create the netcdf file on a grid, some premises are made:

        1 - Bounding box: a bounding box is defined from all time steps in the spill properties netcdf, therefore collecting

            maximum and mininum latitudes and longitudes in all particles in the given output.

        2 - Resolution: Resolution can be modified, standard is 150 meters to respect oil grid in medslik.for.

            IMPORTANT! This code creates area values for each grid cell considering this value, so do not use degrees ads unit

        3 - Multiple slick flag allows the user to construct the netcdf from multiple slicks, combining all outputs on a single
            point netcdf and the gridded as well.

        """

        path_to_spillprop = os.path.join(filepath, "spill_properties.nc")
        # Opens all separate slicks if needed and group on a single netcdf
        if multiple_slick == True:
            files = glob(filepath + "*/spill_properties.nc")
            ds = xr.open_mfdataset(files, combine="nested", concat_dim="parcel_id")
        # Get the single file for regular simulations
        else:
            try:
                ds = xr.open_mfdataset(path_to_spillprop)
            except:
                file = glob(path_to_spillprop)
                ds = xr.open_mfdataset(file)

        # get the oil density in kg/m3 from the netcdf
        oil_density = ds.non_evaporative_volume.oil_density

        # Calculate latitude and longitude resolution from the given output
        if lat_min is None:
            lat_min = ds.latitude.values.min()
        if lon_min is None:
            lon_min = ds.longitude.values.min()
        if lat_max is None:
            lat_max = ds.latitude.values.max()
        if lon_max is None:
            lon_max = ds.longitude.values.max()

        # Obtain the resolution on degrees for latitude and longitude given the mean latitude value
        lat_mean = (lat_min + lat_max) / 2

        lat_resolution_degree = resolution / (111320)
        lon_resolution_degree = resolution / (111320 * np.cos(np.pi * (lat_mean) / 180))

        # Create latitude and longitude arrays
        lats_array = np.arange(lat_min, lat_max, lat_resolution_degree)
        lons_array = np.arange(lon_min, lon_max, lon_resolution_degree)

        # Create concentration grid - Later to be the base for the netcdf
        concentration_grid = np.zeros((len(ds.time), len(lats_array), len(lons_array)))
        lon_gravity_center = np.zeros(len(ds.time))
        lat_gravity_center = np.zeros(len(ds.time))
        # Loop over time and create the concentrations for each grid element
        for t in range(0, len(ds.time)):

            print(f"timestep {t}")

            # Select a single timesetp
            rec = ds.isel(time=t)

            # Obtain the coordinates for each point and its repective volume converted to tonnes
            lats = rec.latitude.values
            lons = rec.longitude.values
            statuses = rec.particle_status.values

            # Get the total amount of volume on water surface
            volumes = rec.non_evaporative_volume.values + rec.evaporative_volume.values

            # obtain the mass of each particle by multiplying by the oil density
            mass = volumes * oil_density

            # Create a Pandas DataFrame with information for each particle
            df = pd.DataFrame(
                {"latitude": lats, "longitude": lons, "mass": mass, "status": statuses}
            )

            # Keep only paticles that have status > 0 and less than 3
            particle_data = df[(df.status > 0) & (df.status < 3)]

            lon_gravity_center[t] = np.nanmean(particle_data["longitude"])
            lat_gravity_center[t] = np.nanmean(particle_data["latitude"])
            # Assign particles to grid cells -  Core of the solution with np.digitize
            particle_data["lat_bin"] = np.digitize(
                particle_data["latitude"].values, lats_array
            )
            particle_data["lon_bin"] = np.digitize(
                particle_data["longitude"].values, lons_array
            )

            # Aggregate masses within each grid cell and normalize by grid area
            aggregated = particle_data.groupby(["lat_bin", "lon_bin"]).agg(
                {"mass": "sum"}
            )

            # Reset index
            aggregated = aggregated.reset_index()

            # Transform from mass on cell to concentration, dividing by the resolution area
            aggregated["concentration"] = aggregated["mass"] / (resolution**2)

            # insert each grid concentration on the matrix
            for x in range(0, len(aggregated)):
                concentration_grid[
                    t,
                    int(aggregated.iloc[x].lat_bin) - 1,
                    int(aggregated.iloc[x].lon_bin) - 1,
                ] = aggregated.iloc[x].concentration

        # Create xarray Dataset from the array
        concentration_dataset = xr.Dataset(
            {
                "concentration": (["time", "lat", "lon"], concentration_grid),
                "lon_gravity_center": (["time"], lon_gravity_center),
                "lat_gravity_center": (["time"], lat_gravity_center),
            },
            coords={"time": ds.time.values, "lat": lats_array, "lon": lons_array},
        )
        concentration_dataset["concentration"].attrs["units"] = "kg/m^2"
        # Saves merged netcdf if needed
        if multiple_slick == True:
            ds.to_netcdf(filepath + "spill_properties_merged.nc")

        # Saves concentration grid netcdf
        concentration_path = os.path.join(filepath, "oil_concentration.nc")
        concentration_dataset.to_netcdf(concentration_path)


if __name__ == "__main__":
    PostProcessing.create_concentration_dataset(
        -65,
        -62,
        9,
        12,
        150,
        False,
        "/Users/francesco/shared/Medslik-II/testcase/paria/out_files/",
    )
