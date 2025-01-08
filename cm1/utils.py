import argparse
import logging
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from metpy.units import units
from sklearn.neighbors import BallTree

TMPDIR = Path(os.getenv("TMPDIR"))


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for ERA5 data retrieval.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments including time, longitude, latitude,
        and options for model levels and campaign storage.
    """
    parser = argparse.ArgumentParser(description="get ERA5 sounding at time, lon, lat")
    parser.add_argument("--time", help="time")
    parser.add_argument(
        "--lon",
        type=lambda x: float(x) % 360 * units.degreeE,
        help="longitude in degrees East",
    )
    parser.add_argument(
        "--lat",
        type=lambda x: float(x) * units.degreeN,
        help="latitude in degrees North",
    )
    parser.add_argument(
        "--model_levels", action="store_true", help="native model levels"
    )
    parser.add_argument("--campaign", action="store_true", help="use campaign storage")
    parser.add_argument("--glade", default="/", help="parent of glade directory")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=1,
        help="number of neighbors to average",
    )
    args = parser.parse_args()
    logging.info(args)
    return args


def mean_lat_lon(lats_deg, lons_deg):
    """
    Calculates the mean latitude and longitude of a set of points on a sphere.

    Args:
        lats (list): List of latitudes in degrees.
        lons (list): List of longitudes in degrees.

    Returns:
        tuple: Mean latitude and longitude in degrees.
    """

    # Convert to radians
    lats = np.radians(lats_deg)
    lons = np.radians(lons_deg)

    # Calculate Cartesian coordinates
    x = np.cos(lats) * np.cos(lons)
    y = np.cos(lats) * np.sin(lons)
    z = np.sin(lats)

    # Calculate mean Cartesian coordinates
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    # Convert back to spherical coordinates
    lon_mean = np.arctan2(y_mean, x_mean)
    hyp = np.sqrt(x_mean**2 + y_mean**2)
    lat_mean = np.arctan2(z_mean, hyp)

    # Convert back to degrees
    lat_mean = np.degrees(lat_mean)
    lon_mean = np.degrees(lon_mean)

    # Transfer attributes of inputs to outputs lat_mean and lon_mean.
    lat_mean = lat_mean * units.degrees_N
    lon_mean = lon_mean * units.degrees_E

    return lat_mean, lon_mean


def era5_circle_neighborhood(ds, lat, lon, neighbors, debug=True):
    """
    mean in neighborhood of
    of lat, lon
    Average args.neighbor points.
    """
    if neighbors > 100:
        logging.warning(
            f"averaging {neighbors} neighbors along model levels may result in averaging between wildly different pressures and heights"
        )
    # indexing="ij" allows Dataset.stack dims order to be "latitude", "longitude"
    lat2d, lon2d = np.meshgrid(ds.latitude, ds.longitude, indexing="ij")
    latlon = np.deg2rad(np.vstack([lat2d.ravel(), lon2d.ravel()]).T)
    X = [[lat.m_as("radian"), lon.m_as("radian")]]

    (idx,) = BallTree(latlon, metric="haversine").query(
        X, return_distance=False, k=neighbors
    )
    ds = ds.stack(z=("latitude", "longitude")).isel(z=idx).load()
    lat_mean, lon_mean = mean_lat_lon(ds.latitude, ds.longitude)
    sfc_pressure_range = ds.SP.max() - ds.SP.min()
    if sfc_pressure_range > 1 * units.hPa:
        logging.warning(f"sfc_pressure_range: {sfc_pressure_range:~}")
    sfc_height_range = ds.Zsfc.max() - ds.Zsfc.min()
    if sfc_height_range > 10 * units.m:
        logging.warning(f"sfc_height_range: {sfc_height_range:~}")

    # Plot requested sounding location and nearest neighbors used for averaging.
    if debug:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig, ax = plt.subplots(
            figsize=(10, 5),
            subplot_kw={
                "projection": ccrs.PlateCarree(central_longitude=lon_mean.data)
            },
        )

        # Add features to the map
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.add_feature(cfeature.OCEAN)

        # Plot the points
        ax.plot(
            lon_mean,
            lat_mean,
            marker="o",
            transform=ccrs.PlateCarree(),
            label="mean",
            linestyle="none",
        )
        ax.plot(
            lon.m_as("degrees_E"),
            lat.m_as("degrees_N"),
            marker=".",
            transform=ccrs.PlateCarree(),
            label="request",
            linestyle="none",
        )
        ax.scatter(ds.longitude, ds.latitude, marker=".", transform=ccrs.PlateCarree())
        ax.set_title(f"{lat_mean.data} {lon_mean.data}")
        ax.gridlines(draw_labels=True)
        ax.legend()
        plt.show()
        # TODO: remove assertions after function is bug-free
        # assert mean location is close to what was requested.
        assert abs(lat_mean - lat) < 1, f"meanlat {lat_mean} lat {lat}"
        assert (
            np.cos(np.radians(lon_mean)) - np.cos(np.radians(lon))
        ) < 0.01, f"meanlon {lon_mean} lon {lon} {np.cos(np.radians(lon_mean)) - np.cos(np.radians(lon))}"
        assert (
            np.sin(np.radians(lon_mean)) - np.sin(np.radians(lon))
        ) < 0.01, f"meanlon {lon_mean} lon {lon} {np.sin(np.radians(lon_mean)) - np.sin(np.radians(lon))}"

    ds = ds.mean(dim="z")  # drop `z` `latitude` `longitude` dimensions
    ds = ds.assign_coords(latitude=lat_mean, longitude=lon_mean)
    ds.attrs["neighbors"] = neighbors
    return ds
