import argparse
import logging
import os
from pathlib import Path

import numpy as np
from metpy.units import units

TMPDIR = Path(os.getenv("TMPDIR"))


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for ERA5 data retrieval.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments including time, longitude, latitude,
        and optional path to glade directory.
    """
    parser = argparse.ArgumentParser(description="get ERA5 sounding at time, lon, lat")
    parser.add_argument("time", help="time")
    parser.add_argument(
        "lon",
        type=lambda x: float(x) % 360 * units.degreeE,
        help="longitude in degrees East",
    )
    parser.add_argument(
        "lat",
        type=lambda x: float(x) * units.degreeN,
        help="latitude in degrees North",
    )
    parser.add_argument("--glade", default="/", help="parent of glade directory")
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
