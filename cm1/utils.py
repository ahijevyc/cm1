import argparse
import logging
import pdb

import numpy as np
from matplotlib import pyplot as plt
from metpy.units import units
from sklearn.neighbors import BallTree


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
        type=lambda x: float(x) * units.degreeE,
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
    # Map to 0-360 degreesE
    setattr(args, "lon", args.lon % (360 * units.degreesE))
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


def neighborhood(args, ds):
    """mean in neighborhood of args.neighbor points
    of args.lat, args.lon
    """
    if args.neighbors > 100:
        logging.warning(
            f"averaging {args.neighbors} neighbors along model levels may result in averaging between wildly different pressures and heights"
        )
    # indexing="ij" allows Dataset.stack dims order to be "latitude", "longitude"
    lat2d, lon2d = np.meshgrid(ds.latitude, ds.longitude, indexing="ij")
    latlon = np.deg2rad(np.vstack([lat2d.ravel(), lon2d.ravel()]).T)
    X = [[args.lat.m_as("radian"), args.lon.m_as("radian")]]

    (idx,) = BallTree(latlon, metric="haversine").query(
        X, return_distance=False, k=args.neighbors
    )
    ds = ds.stack(z=("latitude", "longitude")).isel(z=idx)
    lat_mean, lon_mean = mean_lat_lon(ds.latitude, ds.longitude)

    # Plot requested sounding location and nearest neighbors used for averaging.
    if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
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
            lon_mean, lat_mean, marker="o", color="red", transform=ccrs.PlateCarree()
        )
        ax.scatter(ds.longitude, ds.latitude, marker=".", transform=ccrs.PlateCarree())
        ax.set_title(f"{lat_mean.data} {lon_mean.data}")
        ax.gridlines(draw_labels=True)
        plt.show()
        # TODO: remove assertions after function is bug-free
        # assert mean location is close to what was requested.
        assert abs(lat_mean - args.lat) < 1, f"meanlat {lat_mean} args.lat {args.lat}"
        assert (
            np.cos(np.radians(lon_mean)) - np.cos(np.radians(args.lon))
        ) < 0.01, f"meanlon {lon_mean} args.lon {args.lon} {np.cos(np.radians(lon_mean)) - np.cos(np.radians(args.lon))}"
        assert (
            np.sin(np.radians(lon_mean)) - np.sin(np.radians(args.lon))
        ) < 0.01, f"meanlon {lon_mean} args.lon {args.lon} {np.sin(np.radians(lon_mean)) - np.sin(np.radians(args.lon))}"
    ds = ds.mean(dim="z")  # lose `z` `latitude` `longitude`
    ds = ds.assign_coords(latitude=lat_mean, longitude=lon_mean)
    ds.attrs["neighbors"] = args.neighbors
    return ds
