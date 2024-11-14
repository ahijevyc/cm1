"""
Load ERA5 model dataset for a user-specified time and location.

--campaign means use campaign storage.
Otherwise use s3fs Amazon Web Service bucket or local cached file.
"""

import argparse
import glob
import logging
import os
import pdb
from pathlib import Path
from typing import Tuple

import metpy.calc as mcalc
import metpy.constants
import numpy as np
import pandas as pd
import s3fs
import xarray
from metpy.units import units


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
    parser.add_argument("time", help="time")
    parser.add_argument("lon", type=float, help="longitude in degrees East (0-360)")
    parser.add_argument("lat", type=float, help="latitude in degrees North")
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


import numpy as np
from sklearn.neighbors import BallTree as BallTree


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

    lat_mean = xarray.DataArray(lat_mean, attrs=lats_deg.attrs)
    lon_mean = xarray.DataArray(lon_mean, attrs=lons_deg.attrs)

    return lat_mean, lon_mean


def neighborhood(args, ds):
    """mean in neighborhood of args.neighbor points
    of args.lat, args.lon
    """
    # indexing="ij" allows Dataset.stack dim order to be "latitude", "longitude"
    lat2d, lon2d = np.meshgrid(ds.latitude, ds.longitude, indexing="ij")
    latlon = np.deg2rad(np.vstack([lat2d.ravel(), lon2d.ravel()]).T)
    X = np.deg2rad([[args.lat, args.lon]])

    (idx,) = BallTree(latlon, metric="haversine").query(
        X, return_distance=False, k=args.neighbors
    )
    ds = ds.stack(z=("latitude", "longitude")).isel(z=idx)
    print(ds.latitude, ds.longitude)
    lat_mean, lon_mean = mean_lat_lon(ds.latitude, ds.longitude)
    print(lat_mean, lon_mean)
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


def main() -> None:
    """
    Main function for loading ERA5 data and printing sounding data.

    If cached data exists, it is loaded from a local file. Otherwise,
    it retrieves data based on user input and saves it for later use.
    """

    import pickle

    args = parse_args()

    ofile = "t.nc"
    if os.path.exists(ofile):
        logging.warning(f"read {ofile}")
        with open(ofile, "rb") as file:
            ds = pickle.load(file)
    else:
        ds = load_era5(
            pd.to_datetime(args.time),
            campaign=args.campaign,
            model_levels=args.model_levels,
            glade=args.glade,
        )
    with open(ofile, "wb") as file:
        pickle.dump(ds, file)

    if args.neighbors == 1:
        ds = ds.sel(
            longitude=args.lon,
            latitude=args.lat,
            method="nearest",
        )
    else:
        ds = neighborhood(args, ds)

    print(to_sounding_txt(ds))


def load_era5(
    time: pd.Timestamp,
    campaign: bool = False,
    model_levels: bool = False,
    glade: Path = Path("/"),
) -> xarray.Dataset:
    """
    Load ERA5 dataset for specified time and configuration.

    Parameters
    ----------
    time : pd.Timestamp
        Desired timestamp for data retrieval.
    campaign : bool, optional
        Whether to use campaign storage, by default False.
    model_levels : bool, optional
        Use native model levels instead of pressure levels, by default False.

    Returns
    -------
    xarray.Dataset
        Dataset containing ERA5 data for the specified time and configuration.
    """
    if campaign or model_levels:
        if not campaign:
            logging.warning("getting model_levels from campaign storage")

        # get from campaign storage
        rdapath = Path(glade) / "glade/campaign/collections/rda/data"

        rdaindex, level_type = "d633000", "pl"
        varnames = [
            "128_129_z.ll025sc",
            "128_130_t.ll025sc",
            "128_131_u.ll025uv",
            "128_132_v.ll025uv",
            "128_133_q.ll025sc",
            "128_135_w.ll025sc",
        ]
        start_hour = time.floor("1d")
        end_hour = start_hour + pd.Timedelta(23, unit="hour")

        if model_levels:
            # rdaindex, level_type, varnames, start_hour, and end_hour
            # are different when you select model levels.
            rdaindex, level_type = "d633006", "ml"
            varnames = [
                "0_5_0_0_0_t.regn320sc",
                "0_5_0_1_0_q.regn320sc",
                "0_5_0_2_2_u.regn320uv",
                "0_5_0_2_3_v.regn320uv",
                "0_5_0_2_8_w.regn320sc",
                "128_134_sp.regn320sc",
            ]
            start_hour = time.floor("6h")
            end_hour = start_hour + pd.Timedelta(5, unit="hour")

        local_path = (
            rdapath / rdaindex / f"e5.oper.an.{level_type}" / time.strftime("%Y%m")
        )
        start_end = f"{start_hour.strftime('%Y%m%d%H')}_{end_hour.strftime('%Y%m%d%H')}"

        local_files = [
            local_path / f"e5.oper.an.{level_type}.{varname}.{start_end}.nc"
            for varname in varnames
        ]

        for local_file in local_files:
            assert os.path.exists(local_file), f"Could not find {local_file}"

        # Drop "utc_date" to avoid error about "Gregorian year" when quantifying
        # For some reason Gaussian "zero" is flipped for some latitudes in "SP" file.
        ds = xarray.open_mfdataset(local_files, drop_variables=["zero", "utc_date"])
        logging.warning(f"opened {local_files}")
        logging.warning(f"selected {time}")
        ds = ds.sel(time=time)
        ds = ds.metpy.quantify()

        if model_levels:
            # Derive pressure from a and b coefficients
            ds["P"] = ds.a_model + ds.b_model * ds.SP
            ds["P"].attrs.update(dict(long_name="pressure"))
            ds["P"] = ds["P"].transpose(*ds.U.dims)  # keep dim order consistent with U
            ds["P_half"] = ds.a_half + ds.b_half * ds.SP
            ds["P_half"].attrs.update(dict(long_name="pressure"))
            # Invariant field geopotential height at surface
            Zsfc = (
                xarray.open_dataset(
                    rdapath
                    / rdaindex
                    / "e5.oper.invariant/e5.oper.invariant.128_129_z.regn320sc.2016010100_2016010100.nc"
                )
                .squeeze()
                .drop_vars(["utc_date", "time"])
                .metpy.quantify()
                .rename_vars({"Z": "Zsfc"})
            )
            ds = ds.merge(Zsfc)

            logging.warning("filling height using hypsometric equation")
            ds["Tv"] = mcalc.thermo.virtual_temperature(ds.T, ds.Q)
            z_h = ds.Zsfc.assign_coords(half_level=ds.half_level.max())
            Z = []  # geopotential on full levels
            Z_h = [z_h]  # geopotential on half levels
            # Loop from last to first level (sfc upward)
            for level in ds.level[
                ::-1
            ]:  # accumulate geopotential in z_h upward from sfc
                z_h, z_f = compute_z_level(ds, level, z_h)
                Z.append(z_f)
                Z_h.append(z_h)

            ds["Z_h"] = xarray.concat(Z_h, dim="half_level") / metpy.constants.g
            ds["Z_h"].attrs["long_name"] = "geopotential height"
            ds["Z"] = xarray.concat(Z, dim="level") / metpy.constants.g
            ds["Z"].attrs["long_name"] = "geopotential height"
            ds["Zsfc"] = ds["Zsfc"] / metpy.constants.g
            ds["Zsfc"].attrs["long_name"] = "geopotential height at surface"
            ds = ds.drop_dims("half_level")

        else:
            ds["P"] = (
                ds.level * ds.level.metpy.units
            )  # ds.level.metpy.quantify() didn't preserve units

    else:
        # If not local download from S3 bucket.
        ds = s3_era5_dataset(time)

    return ds


def s3_era5_dataset(time: pd.Timestamp) -> xarray.Dataset:
    """
    Retrieve ERA5 data from an S3 bucket and cache it locally.

    Parameters
    ----------
    time : pd.Timestamp
        Desired timestamp for data retrieval.

    Returns
    -------
    xarray.Dataset
        Dataset containing ERA5 data for the specified time, downloaded from S3.
    """
    # Define the S3 bucket name and ERA5 path
    S3_BUCKET = "nsf-ncar-era5"
    S3_PATH_TEMPLATE = S3_BUCKET + f"/e5.oper.an.pl/{time.year}{time.month:02d}"

    # Define the cache directory
    tmpdir = Path(os.getenv("TMPDIR"))
    CACHE_DIR = tmpdir / "era5_cache"
    # Ensure cache directory exists
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    cache_file_paths = []
    for var in [
        "133_q.ll025sc",
        "130_t.ll025sc",
        "131_u.ll025uv",
        "132_v.ll025uv",
        "135_w.ll025sc",
        "129_z.ll025sc",
    ]:
        file_name = f"e5.oper.an.pl.128_{var}.{time.strftime('%Y%m%d00')}_{time.strftime('%Y%m%d23')}.nc"
        cache_file_path = CACHE_DIR / file_name
        cache_file_paths.append(cache_file_path)

        # Check if the file is already cached.
        if os.path.exists(cache_file_path):
            logging.warning(f"Found cached s3 {var} {time}")
            continue

        # If not cached, download from S3

        # Connect to S3 bucket
        s3 = s3fs.S3FileSystem(anon=True)

        # Generate the file name based on time
        # Full key of the file in the S3 bucket
        s3_file_path = f"{S3_PATH_TEMPLATE}/{file_name}"

        # Check if file exists in the S3 bucket
        if not s3.exists(s3_file_path):
            raise FileNotFoundError(f"{s3_file_path} not found in S3 bucket")

        # Download the file
        logging.warning(f"Downloading {s3_file_path} from S3...")
        with s3.open(s3_file_path, "rb") as f:
            xarray.open_dataset(f).to_netcdf(cache_file_path)
        logging.warning(f"Downloaded and cached: {cache_file_path}")

    ds = xarray.open_mfdataset(cache_file_paths).drop_vars("utc_date")
    logging.warning(f"selected {time}")
    ds = ds.sel(time=time)
    ds = ds.metpy.quantify()
    ds["P"] = ds.level * ds.level.metpy.units

    return ds


def compute_z_level(ds: xarray.Dataset, lev: int, z_h: float) -> Tuple[float, float]:
    r"""
    Compute the geopotential at a full level and the overlying half-level.

    This function calculates the geopotential \( z_f \) at a specified full level
    (given by `lev`) and updates the geopotential at the overlying half-level
    \( z_h \). The calculation is based on the virtual temperature at full level
    and pressure at half-levels.

    Parameters:
    ----------
    ds : xarray.Dataset
        The dataset containing the variables required for computation,
        specifically "Tv" (virtual temperature) and "P_half" (pressure on half-levels).
    lev : int
        The level index for the desired full-level geopotential calculation.
    z_h : float
        The initial geopotential height at the lower half-level.

    Returns:
    -------
    Tuple[float, float]
        A tuple containing:
            - `z_h`: The updated geopotential at the overlying half-level.
            - `z_f`: The computed geopotential at the specified full level.

    References
    ----------
    ERA5: Compute pressure and geopotential on model levels, geopotential height, and geometric height.
    ECMWF Confluence: https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height
    """
    # Virtual temperature at the specified level
    t_level = ds["Tv"].sel(level=lev)

    # Pressures at the half-levels above and below
    ph_lev = ds["P_half"].sel(half_level=lev)
    ph_levplusone = ds["P_half"].sel(half_level=lev + 1)
    pf_lev = ds["P"].sel(level=lev)

    if lev == 1:
        dlog_p = np.log(ph_levplusone / (0.1 * units.Pa))
        alpha = np.log(2)
    else:
        dlog_p = np.log(ph_levplusone / ph_lev)
        # TODO: understand IFS formulation for alpha. See IFS-documentation-cy47r3 eqn 2.23
        # alphaIFS = 1.0 - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)
        # @ahijevyc formulation
        alpha = np.log(ph_levplusone / pf_lev)
        # Make sure official and @ahijevyc values are close to each other.
        # assert np.allclose(
        #    alphaIFS.load(), alpha.load(), atol=1e-2
        # ), f"{np.abs((alphaIFS-alpha)).max()}"

    t_level = t_level * metpy.constants.Rd

    # Calculate the full-level geopotential `z_f`
    # Integrate from previous (lower) half-level `z_h` to the
    # full level
    z_f = z_h + (t_level * alpha)
    if "half_level" in z_f.coords:
        z_f = z_f.drop_vars("half_level")
    z_f = z_f.assign_coords(level=lev)

    # Update the half-level geopotential `z_h`
    z_h = z_h + (t_level * dlog_p)
    z_h = z_h.assign_coords(half_level=lev).drop_vars("level")

    return z_h, z_f


def to_sounding_txt(ds: xarray.Dataset) -> str:
    """
       The format is the same as that for the WRF Model.

     One-line header containing:   sfc pres (mb)    sfc theta (K)    sfc qv (g/kg)

      (Note1: here, "sfc" refers to near-surface atmospheric conditions.
       Technically, this should be z = 0, but in practice is obtained from the
       standard reporting height of 2 m AGL/ASL from observations)
      (Note2: land-surface temperature and/or sea-surface temperature (SST) are
       specified elsewhere: see tsk0 in namelist.input and/or tsk array in
       init_surface.F)

    Then, the following lines are:   z (m)    theta (K)   qv (g/kg)    u (m/s)    v (m/s)

      (Note3: # of levels is arbitrary)

        Index:   sfc    =  surface (technically z=0, but typically from 2 m AGL/ASL obs)
                 z      =  height AGL/ASL
                 pres   =  pressure
                 theta  =  potential temperature
                 qv     =  mixing ratio
                 u      =  west-east component of velocity
                 v      =  south-north component of velocity

    Note4:  For final line of input_sounding file, z (m) must be greater than the model top
            (which is nz * dz when stretch_z=0, or ztop when stretch_z=1,  etc)
    """
    # level is either 1-137 for model_levels, or pressure from top to sfc.
    # Either way, the closest to the sfc is the maximum.
    assert all(ds.level.diff("level") > 0), "levels not increasing"
    bottom = ds.level.max()
    # ds.SP (surface pressure) is half-level below ds.level.max()
    sfc_pres = ds.SP if "SP" in ds else ds.P.sel(level=bottom)
    ds["theta"] = mcalc.potential_temperature(ds.P, ds.T).metpy.convert_units("K")
    sfc_theta_K = ds["theta"].sel(level=bottom)
    ds["Q"] = ds["Q"].metpy.convert_units("g/kg")
    sfc_qv_gkg = ds.Q.sel(level=bottom)

    s = f"{sfc_pres.values} {sfc_theta_K.values} {sfc_qv_gkg.values}\n"
    s += (
        ds.to_dataframe()
        .sort_index(ascending=False)  # from surface upward
        .to_csv(
            columns=["Z", "theta", "Q", "U", "V"], sep=" ", header=False, index=False
        )
    )

    return s


if __name__ == "__main__":
    main()
