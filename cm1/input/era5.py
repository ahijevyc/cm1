import logging
import os
import pdb
from pathlib import Path
from typing import Tuple

import metpy
import metpy.calc as mcalc
import numpy as np
import pandas as pd
import s3fs
import xarray
from metpy.units import units

from cm1.utils import TMPDIR


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
    z_f = z_f.drop_vars("half_level")
    z_f = z_f.assign_coords(level=lev)

    # Update the half-level geopotential `z_h`
    z_h = z_h + (t_level * dlog_p)
    z_h = z_h.assign_coords(half_level=lev).drop_vars("level")

    return z_h, z_f


def get(
    time: pd.Timestamp,
    campaign: bool = True,
    model_levels: bool = True,
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
            logging.warning(
                "getting model_levels from campaign storage because model_levels=True"
            )

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
        logging.warning(f"opened {len(local_files)} local files")
        logging.info(local_files)
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

            ds["Z_half"] = xarray.concat(Z_h, dim="half_level") / metpy.constants.g
            ds["Z_half"].attrs["long_name"] = "geopotential height"
            ds["Z"] = xarray.concat(Z, dim="level") / metpy.constants.g
            ds["Z"].attrs["long_name"] = "geopotential height"
            ds["Zsfc"] = ds["Zsfc"] / metpy.constants.g
            ds["Zsfc"].attrs["long_name"] = "geopotential height at surface"
            # ds = ds.drop_dims("half_level") # why drop this?

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

    # Why not use this? It's managed by some private company
    # https://registry.opendata.aws/ecmwf-era5/

    # Define the cache directory
    CACHE_DIR = TMPDIR / "era5_cache"
    # Ensure cache directory exists
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # pressure level data
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

        # Download from S3
        if "s3" not in globals():
            s3 = s3fs.S3FileSystem(anon=True)

        s3_file_path = (
            S3_BUCKET + f"/e5.oper.an.pl/{time.year}{time.month:02d}/{file_name}"
        )

        # Check if file exists in the S3 bucket
        if not s3.exists(s3_file_path):
            raise FileNotFoundError(f"{s3_file_path} not found in S3 bucket")

        # Download the file
        logging.warning(f"Downloading {s3_file_path} from S3...")
        with s3.open(s3_file_path, "rb") as f:
            xarray.open_dataset(f).to_netcdf(cache_file_path)
        logging.warning(f"Downloaded and cached: {cache_file_path}")

    ds_pl = xarray.open_mfdataset(cache_file_paths).drop_vars("utc_date")
    logging.info(f"selecting {time} from ds_pl")
    ds_pl = ds_pl.sel(time=time)
    ds_pl = ds_pl.metpy.quantify()
    ds_pl["P"] = ds_pl.level * ds_pl.level.metpy.units
    # Convert geopotential to geopotential height
    ds_pl["Z"] /= metpy.constants.g

    # surface data
    cache_file_paths = []
    for var in [
        "134_sp.ll025sc",
        "167_2t.ll025sc",
        "168_2d.ll025sc",
    ]:
        lastdayofmonth = time + pd.offsets.MonthEnd(0)
        file_name = f"e5.oper.an.sfc.128_{var}.{time.strftime('%Y%m')}0100_{time.strftime('%Y%m')}{lastdayofmonth.strftime('%d')}23.nc"
        cache_file_path = CACHE_DIR / file_name
        cache_file_paths.append(cache_file_path)

        # Check if the file is already cached.
        if os.path.exists(cache_file_path):
            logging.warning(f"Found cached s3 {var} {time}")
            continue

        # Download from S3
        if "s3" not in globals():
            s3 = s3fs.S3FileSystem(anon=True)

        s3_file_path = (
            S3_BUCKET + f"/e5.oper.an.sfc/{time.year}{time.month:02d}/{file_name}"
        )

        # Check if file exists in the S3 bucket
        if not s3.exists(s3_file_path):
            raise FileNotFoundError(f"{s3_file_path} not found in S3 bucket")

        # Download the file
        logging.warning(f"Downloading {s3_file_path} from S3...")
        with s3.open(s3_file_path, "rb") as f:
            xarray.open_dataset(f).to_netcdf(cache_file_path)
        logging.warning(f"Downloaded and cached: {cache_file_path}")

    ds_sfc = xarray.open_mfdataset(cache_file_paths).drop_vars("utc_date")
    logging.warning(f"selecting {time} from ds_sfc")
    ds_sfc = ds_sfc.sel(time=time)
    ds_sfc = ds_sfc.metpy.quantify()

    # Calculate surface potential temperature and mixing ratio
    ds_sfc["surface_potential_temperature"] = mcalc.potential_temperature(
        ds_sfc.SP,
        ds_sfc.VAR_2T,
    )
    ds_sfc["surface_mixing_ratio"] = mcalc.mixing_ratio_from_specific_humidity(
        mcalc.specific_humidity_from_dewpoint(ds_sfc.SP, ds_sfc.VAR_2D)
    )

    # invariant field geopotential height at surface
    file_name = "e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
    cache_file_path = CACHE_DIR / file_name
    # Check if the file is already cached.
    if os.path.exists(cache_file_path):
        logging.warning(f"Found cached s3 invariant z")
    else:
        # Download from S3
        # Connect to S3 bucket
        if "s3" not in globals():
            s3 = s3fs.S3FileSystem(anon=True)

        s3_file_path = S3_BUCKET + "/e5.oper.invariant/197901/" + file_name

        # Check if file exists in the S3 bucket
        if not s3.exists(s3_file_path):
            raise FileNotFoundError(f"{s3_file_path} not found in S3 bucket")

        # Download the file
        logging.warning(f"Downloading {s3_file_path} from S3...")
        with s3.open(s3_file_path, "rb") as f:
            xarray.open_dataset(f).to_netcdf(cache_file_path)
        logging.warning(f"Downloaded and cached: {cache_file_path}")

    Zsfc = (
        xarray.open_dataset(cache_file_path, drop_variables=["utc_date", "time"])
        .squeeze(dim="time")
        .metpy.quantify()
        .rename_vars({"Z": "Zsfc"})
    ) / metpy.constants.g
    ds = ds_pl.merge(ds_sfc).merge(Zsfc)

    return ds
