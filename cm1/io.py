"""
Get model dataset for a user-specified time.

If campaign=True get from campaign storage.
Otherwise get from Amazon bucket.
"""
import argparse
import os
from pathlib import Path
import pdb

import pandas as pd
import s3fs
import xarray


def parse_args():
    parser = argparse.ArgumentParser(description="get ERA5 sounding at time, lon, lat")
    parser.add_argument("time", help="time")
    parser.add_argument("lon", help="longitude in degrees East")
    parser.add_argument("lat", help="latitude in degrees North")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    s = load_era5(pd.to_datetime(args.time)).sel(longitude=args.lon, latitude=args.lat)
    print(s)


def download_era5_data(time: pd.Timestamp) -> xarray.Dataset:
    # Define the S3 bucket name and ERA5 path
    S3_BUCKET = "nsf-ncar-era5"
    S3_PATH_TEMPLATE = f"nsf-ncar-era5/e5.oper.an.pl/{time.year}{time.month:02d}"

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
            print(f"Data for {time} already cached.")
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
        print(f"Downloading {s3_file_path} from S3...")
        with s3.open(s3_file_path, "rb") as f:
            xarray.open_dataset(f).to_netcdf(cache_file_path)
        print(f"Downloaded and cached: {cache_file_path}")

    ds = xarray.open_mfdataset(cache_file_paths)
    return ds


def load_era5(time: pd.Timestamp, campaign: bool = True, model_levels=False) -> xarray.Dataset:
    if campaign:
        # get from campaign storage
        rdapath = Path("/glade/campaign/collections/rda/data")

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

        local_path = rdapath / rdaindex / f"e5.oper.an.{level_type}" / time.strftime("%Y%m")
        start_end = f"{start_hour.strftime('%Y%m%d%H')}_{end_hour.strftime('%Y%m%d%H')}"

        local_files = [
            local_path / f"e5.oper.an.{level_type}.{varname}.{start_end}.nc"
            for varname in varnames
        ]
        for local_file in local_files:
            assert os.path.exists(local_file), f"Could not find {local_file}"

        # For some reason the "zero" variable is different for surface_pressure
        ds = xarray.open_mfdataset(local_files, drop_variables="zero")
    else:
        # If not local download from S3 bucket.
        ds = download_era5_data(time)

    return ds


if __name__ == "__main__":
    main()
