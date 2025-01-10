"""
Load ERA5 model dataset for a user-specified time and location.

--campaign means use campaign storage.
Otherwise use s3fs Amazon Web Service bucket or local cached file.
"""

import logging
import os

import metpy.calc as mcalc
import metpy.constants
import numpy as np
import pandas as pd
import xarray
from metpy.units import units

import cm1.input.era5
from cm1.utils import TMPDIR, era5_circle_neighborhood, parse_args

# Assuming this script is located in a subdirectory of the repository
repo_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
soundings_path = os.path.join(repo_base_path, "soundings")


def era5(ds, lat, lon):
    ds = ds.sel(
        longitude=lon,
        latitude=lat,
        method="nearest",
    )
    return ds


def get_ofile(args):
    ofile = (
        TMPDIR
        / f"{pd.to_datetime(args.time).strftime('%Y%m%d_%H%M%S')}.{args.lat:~}.{args.lon:~}.nc"
    )
    return ofile


def get_case(case):
    file_path = os.path.join(soundings_path, f"input_sounding_{case}")
    ds = read_from_txt(file_path)
    ds["case"] = case
    return ds


def trier():
    return get_case("trier")


def jordan_allmean():
    return get_case("jordan_allmean")


def jordan_hurricane():
    return get_case("jordan_hurricane")


def rotunno_emanuel():
    return get_case("rotunno_emanuel")


def dunion_MT():
    return get_case("dunion_MT")


def bryan_morrison():
    return get_case("bryan_morrison")


def seabreeze_test():
    return get_case("seabreeze_test")


def read_from_txt(file_path):
    """
    read CM1 sounding file format
    """
    # Open the file and read the first line (header with surface variables)
    with open(file_path, "r") as file:
        # Read the first line
        header = file.readline().strip()
        # Split the header into surface variables
        surface_pressure, surface_theta, surface_mixing_ratio = map(
            float, header.split()
        )

    # Read the remaining columns into a pandas DataFrame
    # Specify column names and skip the first line (header)
    column_names = ["Z", "theta", "Q", "U", "V"]
    df = pd.read_csv(file_path, sep=r"\s+", skiprows=1, names=column_names)
    # Rename the index to "level"
    df = df.rename_axis("level")
    ds = df.to_xarray()
    ds["Q"] = ds["Q"] * units.g / units.kg
    ds["SP"] = surface_pressure
    ds["SP"] *= units.hPa
    ds["theta"] = ds["theta"] * units.K
    ds["Z"] = ds["Z"] * units.m
    # TODO: calculate pressure from surface pressure and theta
    ds["P"] = ds["SP"] * np.exp(
        -metpy.constants.g * ds["Z"] / metpy.constants.Rd / ds["theta"]
    )
    p_bot = ds.SP.copy()  # copy to avoid modifying the original when adding dp
    z_bot = 0.0 * units.m
    P = []  # pressure
    dz = ds["Z"].diff("level")
    for level in ds.level:
        T = mcalc.temperature_from_potential_temperature(
            p_bot, ds.theta.sel(level=level)
        )
        dz = ds.Z.sel(level=level) - z_bot
        dp = p_bot * -metpy.constants.g * dz / metpy.constants.Rd / T
        P.append(p_bot.values + dp.values)
        p_bot += dp
        z_bot = ds.Z.sel(level=level)
    ds["p"] = xarray.DataArray(np.array(P), coords=[ds.level])
    ds["p"] *= ds.SP.metpy.units
    ds["T"] = mcalc.temperature_from_potential_temperature(ds["P"], ds["theta"])
    ds["Tv"] = mcalc.thermo.virtual_temperature(ds.T, ds.Q)
    ds["surface_potential_temperature"] = surface_theta
    ds["surface_potential_temperature"] *= units.K
    ds["surface_mixing_ratio"] = surface_mixing_ratio
    ds["surface_mixing_ratio"] *= units.g / units.kg
    ds["Zsfc"] = 0.0
    ds["Zsfc"] *= units.m
    ds["U"] = ds["U"] * units.m / units.s
    ds["V"] = ds["V"] * units.m / units.s
    return ds


def main() -> None:
    """
    Main function for loading ERA5 data and printing sounding data.

    If cached data exists, it is loaded from a local file. Otherwise,
    it retrieves data based on user input and saves it for later use.
    """

    import pickle

    args = parse_args()

    ofile = get_ofile(args)
    if os.path.exists(ofile):
        logging.warning(f"read {ofile}")
        with open(ofile, "rb") as file:
            ds = pickle.load(file)
    else:
        ds = cm1.input.era5.get(
            pd.to_datetime(args.time),
            campaign=args.campaign,
            model_levels=args.model_levels,
            glade=args.glade,
        )
        with open(ofile, "wb") as file:
            logging.warning(f"pickle dump {ofile}")
            pickle.dump(ds, file)

    if args.neighbors == 1:
        ds = era5(ds, args.lat, args.lon)
    else:
        ds = era5_circle_neighborhood(ds, args.lat, args.lon, args.neighbors)

    print(to_txt(ds))


def to_txt(ds: xarray.Dataset) -> str:
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

    s = f"{sfc_pres.compute().item().m_as('hPa')} {sfc_theta_K.values} {sfc_qv_gkg.values}\n"
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
