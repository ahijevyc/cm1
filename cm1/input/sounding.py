"""
Load ERA5 model dataset for a user-specified time and location.

--campaign means use campaign storage.
Otherwise use s3fs Amazon Web Service bucket or local cached file.
"""

import logging
import os

import metpy.calc as mcalc
import pandas as pd
import xarray

import cm1.input.era5
from cm1.utils import TMPDIR, era5_circle_neighborhood, parse_args


def get_ofile(args):
    ofile = (
        TMPDIR
        / f"{pd.to_datetime(args.time).strftime('%Y%m%d_%H%M%S')}.{args.lat:~}.{args.lon:~}.nc"
    )
    return ofile


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
        ds = ds.sel(
            longitude=args.lon,
            latitude=args.lat,
            method="nearest",
        )
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
