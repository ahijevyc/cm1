"""
Plot ERA5 Skew-T and hodograph for a specified time and location.

This script generates a Skew-T diagram and a hodograph for ERA5 model data at 
a user-specified time and location. It includes temperature, dewpoint, and wind
data, as well as dry adiabats, moist adiabats, and mixing lines on the Skew-T diagram.
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from metpy.interpolate import interpolate_1d
from metpy.plots import Hodograph, SkewT
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pint import Quantity

from cm1.io import load_era5, parse_args


def main() -> None:
    """
    Main function to load ERA5 data for a specified time and location,
    and plot a Skew-T diagram with hodograph.

    Parameters are read from the command-line arguments using `parse_args()`.
    """

    args = parse_args()
    ds = load_era5(
        pd.to_datetime(args.time),
        campaign=args.campaign,
        model_levels=args.model_levels,
        glade=args.glade,
    )

    logging.warning(f"select {args}")
    ds = ds.sel(longitude=args.lon, latitude=args.lat, method="nearest")
    skewt(ds)
    plt.show()


def skewt(
    ds: xarray.Dataset,
    fig: Optional[Figure] = None,
    subplot: Optional[Tuple[int, int, int]] = None,
    rotation: int = 40,
    ptop: Quantity = 100 * units.hPa,
) -> SkewT:
    """
    Generates a Skew-T diagram with temperature, dewpoint, wind barbs, CAPE, CIN,
    and other atmospheric features, using data from an ERA5 dataset. A hodograph
    is also plotted as an inset within the Skew-T diagram.

    Parameters:
    ----------
    ds : xr.Dataset
        ERA5 dataset containing temperature, dewpoint, wind components, and pressure
        data for a specific time and location. Expected variables include 'T'
        (temperature), 'Q' (specific humidity), 'U' (zonal wind), 'V' (meridional wind),
        'Z' (geopotential height), and 'P' (pressure).
    fig : Optional[Figure], default=None
        Matplotlib figure object to use for plotting. If None, a new figure is created.
    subplot : Optional[Tuple[int, int, int]], default=None
        Tuple specifying (nrows, ncols, index) for subplot positioning within the
        figure. This defines the grid layout and the subplot’s position within it.
    rotation : int, default=40
        Rotation angle for the Skew-T plot lines (degrees).
    ptop : Quantity, default=100 * units.hPa
        The upper limit of pressure (in hPa) for the Skew-T plot. Data below this
        pressure level will be excluded from the plot.

    Returns:
    -------
    skew : skewT
    """

    # Validate required variables in dataset
    assert "Zsfc" in ds, "skewt needs geopotential height at the surface Zsfc"
    assert "SP" in ds, "skewt needs surface pressure SP"
    # Avoid plot_colormapped KeyError: 'Indexing with a boolean dask array is not allowed.
    # and allow .where function to check condition
    ds = ds.compute()
    # Drop low pressure model levels to avoid
    #  ValueError: ODE Integration failed likely due to small values of pressure.
    # Side effect of Dataset.where is that DataArrays without
    # a level dimension like SP are broadcast to all levels.
    ds = ds.where(ds.P >= ptop, drop=True)

    # if args.model_levels, ds["P"] is 3D DataArray with vertical dim = model level.
    # Otherwise pressure is ds.level, a 1-D array of pressure.
    height = ds["Z"]
    p = ds["P"]
    T = ds["T"]
    e = mpcalc.vapor_pressure(p, ds.Q)
    Td = mpcalc.dewpoint(e)

    if "Tv" not in ds:
        ds["Tv"] = mpcalc.thermo.virtual_temperature(ds.T, ds.Q)
    Tv = ds["Tv"]
    barb_increments = {"flag": 25, "full": 5, "half": 2.5}
    plot_barbs_units = "m/s"
    u = ds.U.metpy.convert_units(plot_barbs_units)
    v = ds.V.metpy.convert_units(plot_barbs_units)

    sfc = ds.level.max()
    skew = SkewT(fig, subplot=subplot, rotation=rotation)
    # Add the relevant special lines
    skew.plot_dry_adiabats(lw=0.75, alpha=0.5)
    skew.plot_moist_adiabats(lw=0.75, alpha=0.5)
    skew.plot_mixing_lines(alpha=0.5)
    # Slanted line on 0 isotherm
    skew.ax.axvline(0, color="c", linestyle="--", linewidth=2)

    # Calculate LCL pressure and label it on SkewT.
    lcl_pressure, lcl_temperature = mpcalc.lcl(
        p.sel(level=sfc), T.sel(level=sfc), Td.sel(level=sfc)
    )
    logging.info(f"lcl_pressure {lcl_pressure} lcl_temperature {lcl_temperature}")

    trans = transforms.blended_transform_factory(skew.ax.transAxes, skew.ax.transData)
    skew.ax.plot(
        [0.86, 0.88], 2 * [lcl_pressure.m_as("hPa")], transform=trans, color="brown"
    )
    skew.ax.text(
        0.885,
        lcl_pressure,
        "LCL",
        transform=trans,
        horizontalalignment="left",
        verticalalignment="center",
        color="brown",
    )

    # Calculate full parcel profile with LCL.
    p_without_lcl = p  # remember so we can interpolate other variables later
    # Append LCL to pressure array.
    p = np.append(p.data, lcl_pressure)
    # Create reverse sorted array of pressure including LCL
    p[::-1].sort()
    # Interpolate other variables to p array (which now includes LCL).
    T, Td, u, v, height, Tv, Zsfc, SP = interpolate_1d(
        p,
        p_without_lcl,
        T,
        Td,
        u,
        v,
        height,
        Tv,
        ds.Zsfc,
        ds.SP,
    )

    # T and Td were already turned from DataArrays into simple Quantity arrays
    # so you can't use .sel method here.
    prof = mpcalc.parcel_profile(p, T[0], Td[0])
    prof_mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(
        p, prof, 1
    )  # saturated mixing ratio

    prof_mixing_ratio[p >= lcl_pressure] = ds.Q.sel(
        level=sfc
    ).item()  # unsaturated mixing ratio (constant up to LCL)
    # parcel virtual temperature
    profTv = mpcalc.virtual_temperature(prof, prof_mixing_ratio)

    # Plot temperature and dewpoint.
    skew.plot(p, T, "r")
    skew.plot(p, Td, "g")
    # Draw virtual temperature like temperature, but thin and dashed.
    skew.plot(p, Tv, "r", lw=0.5, linestyle="dashed")
    # Parcel virtual temperature
    skew.plot(p, profTv, "k", linewidth=1.5, linestyle="dashed")

    sfcape, sfcin = mpcalc.cape_cin(p, Tv, Td, profTv)
    # Shade areas of CAPE and CIN
    skew.shade_cin(p, Tv, profTv)
    skew.shade_cape(p, Tv, profTv)

    logging.info("work on winds and kinematics")
    storm_u = 0.0 * units("m/s")
    storm_v = 0.0 * units("m/s")
    right_mover, left_mover, wind_mean = mpcalc.bunkers_storm_motion(p, u, v, height)
    storm_u, storm_v = wind_mean
    srh03_pos, srh03_neg, srh03_tot = mpcalc.storm_relative_helicity(
        height, u, v, 3 * units.km, storm_u=storm_u, storm_v=storm_v
    )

    bbz = skew.plot_barbs(
        p,
        u,
        v,
        length=6,
        plot_units=plot_barbs_units,
        linewidth=0.6,
        xloc=1.05,
        barb_increments=barb_increments,
    )

    # Good bounds for aspect ratio
    skew.ax.set_xlim(-40, 55)
    skew.ax.set_ylim(None, ptop)

    title = f"{ds.time.data} {ds.longitude.data:.3f} {ds.latitude.data:.3f}"
    title += f"\nwind barbs and hodograph in {plot_barbs_units} {barb_increments}"
    title += f"\nsfcape={sfcape:~.0f}   sfcin={sfcin:~.0f}   storm_u={storm_u:~.1f}   storm_v={storm_v:~.1f}"
    title += f"\n0-3km srh+={srh03_pos:~.0f}   srh-={srh03_neg:~.0f}   srh(tot)={srh03_tot:~.0f}"
    skew.ax.set_title(title, fontsize="x-small")

    logging.warning("Create hodograph")
    ax_hod = inset_axes(skew.ax, "40%", "40%", loc=1)
    h = Hodograph(ax_hod, component_range=30.0)
    h.add_grid(increment=10, linewidth=0.75)
    ax_hod.set_xlabel("")
    ax_hod.set_ylabel("")

    agl = height - Zsfc
    label_hgts = [0, 1, 3, 6, 9, 12, 15] * units("km")

    # Label AGL intervals in hodograph and along the y-axis of the skewT.
    for label_hgt in label_hgts:
        (agl2p,) = interpolate_1d(label_hgt, agl, p)
        s = f"{label_hgt:~.0f}"
        if label_hgt == 0 * units.km:
            agl2p = SP[0]
            s = f"SFC ({Zsfc[0]:~.0f})"
        skew.ax.plot([0, 0.01], 2 * [agl2p.m_as("hPa")], transform=trans, color="brown")
        skew.ax.text(
            0.01,
            agl2p,
            s,
            transform=trans,
            horizontalalignment="left",
            verticalalignment="center",
            color="brown",
        )
        ax_hod.text(
            np.interp(label_hgt, agl, u),
            np.interp(label_hgt, agl, v),
            label_hgt.to("km").m,
            fontsize=7,
        )
    h.plot_colormapped(
        u,
        v,
        height,
        intervals=label_hgts,
        colors=["red", "red", "lime", "green", "blueviolet", "cyan"],
    )
    ax_hod.plot(storm_u, storm_v, "x")

    return skew


if __name__ == "__main__":
    main()
