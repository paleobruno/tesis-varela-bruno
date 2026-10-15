from pathlib import Path

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm

MERRA2_FILE = (
    Path.home()
    / "MERRA2_dust"
    / "processed"
    / "MERRA2_dust_reference_2001_2020.nc"
)

ERA5_FILE = (
    Path.home()
    / "ERA5_wind"
    / "processed"
    / "ERA5_wind_climatology_2001_2020.nc"
)

FIGURE_DIR = (
    Path.home()
    / "MERRA2_dust"
    / "processed"
    / "figures"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Leer MERRA-2 y ERA5
# -----------------------------------------------------------------------------

with xr.open_dataset(MERRA2_FILE) as ds_merra:
    lon_merra = ds_merra["lon"].values
    lat_merra = ds_merra["lat"].values
    f_ref = ds_merra["F_ref_g_m2_yr"].values

with xr.open_dataset(ERA5_FILE) as ds_era5:
    lon_era5 = ds_era5["longitude"].values
    lat_era5 = ds_era5["latitude"].values
    u_col = ds_era5["u_column_mean"].values
    v_col = ds_era5["v_column_mean"].values

# -----------------------------------------------------------------------------
# Crear figura y mapa base
# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(11, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent(
    [-80, -20, -60, -35],
    crs=ccrs.PlateCarree()
)

ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

# -----------------------------------------------------------------------------
# Fondo: deposición de polvo MERRA-2
# -----------------------------------------------------------------------------

mesh = ax.pcolormesh(
    lon_merra,
    lat_merra,
    f_ref,
    transform=ccrs.PlateCarree(),
    norm=LogNorm(
        vmin=f_ref.min(),
        vmax=f_ref.max()
    ),
    cmap="viridis",
    shading="auto"
)

cbar = plt.colorbar(
    mesh,
    ax=ax,
    orientation="vertical",
    pad=0.03,
    shrink=0.85
)

cbar.set_label(
    r"Dust deposition flux [g m$^{-2}$ yr$^{-1}$]"
)

# -----------------------------------------------------------------------------
# Flechas: viento medio de columna ERA5
# -----------------------------------------------------------------------------

STEP = 8

lon_q = lon_era5[::STEP]
lat_q = lat_era5[::STEP]
u_q = u_col[::STEP, ::STEP]
v_q = v_col[::STEP, ::STEP]

quiver = ax.quiver(
    lon_q,
    lat_q,
    u_q,
    v_q,
    transform=ccrs.PlateCarree(),
    scale=250,
    width=0.0022
)

ax.quiverkey(
    quiver,
    X=0.88,
    Y=-0.08,
    U=10,
    label="10 m s$^{-1}$",
    labelpos="E"
)

# -----------------------------------------------------------------------------
# Grilla, título y guardado
# -----------------------------------------------------------------------------

gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.5,
    linestyle="--",
    alpha=0.6
)

gl.top_labels = False
gl.right_labels = False

ax.set_title(
    "MERRA-2 dust deposition with ERA5 column-mean wind\n"
    "2001–2020 climatology"
)

output_figure = (
    FIGURE_DIR
    / "MERRA2_dust_with_ERA5_column_wind_2001_2020.png"
)

plt.tight_layout()
plt.savefig(
    output_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Figura guardada en: {output_figure}")