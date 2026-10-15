from pathlib import Path

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BASE_DIR = Path.home() / "ERA5_wind"

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ERA5_wind_climatology_2001_2020.nc"
)

FIGURE_DIR = BASE_DIR / "processed" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

USE_COLUMN_MEAN = True

# -----------------------------------------------------------------------------
# Leer viento climatológico al nivel seleccionado
# -----------------------------------------------------------------------------

with xr.open_dataset(INPUT_FILE) as ds:
    lon = ds["longitude"].values
    lat = ds["latitude"].values

    u = ds["u_column_mean"].values
    v = ds["v_column_mean"].values
    speed = ds["wind_speed_column_mean"].values

# -----------------------------------------------------------------------------
# Crear figura y mapa base
# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(11, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([-80, -20, -60, -35], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

# -----------------------------------------------------------------------------
# Fondo: rapidez del viento
# -----------------------------------------------------------------------------

mesh = ax.pcolormesh(
    lon,
    lat,
    speed,
    transform=ccrs.PlateCarree(),
    shading="auto",
    cmap="viridis"
)

cbar = plt.colorbar(
    mesh,
    ax=ax,
    orientation="vertical",
    pad=0.03,
    shrink=0.85
)

cbar.set_label(r"Wind speed [m s$^{-1}$]")

# -----------------------------------------------------------------------------
# Flechas: dirección, sentido y magnitud del viento
# -----------------------------------------------------------------------------

STEP = 8

lon_q = lon[::STEP]
lat_q = lat[::STEP]
u_q = u[::STEP, ::STEP]
v_q = v[::STEP, ::STEP]

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
    "ERA5 pressure-weighted column mean wind\n"
    "1000–200 hPa, 2001–2020"
)

output_figure = (
    FIGURE_DIR
    / "ERA5_wind_vectors_column_mean_1000_200hPa_2001_2020.png"
)

plt.tight_layout()
plt.savefig(output_figure, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figura guardada en: {output_figure}")