from pathlib import Path

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm

BASE_DIR = Path.home() / "MERRA2_dust"
INPUT_FILE = BASE_DIR / "processed" / "MERRA2_dust_reference_2001_2020.nc"
FIGURE_DIR = BASE_DIR / "processed" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Leer campo de referencia
# -----------------------------------------------------------------------------

with xr.open_dataset(INPUT_FILE) as ds:
    lon = ds["lon"].values
    lat = ds["lat"].values
    f_ref = ds["F_ref"].values

# -----------------------------------------------------------------------------
# Crear mapa
# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(11, 7))

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

ax.set_extent(
    [-80, -20, -60, -35],
    crs=ccrs.PlateCarree()
)

ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

# -----------------------------------------------------------------------------
# Dibujar campo de deposición
# -----------------------------------------------------------------------------

mesh = ax.pcolormesh(
    lon,
    lat,
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
    r"Dust deposition flux [kg m$^{-2}$ s$^{-1}$]"
)

# -----------------------------------------------------------------------------
# Coordenadas, título y guardado
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
    "MERRA-2 Dust Deposition Reference Field\n"
    "2001–2020 climatological mean"
)

output_figure = FIGURE_DIR / "MERRA2_dust_reference_2001_2020.png"

plt.tight_layout()
plt.savefig(output_figure, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figura guardada en: {output_figure}")