from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm

BASE_DIR = Path.home() / "MERRA2_dust"

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "MERRA2_dust_reference_2001_2020.nc"
)

OUTPUT_DIR = BASE_DIR / "processed" / "kriging_samples"
FIGURE_DIR = BASE_DIR / "processed" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 100
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Leer campo de referencia y convertir la grilla en puntos
# -----------------------------------------------------------------------------

with xr.open_dataset(INPUT_FILE) as ds:
    lon = ds["lon"].values
    lat = ds["lat"].values
    f_ref = ds["F_ref_g_m2_yr"].values

# Crear coordenadas 2D para cada celda de la grilla
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Convertir todo a vectores 1D
lon_all = lon_grid.ravel()
lat_all = lat_grid.ravel()
f_all = f_ref.ravel()

# Verificación básica
if not (
    len(lon_all) == len(lat_all) == len(f_all)
):
    raise RuntimeError("Las coordenadas y los valores no tienen la misma longitud.")

if np.isnan(f_all).any():
    raise RuntimeError("El campo F_ref contiene valores NaN.")

print(f"Celdas disponibles para muestreo: {len(f_all)}")

# -----------------------------------------------------------------------------
# Muestreo aleatorio reproducible
# -----------------------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

if N_SAMPLES > len(f_all):
    raise RuntimeError(
        f"Se solicitaron {N_SAMPLES} puntos, "
        f"pero solo existen {len(f_all)} celdas disponibles."
    )

sample_indices = rng.choice(
    len(f_all),
    size=N_SAMPLES,
    replace=False
)

lon_sample = lon_all[sample_indices]
lat_sample = lat_all[sample_indices]
f_sample = f_all[sample_indices]

# -----------------------------------------------------------------------------
# Construir tabla de la muestra
# -----------------------------------------------------------------------------

sample_data = np.column_stack(
    (
        lon_sample,
        lat_sample,
        f_sample
    )
)

print(f"Puntos seleccionados: {len(f_sample)}")
print(f"Semilla aleatoria: {RANDOM_SEED}")

# -----------------------------------------------------------------------------
# Visualizar muestra sobre el campo completo
# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(11, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent(
    [-80, -20, -60, -35],
    crs=ccrs.PlateCarree()
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

# Campo completo MERRA-2 como fondo
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

# Los 100 puntos que "sobreviven"
ax.scatter(
    lon_sample,
    lat_sample,
    s=32,
    facecolors="none",
    edgecolors="black",
    linewidths=1.0,
    transform=ccrs.PlateCarree(),
    zorder=3
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

gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.5,
    linestyle="--",
    alpha=0.6
)

gl.top_labels = False
gl.right_labels = False

ax.set_title(
    f"MERRA-2 reference field with {N_SAMPLES} sampled cells\n"
    f"Random seed = {RANDOM_SEED}"
)

output_figure = (
    FIGURE_DIR
    / f"MERRA2_sample_N{N_SAMPLES}_seed{RANDOM_SEED}.png"
)

plt.tight_layout()
plt.savefig(
    output_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Figura guardada en: {output_figure}")

# -----------------------------------------------------------------------------
# Exportar muestra para Global-Krigger
# -----------------------------------------------------------------------------

gk_file = (
    OUTPUT_DIR
    / f"MERRA2_dust_N{N_SAMPLES}_seed{RANDOM_SEED}.txt"
)

np.savetxt(
    gk_file,
    sample_data,
    fmt=["%.6f", "%.6f", "%.10e"],
    delimiter="\t"
)

print(f"Archivo Global-Krigger guardado en: {gk_file}")

# Guardar índices originales de las celdas seleccionadas
index_file = (
    OUTPUT_DIR
    / f"MERRA2_dust_N{N_SAMPLES}_seed{RANDOM_SEED}_indices.txt"
)

np.savetxt(
    index_file,
    sample_indices,
    fmt="%d"
)

print(f"Índices guardados en: {index_file}")

# -----------------------------------------------------------------------------
# Exportar también versión log10 para Global-Krigger
# -----------------------------------------------------------------------------

log_sample_data = np.column_stack(
    (
        lon_sample,
        lat_sample,
        np.log10(f_sample)
    )
)

gk_log_file = (
    OUTPUT_DIR
    / f"MERRA2_dust_N{N_SAMPLES}_seed{RANDOM_SEED}_log10.txt"
)

np.savetxt(
    gk_log_file,
    log_sample_data,
    fmt=["%.6f", "%.6f", "%.10e"],
    delimiter="\t"
)

print(f"Archivo Global-Krigger log10 guardado en: {gk_log_file}")