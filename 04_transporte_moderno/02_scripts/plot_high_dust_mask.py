import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
BASE = Path.home() / "MERRA2_dust" / "processed"
PHYS = BASE / "physical_analysis"

MERRA_FILE = BASE / "MERRA2_reference_2001_2020_1deg.nc"
MASK_FILE = PHYS / "high_dust_top10_mask.txt"

OUTPUT_DIR = PHYS / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "MERRA2_high_dust_top10.png"


# ----------------------------------------------------------------------
# Leer datos
# ----------------------------------------------------------------------
ds = xr.open_dataset(MERRA_FILE)

dust = ds["F_ref_g_m2_yr"].sortby("lat").sortby("lon")

lon = dust["lon"].values
lat = dust["lat"].values
values = dust.values

mask = np.loadtxt(
    MASK_FILE,
    delimiter=","
).astype(bool)


# ----------------------------------------------------------------------
# Comprobar dimensiones
# ----------------------------------------------------------------------
if mask.shape != values.shape:
    raise ValueError(
        f"Dimensiones incompatibles: "
        f"dust={values.shape}, mask={mask.shape}"
    )


# ----------------------------------------------------------------------
# Coordenadas
# ----------------------------------------------------------------------
LON, LAT = np.meshgrid(lon, lat)


# ----------------------------------------------------------------------
# Gráfico
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

m = ax.pcolormesh(
    LON,
    LAT,
    values,
    shading="auto",
    cmap="viridis",
    norm=LogNorm(
        vmin=values.min(),
        vmax=values.max()
    )
)

cbar = fig.colorbar(m, ax=ax)

cbar.set_label(
    "Dust deposition [g m$^{-2}$ yr$^{-1}$]"
)

# Celdas pertenecientes al top 10 %
ax.scatter(
    LON[mask],
    LAT[mask],
    s=28,
    facecolors="none",
    edgecolors="black",
    linewidths=0.8,
    label="Top 10% dust deposition"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_title(
    "MERRA-2 dust deposition: highest 10% of grid cells"
)

ax.set_xlim(-80, -20)
ax.set_ylim(-60, -35)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ----------------------------------------------------------------------
# Información
# ----------------------------------------------------------------------
print()
print("Mapa generado correctamente")
print("----------------------------")
print(f"Celdas seleccionadas : {np.sum(mask)}")
print()
print("Guardado en:")
print(OUTPUT_FILE)

ds.close()
