import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
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

OUTPUT_FIG = OUTPUT_DIR / "MERRA2_zonal_dust_profile.png"
OUTPUT_TXT = PHYS / "MERRA2_zonal_dust_profile.txt"


# ----------------------------------------------------------------------
# Leer datos
# ----------------------------------------------------------------------
ds = xr.open_dataset(MERRA_FILE)

dust = (
    ds["F_ref_g_m2_yr"]
    .sortby("lat")
    .sortby("lon")
)

lat = dust["lat"].values
lon = dust["lon"].values
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
# Identificar automáticamente la banda latitudinal principal
#
# Seleccionamos todas las latitudes que contienen al menos una celda
# perteneciente al top 10 % de deposición.
# ----------------------------------------------------------------------
rows_with_high_dust = np.any(mask, axis=1)

lat_band = lat[rows_with_high_dust]

lat_min = lat_band.min()
lat_max = lat_band.max()

band_mask = (
    (lat >= lat_min)
    & (lat <= lat_max)
)


# ----------------------------------------------------------------------
# Perfil zonal medio
#
# Para cada longitud promediamos la deposición sobre la banda latitudinal.
# ----------------------------------------------------------------------
dust_band = values[band_mask, :]

zonal_mean = np.mean(
    dust_band,
    axis=0
)

zonal_std = np.std(
    dust_band,
    axis=0
)


# ----------------------------------------------------------------------
# Identificar máximo del perfil
# ----------------------------------------------------------------------
imax = np.argmax(zonal_mean)

peak_lon = lon[imax]
peak_value = zonal_mean[imax]


# ----------------------------------------------------------------------
# Guardar datos
# ----------------------------------------------------------------------
output_data = np.column_stack(
    (
        lon,
        zonal_mean,
        zonal_std
    )
)

header = (
    "longitude_deg,"
    "mean_dust_g_m2_yr,"
    "std_dust_g_m2_yr"
)

np.savetxt(
    OUTPUT_TXT,
    output_data,
    delimiter=",",
    header=header,
    comments="",
    fmt="%.8f"
)


# ----------------------------------------------------------------------
# Gráfico
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    lon,
    zonal_mean,
    linewidth=2,
    label="Latitudinal mean"
)

ax.fill_between(
    lon,
    zonal_mean - zonal_std,
    zonal_mean + zonal_std,
    alpha=0.20,
    label="±1 spatial SD"
)

# Longitud del máximo
ax.axvline(
    peak_lon,
    linestyle="--",
    linewidth=1.5,
    label=f"Maximum at {peak_lon:.1f}°"
)

ax.scatter(
    peak_lon,
    peak_value,
    s=50,
    zorder=3
)

ax.set_xlabel("Longitude")
ax.set_ylabel(
    "Dust deposition [g m$^{-2}$ yr$^{-1}$]"
)

ax.set_title(
    "MERRA-2 zonal dust-deposition profile\n"
    f"Latitude band: {lat_min:.1f}° to {lat_max:.1f}°"
)

ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_FIG,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ----------------------------------------------------------------------
# Mostrar resultados
# ----------------------------------------------------------------------
print()
print("Perfil zonal generado correctamente")
print("-----------------------------------")
print(
    f"Banda latitudinal : "
    f"{lat_min:.1f}° a {lat_max:.1f}°"
)
print(
    f"Longitud del máximo : "
    f"{peak_lon:.1f}°"
)
print(
    f"Deposición media máxima : "
    f"{peak_value:.6f} g m^-2 yr^-1"
)
print()
print("Archivos guardados en:")
print(OUTPUT_TXT)
print(OUTPUT_FIG)

ds.close()
