import numpy as np
import xarray as xr
from pathlib import Path

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
MERRA_FILE = (
    Path.home()
    / "MERRA2_dust"
    / "processed"
    / "MERRA2_reference_2001_2020_1deg.nc"
)

ERA5_FILE = (
    Path.home()
    / "ERA5_wind"
    / "processed"
    / "ERA5_wind_climatology_2001_2020.nc"
)

OUTPUT_DIR = (
    Path.home()
    / "MERRA2_dust"
    / "processed"
    / "physical_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "MERRA2_vs_ERA5_wind_correlations.txt"


# ----------------------------------------------------------------------
# Abrir datasets
# ----------------------------------------------------------------------
merra = xr.open_dataset(MERRA_FILE)
era5 = xr.open_dataset(ERA5_FILE)


# ----------------------------------------------------------------------
# Campo MERRA-2
# ----------------------------------------------------------------------
dust = merra["F_ref_g_m2_yr"]

# Ordenar coordenadas por seguridad
dust = dust.sortby("lat").sortby("lon")

lat_target = dust["lat"].values
lon_target = dust["lon"].values


# ----------------------------------------------------------------------
# Función para llevar ERA5 a la grilla MERRA-2 de 1° x 1°
# ----------------------------------------------------------------------
def regrid_era5(field):

    field = field.sortby("latitude").sortby("longitude")

    field_1deg = field.interp(
        latitude=lat_target,
        longitude=lon_target,
        method="linear"
    )

    return field_1deg.values


# ----------------------------------------------------------------------
# Preparar deposición
# ----------------------------------------------------------------------
dust_values = dust.values

if np.any(dust_values <= 0):
    raise ValueError(
        "MERRA-2 contiene valores <= 0 y no se puede calcular log10."
    )

dust_log = np.log10(dust_values)


# ----------------------------------------------------------------------
# Campos de viento a comparar
# ----------------------------------------------------------------------
wind_fields = {}

# Promedio vertical de columna
wind_fields["Column mean"] = regrid_era5(
    era5["wind_speed_column_mean"]
)

# Niveles individuales
levels = [850, 500, 300, 200]

for level in levels:

    field = era5["wind_speed_clim"].sel(
        pressure_level=level
    )

    wind_fields[f"{level} hPa"] = regrid_era5(field)


# ----------------------------------------------------------------------
# Correlaciones espaciales
#
# Calcularemos:
#
# 1) Pearson entre F_ref y velocidad del viento
# 2) Pearson entre log10(F_ref) y velocidad del viento
#
# La segunda es útil porque el campo de deposición es fuertemente
# asimétrico.
# ----------------------------------------------------------------------
results = []

for name, wind in wind_fields.items():

    if wind.shape != dust_values.shape:
        raise ValueError(
            f"Dimensiones incompatibles en {name}: "
            f"wind={wind.shape}, dust={dust_values.shape}"
        )

    mask = (
        np.isfinite(dust_values)
        & np.isfinite(dust_log)
        & np.isfinite(wind)
    )

    dust_flat = dust_values[mask]
    dust_log_flat = dust_log[mask]
    wind_flat = wind[mask]

    r_raw = np.corrcoef(
        dust_flat,
        wind_flat
    )[0, 1]

    r_log = np.corrcoef(
        dust_log_flat,
        wind_flat
    )[0, 1]

    mean_wind = np.mean(wind_flat)

    results.append(
        (
            name,
            mean_wind,
            r_raw,
            r_log
        )
    )


# ----------------------------------------------------------------------
# Crear tabla
# ----------------------------------------------------------------------
header = (
    "Comparación espacial MERRA-2 vs ERA5\n"
    "=====================================\n\n"
    "Periodo climatológico: 2001-2020\n"
    "Dominio: 80°W-20°W, 60°S-35°S\n"
    "Grilla de comparación: 1° x 1°\n\n"
    "Campo de viento       Mean wind [m/s]    r(F, wind)    r(log10(F), wind)\n"
    "------------------------------------------------------------------------\n"
)

lines = [header]

for name, mean_wind, r_raw, r_log in results:

    line = (
        f"{name:<20}"
        f"{mean_wind:>12.4f}"
        f"{r_raw:>15.6f}"
        f"{r_log:>21.6f}\n"
    )

    lines.append(line)


text = "".join(lines)

OUTPUT_FILE.write_text(text)


# ----------------------------------------------------------------------
# Mostrar resultados
# ----------------------------------------------------------------------
print()
print(text)

print("Guardado en:")
print(OUTPUT_FILE)


# ----------------------------------------------------------------------
# Cerrar datasets
# ----------------------------------------------------------------------
merra.close()
era5.close()
