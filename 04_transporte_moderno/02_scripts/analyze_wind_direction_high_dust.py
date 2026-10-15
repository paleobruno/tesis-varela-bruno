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

OUTPUT_FILE = OUTPUT_DIR / "wind_direction_high_dust_top10.txt"
MASK_FILE = OUTPUT_DIR / "high_dust_top10_mask.txt"


# ----------------------------------------------------------------------
# Abrir datasets
# ----------------------------------------------------------------------
merra = xr.open_dataset(MERRA_FILE)
era5 = xr.open_dataset(ERA5_FILE)


# ----------------------------------------------------------------------
# Campo de deposición MERRA-2
# ----------------------------------------------------------------------
dust = merra["F_ref_g_m2_yr"]

dust = dust.sortby("lat").sortby("lon")

lat_target = dust["lat"].values
lon_target = dust["lon"].values
dust_values = dust.values


# ----------------------------------------------------------------------
# Seleccionar el 10 % de celdas con mayor deposición
# ----------------------------------------------------------------------
threshold = np.percentile(dust_values, 90.0)

high_dust_mask = dust_values >= threshold

n_total = dust_values.size
n_high = np.sum(high_dust_mask)

np.savetxt(
    MASK_FILE,
    high_dust_mask.astype(int),
    delimiter=",",
    fmt="%d"
)


# ----------------------------------------------------------------------
# Función para llevar ERA5 a grilla MERRA-2 1° x 1°
# ----------------------------------------------------------------------
def regrid_era5(field):

    field = field.sortby("latitude").sortby("longitude")

    return field.interp(
        latitude=lat_target,
        longitude=lon_target,
        method="linear"
    ).values


# ----------------------------------------------------------------------
# Función para calcular estadísticas vectoriales
#
# u > 0 : hacia el este
# v > 0 : hacia el norte
#
# El ángulo "transport_direction" indica HACIA DÓNDE se mueve el flujo:
#
#   0°   = Norte
#   90°  = Este
#   180° = Sur
#   270° = Oeste
# ----------------------------------------------------------------------
def analyze_vector(u, v):

    u_sel = u[high_dust_mask]
    v_sel = v[high_dust_mask]

    mean_u = np.mean(u_sel)
    mean_v = np.mean(v_sel)

    vector_speed = np.sqrt(
        mean_u**2 + mean_v**2
    )

    mean_local_speed = np.mean(
        np.sqrt(u_sel**2 + v_sel**2)
    )

    transport_direction = (
        np.degrees(
            np.arctan2(mean_u, mean_v)
        )
        + 360.0
    ) % 360.0

    # Dirección meteorológica:
    # de dónde viene el viento
    wind_from_direction = (
        transport_direction + 180.0
    ) % 360.0

    return (
        mean_u,
        mean_v,
        vector_speed,
        mean_local_speed,
        transport_direction,
        wind_from_direction
    )


# ----------------------------------------------------------------------
# Campos a analizar
# ----------------------------------------------------------------------
wind_fields = {}

# Promedio de columna
wind_fields["Column mean"] = (
    regrid_era5(era5["u_column_mean"]),
    regrid_era5(era5["v_column_mean"])
)

# Niveles individuales
levels = [850, 500, 300, 200]

for level in levels:

    u = era5["u_clim"].sel(
        pressure_level=level
    )

    v = era5["v_clim"].sel(
        pressure_level=level
    )

    wind_fields[f"{level} hPa"] = (
        regrid_era5(u),
        regrid_era5(v)
    )


# ----------------------------------------------------------------------
# Calcular resultados
# ----------------------------------------------------------------------
results = []

for name, (u, v) in wind_fields.items():

    stats = analyze_vector(u, v)

    results.append(
        (name,) + stats
    )


# ----------------------------------------------------------------------
# Crear salida
# ----------------------------------------------------------------------
header = f"""
Análisis vectorial ERA5 en celdas de alta deposición MERRA-2
=============================================================

Periodo climatológico: 2001-2020
Dominio: 80°W-20°W, 60°S-35°S
Grilla: 1° x 1°

Selección:
10 % de celdas con mayor deposición

Umbral de deposición:
{threshold:.6f} g m^-2 yr^-1

Número total de celdas:
{n_total}

Número de celdas seleccionadas:
{n_high}

Convención angular:
0°   = Norte
90°  = Este
180° = Sur
270° = Oeste

"Transport dir." = hacia dónde se mueve el flujo
"Wind from dir." = de dónde viene el viento


Campo             mean_u    mean_v    |mean V|   mean |V|   Transport dir.   Wind from dir.
                    m/s       m/s        m/s        m/s          deg               deg
------------------------------------------------------------------------------------------------
"""

lines = [header]

for result in results:

    (
        name,
        mean_u,
        mean_v,
        vector_speed,
        mean_local_speed,
        transport_direction,
        wind_from_direction
    ) = result

    line = (
        f"{name:<16}"
        f"{mean_u:>9.3f}"
        f"{mean_v:>10.3f}"
        f"{vector_speed:>11.3f}"
        f"{mean_local_speed:>11.3f}"
        f"{transport_direction:>17.2f}"
        f"{wind_from_direction:>18.2f}\n"
    )

    lines.append(line)


text = "".join(lines)

OUTPUT_FILE.write_text(text)


# ----------------------------------------------------------------------
# Mostrar
# ----------------------------------------------------------------------
print()
print(text)

print("Archivos guardados en:")
print(OUTPUT_FILE)
print(MASK_FILE)


# ----------------------------------------------------------------------
# Cerrar datasets
# ----------------------------------------------------------------------
merra.close()
era5.close()
