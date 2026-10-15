import numpy as np
import xarray as xr
from pathlib import Path

# ----------------------------------------------------------------------
# Parámetros obtenidos del ajuste exponencial downwind
# ----------------------------------------------------------------------
L_km = 1267.49
L_error_km = 72.77

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
BASE = Path.home() / "MERRA2_dust" / "processed"
PHYS = BASE / "physical_analysis"

ERA5_FILE = (
    Path.home()
    / "ERA5_wind"
    / "processed"
    / "ERA5_wind_climatology_2001_2020.nc"
)

MERRA_FILE = BASE / "MERRA2_reference_2001_2020_1deg.nc"

MASK_FILE = PHYS / "high_dust_top10_mask.txt"

OUTPUT_FILE = PHYS / "effective_decay_timescales.txt"


# ----------------------------------------------------------------------
# Abrir datos
# ----------------------------------------------------------------------
era5 = xr.open_dataset(ERA5_FILE)
merra = xr.open_dataset(MERRA_FILE)

mask = np.loadtxt(
    MASK_FILE,
    delimiter=","
).astype(bool)


# ----------------------------------------------------------------------
# Coordenadas objetivo: misma grilla MERRA-2 1° x 1°
# ----------------------------------------------------------------------
dust = (
    merra["F_ref_g_m2_yr"]
    .sortby("lat")
    .sortby("lon")
)

lat_target = dust["lat"].values
lon_target = dust["lon"].values


# ----------------------------------------------------------------------
# Función para llevar ERA5 a la grilla de 1°
# ----------------------------------------------------------------------
def regrid_era5(field):

    field = (
        field
        .sortby("latitude")
        .sortby("longitude")
    )

    return field.interp(
        latitude=lat_target,
        longitude=lon_target,
        method="linear"
    ).values


# ----------------------------------------------------------------------
# Extraer velocidad zonal U en zona de alta deposición
# ----------------------------------------------------------------------
wind_fields = {}

# Promedio de columna
wind_fields["Column mean"] = regrid_era5(
    era5["u_column_mean"]
)

# Niveles individuales
levels = [850, 500, 300, 200]

for level in levels:

    u = era5["u_clim"].sel(
        pressure_level=level
    )

    wind_fields[f"{level} hPa"] = regrid_era5(u)


# ----------------------------------------------------------------------
# Cálculo:
#
# tau = L / U
# lambda = U / L = 1 / tau
#
# IMPORTANTE:
# L se convierte de km a m.
#
# La incertidumbre propagada aquí considera SOLAMENTE
# la incertidumbre formal del ajuste de L.
# ----------------------------------------------------------------------
L_m = L_km * 1000.0
L_error_m = L_error_km * 1000.0

results = []

for name, u_field in wind_fields.items():

    U = np.mean(
        u_field[mask]
    )

    if U <= 0:
        raise ValueError(
            f"U <= 0 para {name}. "
            "La aproximación downwind hacia el este no sería válida."
        )

    # Tiempo efectivo [s]
    tau_s = L_m / U

    # Tasa efectiva [s^-1]
    lambda_s = U / L_m

    # Conversión a días
    tau_days = tau_s / 86400.0

    # lambda en día^-1
    lambda_day = lambda_s * 86400.0

    # Propagación de error considerando solo sigma_L
    relative_error_L = L_error_m / L_m

    tau_error_days = tau_days * relative_error_L
    lambda_error_day = lambda_day * relative_error_L

    results.append(
        (
            name,
            U,
            tau_days,
            tau_error_days,
            lambda_day,
            lambda_error_day
        )
    )


# ----------------------------------------------------------------------
# Tabla de salida
# ----------------------------------------------------------------------
header = f"""
Escalas efectivas de decaimiento downwind
==========================================

Modelo:
F(x) = F0 exp(-x/L)

L = {L_km:.2f} +/- {L_error_km:.2f} km

Relaciones:

tau_eff = L / U

lambda_eff = U / L = 1 / tau_eff

U corresponde a la componente zonal media del viento
en las celdas pertenecientes al top 10 % de deposición.

IMPORTANTE:
tau_eff es una escala temporal efectiva asociada al
decaimiento espacial observado. No debe interpretarse
todavía como un tiempo de residencia atmosférico literal.

La incertidumbre mostrada considera solamente
la incertidumbre formal de L.


Campo              U [m/s]    tau [days]    sigma_tau    lambda [day^-1]    sigma_lambda
-----------------------------------------------------------------------------------------
"""

lines = [header]

for (
    name,
    U,
    tau_days,
    tau_error_days,
    lambda_day,
    lambda_error_day
) in results:

    line = (
        f"{name:<16}"
        f"{U:>10.3f}"
        f"{tau_days:>14.3f}"
        f"{tau_error_days:>13.3f}"
        f"{lambda_day:>18.5f}"
        f"{lambda_error_day:>16.5f}\n"
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
# Cerrar archivos
# ----------------------------------------------------------------------
era5.close()
merra.close()
