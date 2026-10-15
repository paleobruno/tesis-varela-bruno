import csv
from pathlib import Path

import numpy as np
import xarray as xr


# ======================================================================
# 1. RUTAS
# ======================================================================

WIND_FILE = (
    Path.home()
    / "ERA5_wind"
    / "raw"
    / "1d1642ad6f8637e07120efb9033141b9.nc"
)

SP_FILE = (
    Path.home()
    / "ERA5_wind"
    / "surface_pressure"
    / "raw"
    / "276c4c582675dfdcd278ca2188f42f42.nc"
)

OUTPUT_DIR = (
    Path.home()
    / "ERA5_wind"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_NC = (
    OUTPUT_DIR
    / "ERA5_wind_climatology_2001_2020_terrain_masked.nc"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "ERA5_terrain_mask_diagnostics.csv"
)


# ======================================================================
# 2. ABRIR ARCHIVOS
# ======================================================================

wind = xr.open_dataset(WIND_FILE)
sp_ds = xr.open_dataset(SP_FILE)


# ======================================================================
# 3. COMPROBACIONES
# ======================================================================

required_wind_vars = ["u", "v"]

for var in required_wind_vars:
    if var not in wind:
        raise ValueError(
            f"No se encontró la variable '{var}' "
            f"en el archivo de viento."
        )

if "sp" not in sp_ds:
    raise ValueError(
        "No se encontró la variable 'sp' "
        "en el archivo de presión superficial."
    )

if "pressure_level" not in wind.coords:
    raise ValueError(
        "No se encontró la coordenada 'pressure_level'."
    )


if not np.array_equal(
    wind["valid_time"].values,
    sp_ds["valid_time"].values
):
    raise ValueError(
        "Los tiempos de viento y presión superficial no coinciden."
    )

if not np.array_equal(
    wind["latitude"].values,
    sp_ds["latitude"].values
):
    raise ValueError(
        "Las latitudes no coinciden."
    )

if not np.array_equal(
    wind["longitude"].values,
    sp_ds["longitude"].values
):
    raise ValueError(
        "Las longitudes no coinciden."
    )


# ======================================================================
# 4. CONVERTIR PRESIÓN SUPERFICIAL
#
# ERA5 entrega:
#
#       sp [Pa]
#
# Los niveles están en:
#
#       pressure_level [hPa]
#
# Entonces:
#
#       p_s [hPa] = sp / 100
# ======================================================================

sp_hpa = sp_ds["sp"] / 100.0

sp_hpa.attrs["units"] = "hPa"
sp_hpa.attrs["long_name"] = "Surface pressure"


# ======================================================================
# 5. CONSTRUIR MÁSCARA FÍSICA
#
# Un nivel de presión p existe sobre el terreno si:
#
#                 p <= p_s
#
# Ejemplo:
#
#   p_s = 780 hPa
#
#   850 hPa  -> bajo superficie -> inválido
#   700 hPa  -> atmósfera       -> válido
#
# Xarray realiza automáticamente el broadcasting entre:
#
# p(level)
#
# y
#
# p_s(time, lat, lon)
# ======================================================================

pressure = wind["pressure_level"]

valid = (
    pressure <= sp_hpa
)

valid = valid.transpose(
    "valid_time",
    "pressure_level",
    "latitude",
    "longitude"
)


# ======================================================================
# 6. PESOS TEMPORALES
#
# No todos los meses tienen igual duración.
#
# Una media de enero debe representar 31 días;
# febrero 28 o 29 días, etc.
# ======================================================================

days = wind["valid_time"].dt.days_in_month.astype(
    np.float64
)

total_days = days.sum(
    dim="valid_time"
)


# ======================================================================
# 7. DÍAS VÁLIDOS EN CADA NIVEL Y CELDA
#
# Si un nivel está bajo superficie durante un mes,
# ese mes aporta peso cero.
# ======================================================================

valid_days = (
    valid.astype(np.float64)
    * days
)

sum_valid_days = valid_days.sum(
    dim="valid_time"
)


# ======================================================================
# 8. CLIMATOLOGÍA TOPOGRÁFICAMENTE CORREGIDA
#
#             sum_t d_t M_t u_t
# u_bar = ----------------------------
#             sum_t d_t M_t
#
# donde:
#
# M_t = 1 si el nivel existe físicamente
#       0 si está bajo superficie
# ======================================================================

u_masked = wind["u"].where(
    valid
)

v_masked = wind["v"].where(
    valid
)


u_numerator = (
    u_masked
    * days
).sum(
    dim="valid_time",
    skipna=True
)

v_numerator = (
    v_masked
    * days
).sum(
    dim="valid_time",
    skipna=True
)


u_clim = (
    u_numerator
    / sum_valid_days
)

v_clim = (
    v_numerator
    / sum_valid_days
)


# Si un nivel nunca existe en una celda,
# el denominador es cero y esa celda debe quedar como NaN.

u_clim = u_clim.where(
    sum_valid_days > 0
)

v_clim = v_clim.where(
    sum_valid_days > 0
)


# ======================================================================
# 9. MAGNITUD DEL VECTOR CLIMATOLÓGICO
# ======================================================================

wind_speed_clim = np.sqrt(
    u_clim**2
    + v_clim**2
)


# ======================================================================
# 10. FRACCIÓN DE VALIDEZ
#
# Guardamos dos medidas:
#
# A) fraction_months:
#    fracción de los 240 meses en que el nivel existe.
#
# B) fraction_days:
#    misma idea, pero ponderada por duración de los meses.
# ======================================================================

valid_fraction_months = valid.mean(
    dim="valid_time"
)

valid_fraction_days = (
    sum_valid_days
    / total_days
)


# ======================================================================
# 11. PRESIÓN SUPERFICIAL CLIMATOLÓGICA
# ======================================================================

sp_clim = (
    sp_hpa
    * days
).sum(
    dim="valid_time"
) / total_days


# ======================================================================
# 12. CONSTRUIR DATASET DE SALIDA
# ======================================================================

out = xr.Dataset(
    data_vars={
        "u_clim": u_clim,
        "v_clim": v_clim,
        "wind_speed_clim": wind_speed_clim,
        "valid_fraction_months": valid_fraction_months,
        "valid_fraction_days": valid_fraction_days,
        "surface_pressure_clim": sp_clim,
    }
)


# ======================================================================
# 13. ATRIBUTOS
# ======================================================================

out["u_clim"].attrs = {
    "long_name":
        "Terrain-masked climatological zonal wind",
    "units":
        "m s-1",
}

out["v_clim"].attrs = {
    "long_name":
        "Terrain-masked climatological meridional wind",
    "units":
        "m s-1",
}

out["wind_speed_clim"].attrs = {
    "long_name":
        "Magnitude of terrain-masked climatological wind vector",
    "units":
        "m s-1",
}

out["valid_fraction_months"].attrs = {
    "long_name":
        "Fraction of months for which pressure level is above surface",
    "units":
        "1",
}

out["valid_fraction_days"].attrs = {
    "long_name":
        "Day-weighted fraction for which pressure level is above surface",
    "units":
        "1",
}

out["surface_pressure_clim"].attrs = {
    "long_name":
        "2001-2020 day-weighted climatological surface pressure",
    "units":
        "hPa",
}


out.attrs = {
    "description":
        "ERA5 2001-2020 wind climatology with pressure levels "
        "below the monthly mean surface pressure removed.",
    "terrain_mask_condition":
        "pressure_level_hPa <= surface_pressure_hPa",
    "temporal_weighting":
        "Monthly means weighted by number of days per month.",
    "period":
        "2001-2020",
}


# ======================================================================
# 14. GUARDAR NETCDF
# ======================================================================

out.to_netcdf(
    OUTPUT_NC
)


# ======================================================================
# 15. DIAGNÓSTICOS POR NIVEL
#
# Nos interesa distinguir:
#
# all_months:
#   porcentaje del dominio donde ese nivel existe los 240 meses.
#
# partial:
#   porcentaje donde existe solo parte del período.
#
# never:
#   porcentaje donde nunca existe y, por tanto,
#   corresponde persistentemente al subsuelo.
#
# mean_valid:
#   disponibilidad temporal media del nivel
#   sobre todo el dominio.
# ======================================================================

diagnostic_rows = []

levels = wind["pressure_level"].values


for p in levels:

    vf = valid_fraction_months.sel(
        pressure_level=p
    ).values

    mean_valid = (
        np.nanmean(vf)
        * 100.0
    )

    all_months = (
        np.mean(
            np.isclose(vf, 1.0)
        )
        * 100.0
    )

    never = (
        np.mean(
            np.isclose(vf, 0.0)
        )
        * 100.0
    )

    partial = (
        100.0
        - all_months
        - never
    )

    diagnostic_rows.append(
        [
            float(p),
            mean_valid,
            all_months,
            partial,
            never,
        ]
    )


with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "pressure_hPa",
            "mean_temporal_validity_percent",
            "domain_valid_all_months_percent",
            "domain_partially_valid_percent",
            "domain_never_valid_percent",
        ]
    )

    writer.writerows(
        diagnostic_rows
    )


# ======================================================================
# 16. MOSTRAR RESUMEN
# ======================================================================

print()
print(
    "Climatología corregida por topografía"
)
print(
    "====================================="
)
print()

print(
    "Nivel   Validez media   "
    "Siempre válido   Parcial   Nunca válido"
)

print(
    "hPa          [%]             "
    "[%]          [%]          [%]"
)

print(
    "-----------------------------------------------------------"
)


for row in diagnostic_rows:

    print(
        f"{row[0]:4.0f}     "
        f"{row[1]:8.3f}        "
        f"{row[2]:8.3f}     "
        f"{row[3]:8.3f}     "
        f"{row[4]:8.3f}"
    )


print()
print(
    "Archivo NetCDF:"
)
print(
    OUTPUT_NC
)

print()
print(
    "Diagnóstico CSV:"
)
print(
    OUTPUT_CSV
)


# ======================================================================
# 17. CERRAR ARCHIVOS
# ======================================================================

wind.close()
sp_ds.close()
