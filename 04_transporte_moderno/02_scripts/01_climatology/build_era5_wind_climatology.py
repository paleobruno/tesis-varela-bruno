from pathlib import Path

import numpy as np
import xarray as xr

BASE_DIR = Path.home() / "ERA5_wind"

INPUT_FILE = BASE_DIR / "raw" / "1d1642ad6f8637e07120efb9033141b9.nc"
OUTPUT_DIR = BASE_DIR / "processed"

#####################################################################
################# ABRIR ERA5 Y VERIFICAR ESTRUCTURA #################
#####################################################################

EXPECTED_LEVELS = np.array([
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750,
    700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200
], dtype=float)

ds = xr.open_dataset(INPUT_FILE)

if ds.sizes["valid_time"] != 240:
    raise RuntimeError(
        f"Se esperaban 240 meses, pero se encontraron {ds.sizes['valid_time']}."
    )

if ds.sizes["pressure_level"] != 23:
    raise RuntimeError(
        f"Se esperaban 23 niveles de presión, pero se encontraron "
        f"{ds.sizes['pressure_level']}."
    )

if not np.array_equal(ds["pressure_level"].values, EXPECTED_LEVELS):
    raise RuntimeError(
        "Los niveles de presión no coinciden con los esperados."
    )

if "u" not in ds or "v" not in ds:
    raise RuntimeError("No se encontraron las variables u y v.")

print("ERA5 abierto correctamente.")
print(f"Meses: {ds.sizes['valid_time']}")
print(f"Niveles de presión: {ds.sizes['pressure_level']}")
print(
    f"Período: {ds['valid_time'].values[0]} "
    f"-> {ds['valid_time'].values[-1]}"
)

#####################################################################
############ CLIMATOLOGÍA 2001-2020 POR NIVEL DE PRESIÓN ############
#####################################################################

weighted_u = None
weighted_v = None
total_days = 0

for i in range(ds.sizes["valid_time"]):

    # Número de días del mes correspondiente
    days = int(ds["valid_time"].dt.days_in_month.isel(valid_time=i).values)

    # Mantener pressure_level, latitude y longitude;
    # eliminar solamente la dimensión temporal
    u_month = ds["u"].isel(valid_time=i, drop=True).load()
    v_month = ds["v"].isel(valid_time=i, drop=True).load()

    if weighted_u is None:
        weighted_u = u_month * days
        weighted_v = v_month * days
    else:
        weighted_u += u_month * days
        weighted_v += v_month * days

    total_days += days

if total_days != 7305:
    raise RuntimeError(
        f"Se esperaban 7305 días entre 2001 y 2020, "
        f"pero se obtuvieron {total_days}."
    )

u_clim = weighted_u / total_days
v_clim = weighted_v / total_days

print(f"Días utilizados en la climatología: {total_days}")
print("Climatología por nivel calculada correctamente.")

#####################################################################
############ RAPIDEZ DEL VIENTO CLIMATOLÓGICO POR NIVEL #############
#####################################################################

wind_speed_clim = np.sqrt(u_clim**2 + v_clim**2)

wind_speed_clim.attrs = {
    "long_name": "2001-2020 climatological wind speed",
    "units": "m s-1",
}

#####################################################################
########### VIENTO MEDIO DE COLUMNA ENTRE 1000 Y 200 hPa ############
#####################################################################

# Ordenar los niveles de presión de 200 a 1000 hPa para realizar
# correctamente la integración numérica.
u_sorted = u_clim.sortby("pressure_level")
v_sorted = v_clim.sortby("pressure_level")

P_TOP = 200.0
P_BOTTOM = 1000.0
DELTA_P = P_BOTTOM - P_TOP

# Integral en presión dividida por el espesor de la capa.
# El resultado conserva unidades de m s-1.
u_column_mean = (
    u_sorted.integrate(coord="pressure_level") / DELTA_P
)

v_column_mean = (
    v_sorted.integrate(coord="pressure_level") / DELTA_P
)

# Magnitud del vector medio de columna
wind_speed_column_mean = np.sqrt(
    u_column_mean**2 + v_column_mean**2
)

u_column_mean.attrs = {
    "long_name": "2001-2020 pressure-weighted column mean U wind, 1000-200 hPa",
    "units": "m s-1",
}

v_column_mean.attrs = {
    "long_name": "2001-2020 pressure-weighted column mean V wind, 1000-200 hPa",
    "units": "m s-1",
}

wind_speed_column_mean.attrs = {
    "long_name": "2001-2020 pressure-weighted column mean wind speed, 1000-200 hPa",
    "units": "m s-1",
}

#####################################################################
###################### CREAR DATASET DE SALIDA ######################
#####################################################################

u_clim.attrs = {
    "long_name": "2001-2020 climatological U wind by pressure level",
    "units": "m s-1",
}

v_clim.attrs = {
    "long_name": "2001-2020 climatological V wind by pressure level",
    "units": "m s-1",
}

output = xr.Dataset(
    {
        "u_clim": u_clim,
        "v_clim": v_clim,
        "wind_speed_clim": wind_speed_clim,
        "u_column_mean": u_column_mean,
        "v_column_mean": v_column_mean,
        "wind_speed_column_mean": wind_speed_column_mean,
    }
)

output.attrs = {
    "title": "ERA5 wind climatology over Patagonia and the South Atlantic",
    "period": "2001-2020",
    "temporal_average": "Monthly means weighted by number of days per month",
    "vertical_domain": "1000-200 hPa",
    "column_mean_method": (
        "Pressure-coordinate trapezoidal integration divided by 800 hPa"
    ),
    "spatial_domain": "80W-20W, 60S-35S",
    "source": "ERA5 monthly averaged pressure-level reanalysis",
}

#####################################################################
######################### GUARDAR RESULTADO #########################
#####################################################################

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "ERA5_wind_climatology_2001_2020.nc"

output.to_netcdf(output_file)

ds.close()

print()
print("Proceso terminado correctamente.")
print(f"Archivo generado: {output_file}")