from pathlib import Path

import numpy as np
import xarray as xr

BASE_DIR = Path.home() / "MERRA2_dust"

MAIN_DIR = BASE_DIR / "monthly"
DUSV_DIR = BASE_DIR / "monthly_DUSV"
OUTPUT_DIR = BASE_DIR / "processed"

main_files = sorted(MAIN_DIR.glob("*.nc"))
dusv_files = sorted(DUSV_DIR.glob("*.nc"))


#####################################################################
################## IDENTIFICACION Y EMPAREJAMIENTO ##################
#####################################################################

if len(main_files) != 240:
    raise RuntimeError(f"Se esperaban 240 archivos principales, pero se encontraron {len(main_files)}.")

if len(dusv_files) != 240:
    raise RuntimeError(f"Se esperaban 240 archivos DUSV, pero se encontraron {len(dusv_files)}.")

for main_file, dusv_file in zip(main_files, dusv_files):
    if main_file.name != dusv_file.name:
        raise RuntimeError(
            f"Los archivos no coinciden:\n"
            f"  principal: {main_file.name}\n"
            f"  DUSV:      {dusv_file.name}"
        )

print("Archivos encontrados y emparejados correctamente: 240 meses.")

#####################################################################
####################### VARIABLES MATEMATICAS #######################
#####################################################################

# Variables de deposición de polvo

BINS = range(1, 6)

DUDP_VARS = [f"DUDP{i:03d}" for i in BINS]  # Dry deposition
DUSD_VARS = [f"DUSD{i:03d}" for i in BINS]  # Sedimentation
DUWT_VARS = [f"DUWT{i:03d}" for i in BINS]  # Wet deposition
DUSV_VARS = [f"DUSV{i:03d}" for i in BINS]  # Convective scavenging

MAIN_VARS = DUDP_VARS + DUSD_VARS + DUWT_VARS
ALL_VARS = MAIN_VARS + DUSV_VARS

#####################################################################
## CLIMATOLOGÍA 2001-2020 PONDERADA POR NÚMERO DE DÍAS DE CADA MES ##
#####################################################################

weighted_dudp = None
weighted_dusd = None
weighted_duwt = None
weighted_dusv = None
total_days = 0

for main_file, dusv_file in zip(main_files, dusv_files):

    with xr.open_dataset(main_file) as ds_main, xr.open_dataset(dusv_file) as ds_dusv:

        # Verificar que ambos archivos tengan exactamente la misma grilla
        if not np.array_equal(ds_main["lat"].values, ds_dusv["lat"].values):
            raise RuntimeError(f"Las latitudes no coinciden en {main_file.name}")

        if not np.array_equal(ds_main["lon"].values, ds_dusv["lon"].values):
            raise RuntimeError(f"Las longitudes no coinciden en {main_file.name}")

        # Número de días del mes (incluye correctamente años bisiestos)
        days = int(ds_main["time"].dt.days_in_month.values[0])

        # Sumar los 5 bins de cada mecanismo y eliminamos coordenada temporal
        dudp = sum(ds_main[var] for var in DUDP_VARS).isel(time=0, drop=True).load()
        dusd = sum(ds_main[var] for var in DUSD_VARS).isel(time=0, drop=True).load()
        duwt = sum(ds_main[var] for var in DUWT_VARS).isel(time=0, drop=True).load()
        dusv = sum(ds_dusv[var] for var in DUSV_VARS).isel(time=0, drop=True).load()

        # Acumular ponderando por la duración real del mes
        if weighted_dudp is None:
            weighted_dudp = dudp * days
            weighted_dusd = dusd * days
            weighted_duwt = duwt * days
            weighted_dusv = dusv * days
        else:
            weighted_dudp += dudp * days
            weighted_dusd += dusd * days
            weighted_duwt += duwt * days
            weighted_dusv += dusv * days

        total_days += days

#####################################################################
### CÁLCULO CLIMATOLOGÍAS || CONSTRUCCIÓN DEL CAMPO DE REFERENCIA ###
#####################################################################

# 2001-2020 contiene 7305 días
if total_days != 7305:
    raise RuntimeError(
        f"Se esperaban 7305 días entre 2001 y 2020, pero se obtuvieron {total_days}."
    )

dudp_clim = weighted_dudp / total_days
dusd_clim = weighted_dusd / total_days
duwt_clim = weighted_duwt / total_days
dusv_clim = weighted_dusv / total_days

# Campo climatológico total de deposición/remoción de polvo
f_ref = dudp_clim + dusd_clim + duwt_clim + dusv_clim

# Conversión de unidades: kg m-2 s-1 -> g m-2 yr-1
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
KG_TO_G = 1000.0

conversion_factor = KG_TO_G * SECONDS_PER_YEAR

dudp_clim_yr = dudp_clim * conversion_factor
dusd_clim_yr = dusd_clim * conversion_factor
duwt_clim_yr = duwt_clim * conversion_factor
dusv_clim_yr = dusv_clim * conversion_factor
f_ref_yr = f_ref * conversion_factor

#####################################################################
################### CREACIÓN DE DATASET DE SALIDA ###################
#####################################################################

output = xr.Dataset(
    {
        # Unidades originales de MERRA-2
        "DUDP_bulk_clim": dudp_clim,
        "DUSD_bulk_clim": dusd_clim,
        "DUWT_bulk_clim": duwt_clim,
        "DUSV_bulk_clim": dusv_clim,
        "F_ref": f_ref,

        # Unidades convertidas
        "DUDP_bulk_clim_g_m2_yr": dudp_clim_yr,
        "DUSD_bulk_clim_g_m2_yr": dusd_clim_yr,
        "DUWT_bulk_clim_g_m2_yr": duwt_clim_yr,
        "DUSV_bulk_clim_g_m2_yr": dusv_clim_yr,
        "F_ref_g_m2_yr": f_ref_yr,
    }
)

output["DUDP_bulk_clim"].attrs = {
    "long_name": "2001-2020 climatological dust dry deposition",
    "units": "kg m-2 s-1",
}

output["DUSD_bulk_clim"].attrs = {
    "long_name": "2001-2020 climatological dust sedimentation",
    "units": "kg m-2 s-1",
}

output["DUWT_bulk_clim"].attrs = {
    "long_name": "2001-2020 climatological dust wet deposition",
    "units": "kg m-2 s-1",
}

output["DUSV_bulk_clim"].attrs = {
    "long_name": "2001-2020 climatological dust convective scavenging",
    "units": "kg m-2 s-1",
}

output["F_ref"].attrs = {
    "long_name": "2001-2020 climatological total dust deposition/removal reference field",
    "units": "kg m-2 s-1",
}

output.attrs = {
    "title": "MERRA-2 dust deposition reference field",
    "period": "2001-2020",
    "temporal_average": "Monthly means weighted by number of days per month",
    "spatial_domain": "80W-20W, 60S-35S",
    "source": "MERRA-2 M2TMNXADG v5.12.4",
}

output["DUDP_bulk_clim_g_m2_yr"].attrs = {
    "long_name": "2001-2020 climatological dust dry deposition",
    "units": "g m-2 yr-1",
}

output["DUSD_bulk_clim_g_m2_yr"].attrs = {
    "long_name": "2001-2020 climatological dust sedimentation",
    "units": "g m-2 yr-1",
}

output["DUWT_bulk_clim_g_m2_yr"].attrs = {
    "long_name": "2001-2020 climatological dust wet deposition",
    "units": "g m-2 yr-1",
}

output["DUSV_bulk_clim_g_m2_yr"].attrs = {
    "long_name": "2001-2020 climatological dust convective scavenging",
    "units": "g m-2 yr-1",
}

output["F_ref_g_m2_yr"].attrs = {
    "long_name": "2001-2020 climatological total dust deposition/removal reference field",
    "units": "g m-2 yr-1",
}

######################################################################
############################## GUARDADO ##############################
######################################################################

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "MERRA2_dust_reference_2001_2020.nc"
output.to_netcdf(output_file)

print()
print("Proceso terminado correctamente.")
print(f"Días utilizados: {total_days}")
print(f"Archivo generado: {output_file}")