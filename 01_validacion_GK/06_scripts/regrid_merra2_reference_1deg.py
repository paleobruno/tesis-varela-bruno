import numpy as np
import xarray as xr
from pathlib import Path

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
BASE = Path.home() / "MERRA2_dust" / "processed"

input_file = BASE / "MERRA2_dust_reference_2001_2020.nc"

output_txt = BASE / "MERRA2_reference_2001_2020_1deg.txt"
output_nc = BASE / "MERRA2_reference_2001_2020_1deg.nc"


# ----------------------------------------------------------------------
# Abrir campo de referencia
# ----------------------------------------------------------------------
ds = xr.open_dataset(input_file)

variable = "F_ref_g_m2_yr"

if variable not in ds:
    raise KeyError(
        f"No se encontró la variable '{variable}'. "
        f"Variables disponibles: {list(ds.data_vars)}"
    )

field = ds[variable]


# ----------------------------------------------------------------------
# Ordenar coordenadas
# ----------------------------------------------------------------------
# Esto asegura que latitud y longitud estén en orden creciente.
field = field.sortby("lat")
field = field.sortby("lon")


# ----------------------------------------------------------------------
# Definir grilla objetivo de Global-Krigger
#
# Longitude: -80 a -20 grados
# Latitude:  -60 a -35 grados
# Resolución: 1 grado
# ----------------------------------------------------------------------
lon_1deg = np.arange(-80.0, -20.0 + 1.0, 1.0)
lat_1deg = np.arange(-60.0, -35.0 + 1.0, 1.0)


# ----------------------------------------------------------------------
# Interpolar MERRA-2 a 1° x 1°
# ----------------------------------------------------------------------
field_1deg = field.interp(
    lon=lon_1deg,
    lat=lat_1deg,
    method="linear"
)


# ----------------------------------------------------------------------
# Comprobar resultado
# ----------------------------------------------------------------------
values = field_1deg.values

if np.isnan(values).any():
    raise ValueError("La grilla interpolada contiene valores NaN.")


# ----------------------------------------------------------------------
# Guardar como TXT
#
# Filas    = latitudes
# Columnas = longitudes
# ----------------------------------------------------------------------
np.savetxt(
    output_txt,
    values,
    delimiter=",",
    fmt="%.15g"
)


# ----------------------------------------------------------------------
# Guardar también como NetCDF, conservando coordenadas
# ----------------------------------------------------------------------
field_1deg.name = "F_ref_g_m2_yr"
field_1deg.attrs["units"] = "g m-2 yr-1"
field_1deg.attrs["description"] = (
    "MERRA-2 dust deposition reference field 2001-2020 "
    "interpolated to a 1 degree x 1 degree grid"
)

field_1deg.to_netcdf(output_nc)


# ----------------------------------------------------------------------
# Diagnóstico
# ----------------------------------------------------------------------
print()
print("Regridding MERRA-2 terminado")
print("---------------------------")
print(f"Dimensiones : {values.shape}")
print(f"Latitudes   : {lat_1deg[0]} a {lat_1deg[-1]} grados")
print(f"Longitudes  : {lon_1deg[0]} a {lon_1deg[-1]} grados")
print(f"Mínimo      : {values.min():.6f} g m^-2 yr^-1")
print(f"Máximo      : {values.max():.6f} g m^-2 yr^-1")
print(f"Media       : {values.mean():.6f} g m^-2 yr^-1")
print()
print("Archivos guardados en:")
print(output_txt)
print(output_nc)

ds.close()
