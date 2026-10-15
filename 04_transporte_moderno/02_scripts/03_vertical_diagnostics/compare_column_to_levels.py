from pathlib import Path

import numpy as np
import xarray as xr

BASE_DIR = Path.home() / "ERA5_wind"

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ERA5_wind_climatology_2001_2020.nc"
)

LEVELS_TO_COMPARE = [850, 500, 300, 200]

# -----------------------------------------------------------------------------
# Cargar campos
# -----------------------------------------------------------------------------

ds = xr.open_dataset(INPUT_FILE)

u_column = ds["u_column_mean"]
v_column = ds["v_column_mean"]

# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------

def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def spatial_correlation(a, b):
    a_flat = a.values.ravel()
    b_flat = b.values.ravel()

    return float(np.corrcoef(a_flat, b_flat)[0, 1])

# -----------------------------------------------------------------------------
# Comparar niveles con el promedio de columna
# -----------------------------------------------------------------------------

print()
print(
    "Nivel [hPa]   RMSE_u [m/s]   RMSE_v [m/s]   Corr_u   Corr_v"
)
print("-------------------------------------------------------------")

for level in LEVELS_TO_COMPARE:

    u_level = ds["u_clim"].sel(pressure_level=level)
    v_level = ds["v_clim"].sel(pressure_level=level)

    rmse_u = rmse(u_level, u_column)
    rmse_v = rmse(v_level, v_column)

    corr_u = spatial_correlation(u_level, u_column)
    corr_v = spatial_correlation(v_level, v_column)

    print(
        f"{level:10.0f}   "
        f"{rmse_u:12.3f}   "
        f"{rmse_v:12.3f}   "
        f"{corr_u:6.3f}   "
        f"{corr_v:6.3f}"
    )

ds.close()