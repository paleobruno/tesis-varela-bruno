import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
BASE = Path.home() / "MERRA2_dust" / "processed"
KRIG = BASE / "kriging_results"

reference_file = BASE / "MERRA2_reference_2001_2020_1deg.txt"

gk_file = (
    KRIG
    / "MERRA2_N100_seed42_log10_1deg_circular_BTF.txt"
)

# Carpeta para resultados de comparación
output_dir = BASE / "comparison_results"
output_dir.mkdir(parents=True, exist_ok=True)

difference_file = (
    output_dir
    / "MERRA2_N100_seed42_log10_1deg_circular_difference.txt"
)

metrics_file = (
    output_dir
    / "MERRA2_N100_seed42_log10_1deg_circular_metrics.txt"
)


# ----------------------------------------------------------------------
# Leer campos
# ----------------------------------------------------------------------
reference = np.loadtxt(reference_file, delimiter=",")
gk = np.loadtxt(gk_file, delimiter=",")


# ----------------------------------------------------------------------
# Comprobaciones
# ----------------------------------------------------------------------
if reference.shape != gk.shape:
    raise ValueError(
        f"Las dimensiones no coinciden: "
        f"referencia={reference.shape}, GK={gk.shape}"
    )

if np.isnan(reference).any():
    raise ValueError("El campo MERRA-2 contiene NaN.")

if np.isnan(gk).any():
    raise ValueError("El campo Global-Krigger contiene NaN.")


# ----------------------------------------------------------------------
# Diferencia
#
# Convención:
#
# difference = Global-Krigger - MERRA-2
#
# > 0  : GK sobreestima
# < 0  : GK subestima
# ----------------------------------------------------------------------
difference = gk - reference


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------
bias = np.mean(difference)

mae = np.mean(
    np.abs(difference)
)

rmse = np.sqrt(
    np.mean(difference**2)
)

correlation = np.corrcoef(
    reference.ravel(),
    gk.ravel()
)[0, 1]

mean_reference = np.mean(reference)
mean_gk = np.mean(gk)

mean_ratio = mean_gk / mean_reference


# ----------------------------------------------------------------------
# Guardar campo de diferencias
# ----------------------------------------------------------------------
np.savetxt(
    difference_file,
    difference,
    delimiter=",",
    fmt="%.15g"
)


# ----------------------------------------------------------------------
# Crear resumen
# ----------------------------------------------------------------------
summary = f"""
Comparación Global-Krigger vs MERRA-2
=====================================

Dataset GK:
N = 100
Seed = 42
Transformación = log10
Backtransform = lognormal corregido
Modelo de variograma = Circular
Grilla = 1 x 1 grado

Convención:
difference = GK - MERRA2

Dimensiones:
{reference.shape}

Media MERRA-2:
{mean_reference:.6f} g m^-2 yr^-1

Media Global-Krigger:
{mean_gk:.6f} g m^-2 yr^-1

Bias:
{bias:.6f} g m^-2 yr^-1

MAE:
{mae:.6f} g m^-2 yr^-1

RMSE:
{rmse:.6f} g m^-2 yr^-1

Correlación de Pearson:
{correlation:.6f}

Razón de medias GK / MERRA-2:
{mean_ratio:.6f}

Mínimo diferencia:
{difference.min():.6f} g m^-2 yr^-1

Máximo diferencia:
{difference.max():.6f} g m^-2 yr^-1
"""

metrics_file.write_text(summary)


# ----------------------------------------------------------------------
# Mostrar resultados
# ----------------------------------------------------------------------
print(summary)

print("Archivos guardados en:")
print(difference_file)
print(metrics_file)
