import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
BASE = Path.home() / "MERRA2_dust" / "processed"
KRIG = BASE / "kriging_results"
COMP = BASE / "comparison_results"

reference_file = BASE / "MERRA2_reference_2001_2020_1deg.txt"

interp_file = (
    KRIG
    / "MERRA2_N100_seed42_log10_1deg_circular_interpolated.txt"
)

uncertainty_file = (
    KRIG
    / "MERRA2_N100_seed42_log10_1deg_circular_uncertainty.txt"
)

z_file = (
    COMP
    / "MERRA2_N100_seed42_log10_1deg_circular_standardized_residual.txt"
)

metrics_file = (
    COMP
    / "MERRA2_N100_seed42_log10_1deg_circular_uncertainty_metrics.txt"
)


# ----------------------------------------------------------------------
# Leer campos
# ----------------------------------------------------------------------
reference = np.loadtxt(reference_file, delimiter=",")
mu = np.loadtxt(interp_file, delimiter=",")
sigma = np.loadtxt(uncertainty_file, delimiter=",")


# ----------------------------------------------------------------------
# Comprobaciones
# ----------------------------------------------------------------------
if reference.shape != mu.shape or reference.shape != sigma.shape:
    raise ValueError(
        f"Dimensiones incompatibles:\n"
        f"MERRA2 = {reference.shape}\n"
        f"GK = {mu.shape}\n"
        f"sigma = {sigma.shape}"
    )

if np.any(reference <= 0):
    raise ValueError(
        "El campo de referencia contiene valores <= 0; "
        "no se puede aplicar log10."
    )

if np.any(sigma <= 0):
    raise ValueError(
        "El campo de incertidumbre contiene valores <= 0."
    )


# ----------------------------------------------------------------------
# Llevar referencia a espacio log10
# ----------------------------------------------------------------------
reference_log = np.log10(reference)


# ----------------------------------------------------------------------
# Residuo estandarizado
#
# z = (MERRA2_log10 - GK_log10) / sigma_GK
#
# Si la incertidumbre está bien calibrada:
#
# aproximadamente 68.3 % debería caer dentro de +/- 1 sigma
# aproximadamente 95.4 % dentro de +/- 2 sigma
# aproximadamente 99.7 % dentro de +/- 3 sigma
# ----------------------------------------------------------------------
z = (reference_log - mu) / sigma


# ----------------------------------------------------------------------
# Coberturas
# ----------------------------------------------------------------------
coverage_1sigma = np.mean(np.abs(z) <= 1.0) * 100.0
coverage_2sigma = np.mean(np.abs(z) <= 2.0) * 100.0
coverage_3sigma = np.mean(np.abs(z) <= 3.0) * 100.0

mean_z = np.mean(z)
std_z = np.std(z)

median_abs_z = np.median(np.abs(z))
max_abs_z = np.max(np.abs(z))


# ----------------------------------------------------------------------
# Guardar campo z
# ----------------------------------------------------------------------
np.savetxt(
    z_file,
    z,
    delimiter=",",
    fmt="%.15g"
)


# ----------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------
summary = f"""
Evaluación de incertidumbre Global-Krigger
==========================================

Dataset:
N = 100
Seed = 42
Transformación = log10
Modelo de variograma = Circular
Grilla = 1 x 1 grado

Definición:
z = [log10(MERRA2) - mu_GK] / sigma_GK

Cobertura dentro de +/- 1 sigma:
{coverage_1sigma:.2f} %

Cobertura esperada gaussiana:
68.27 %

Cobertura dentro de +/- 2 sigma:
{coverage_2sigma:.2f} %

Cobertura esperada gaussiana:
95.45 %

Cobertura dentro de +/- 3 sigma:
{coverage_3sigma:.2f} %

Cobertura esperada gaussiana:
99.73 %

Media de z:
{mean_z:.6f}

Desviación estándar de z:
{std_z:.6f}

Mediana de |z|:
{median_abs_z:.6f}

Máximo de |z|:
{max_abs_z:.6f}
"""

metrics_file.write_text(summary)


# ----------------------------------------------------------------------
# Mostrar resultados
# ----------------------------------------------------------------------
print(summary)

print("Archivos guardados en:")
print(z_file)
print(metrics_file)
