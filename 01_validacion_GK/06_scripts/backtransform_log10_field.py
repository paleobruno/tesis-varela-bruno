import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Carpeta de trabajo
# ----------------------------------------------------------------------
BASE = Path.home() / "MERRA2_dust" / "processed" / "kriging_results"

# ----------------------------------------------------------------------
# Archivos de entrada
# ----------------------------------------------------------------------
interp_file = BASE / "MERRA2_N100_seed42_log10_1deg_circular_interpolated.txt"
unc_file = BASE / "MERRA2_N100_seed42_log10_1deg_circular_uncertainty.txt"

# Archivo de salida
output_file = BASE / "MERRA2_N100_seed42_log10_1deg_circular_BTF.txt"

# ----------------------------------------------------------------------
# Leer campos
# ----------------------------------------------------------------------
mu = np.loadtxt(interp_file, delimiter=",")
sigma = np.loadtxt(unc_file, delimiter=",")

# Comprobar que tengan las mismas dimensiones
if mu.shape != sigma.shape:
    raise ValueError(
        f"Las dimensiones no coinciden: "
        f"interpolado={mu.shape}, incertidumbre={sigma.shape}"
    )

# ----------------------------------------------------------------------
# Backtransform lognormal corregido
#
# Si:
#     Y = log10(X)
#
# y Y ~ N(mu, sigma^2), entonces:
#
# E[X] = exp(mu*ln(10) + 0.5*sigma^2*ln(10)^2)
# ----------------------------------------------------------------------
ln10 = np.log(10.0)

btf = np.exp(
    mu * ln10
    + 0.5 * sigma**2 * ln10**2
)

# ----------------------------------------------------------------------
# Guardar resultado
# ----------------------------------------------------------------------
np.savetxt(
    output_file,
    btf,
    delimiter=",",
    fmt="%.15g"
)

# ----------------------------------------------------------------------
# Mostrar diagnóstico
# ----------------------------------------------------------------------
print()
print("Backtransform terminado")
print("-----------------------")
print(f"Dimensiones : {btf.shape}")
print(f"Mínimo      : {btf.min():.6f} g m^-2 yr^-1")
print(f"Máximo      : {btf.max():.6f} g m^-2 yr^-1")
print(f"Media       : {btf.mean():.6f} g m^-2 yr^-1")
print()
print("Guardado en:")
print(output_file)
