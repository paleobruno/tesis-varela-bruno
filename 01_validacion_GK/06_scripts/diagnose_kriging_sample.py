from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path.home() / "MERRA2_dust"

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "kriging_samples"
    / "MERRA2_dust_N100_seed42.txt"
)

FIGURE_DIR = BASE_DIR / "processed" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Leer muestra
# -----------------------------------------------------------------------------

data = np.loadtxt(INPUT_FILE)

lon = data[:, 0]
lat = data[:, 1]
values = data[:, 2]

print(f"Número de puntos: {len(values)}")

# -----------------------------------------------------------------------------
# Estadística descriptiva de la muestra
# -----------------------------------------------------------------------------

n = len(values)

mean_value = np.mean(values)
median_value = np.median(values)

# Desviación estándar muestral
std_value = np.std(values, ddof=1)

min_value = np.min(values)
max_value = np.max(values)
data_range = max_value - min_value

# Coeficiente de variación
cv = 100.0 * std_value / mean_value

# Asimetría muestral ajustada de Fisher-Pearson
skewness = (
    n
    / ((n - 1) * (n - 2))
    * np.sum(((values - mean_value) / std_value) ** 3)
)

q25, q75 = np.percentile(values, [25, 75])

print()
print("Estadística descriptiva")
print("-----------------------")
print(f"Mínimo:                {min_value:.4f}")
print(f"Máximo:                {max_value:.4f}")
print(f"Rango:                 {data_range:.4f}")
print(f"Media:                 {mean_value:.4f}")
print(f"Mediana:               {median_value:.4f}")
print(f"Desv. estándar:        {std_value:.4f}")
print(f"Coef. variación:       {cv:.2f} %")
print(f"Q1:                    {q25:.4f}")
print(f"Q3:                    {q75:.4f}")
print(f"Asimetría:             {skewness:.4f}")

# -----------------------------------------------------------------------------
# Comparación visual: datos originales vs log10
# -----------------------------------------------------------------------------

log_values = np.log10(values)

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    values,
    bins=15,
    edgecolor="black"
)

ax.set_xlabel(r"Dust deposition [g m$^{-2}$ yr$^{-1}$]")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of sampled MERRA-2 dust deposition")

plt.tight_layout()

output_figure = (
    FIGURE_DIR
    / f"MERRA2_sample_histogram_N{n}.png"
)

plt.savefig(
    output_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Histograma guardado en: {output_figure}")

# -----------------------------------------------------------------------------
# Diagnóstico de la transformación log10
# -----------------------------------------------------------------------------

log_values = np.log10(values)

log_mean = np.mean(log_values)
log_std = np.std(log_values, ddof=1)

log_skewness = (
    n
    / ((n - 1) * (n - 2))
    * np.sum(((log_values - log_mean) / log_std) ** 3)
)

print()
print("Transformación log10")
print("---------------------")
print(f"Media log10:          {log_mean:.4f}")
print(f"Desv. estándar:       {log_std:.4f}")
print(f"Asimetría log10:      {log_skewness:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    log_values,
    bins=15,
    edgecolor="black"
)

ax.set_xlabel(r"log$_{10}$ dust deposition")
ax.set_ylabel("Frequency")
ax.set_title("Distribution after log10 transformation")

plt.tight_layout()

log_figure = (
    FIGURE_DIR
    / f"MERRA2_sample_histogram_log10_N{n}.png"
)

plt.savefig(
    log_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Histograma log10 guardado en: {log_figure}")