import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
BASE = Path.home() / "MERRA2_dust" / "processed"
KRIG = BASE / "kriging_results"
COMP = BASE / "comparison_results"

reference_file = BASE / "MERRA2_reference_2001_2020_1deg.txt"
gk_file = KRIG / "MERRA2_N100_seed42_log10_1deg_circular_BTF.txt"

output_dir = COMP / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "scatter_GK_vs_MERRA2.png"


# ----------------------------------------------------------------------
# Leer campos
# ----------------------------------------------------------------------
reference = np.loadtxt(reference_file, delimiter=",")
gk = np.loadtxt(gk_file, delimiter=",")

if reference.shape != gk.shape:
    raise ValueError(
        f"Las dimensiones no coinciden: "
        f"referencia={reference.shape}, GK={gk.shape}"
    )


# ----------------------------------------------------------------------
# Pasar a vectores
# ----------------------------------------------------------------------
x = reference.ravel()
y = gk.ravel()


# ----------------------------------------------------------------------
# Correlación
# ----------------------------------------------------------------------
r = np.corrcoef(x, y)[0, 1]


# ----------------------------------------------------------------------
# Recta de ajuste lineal
# ----------------------------------------------------------------------
slope, intercept = np.polyfit(x, y, 1)

x_fit = np.linspace(
    min(x.min(), y.min()),
    max(x.max(), y.max()),
    200
)

y_fit = slope * x_fit + intercept


# ----------------------------------------------------------------------
# Gráfico
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    x,
    y,
    s=18,
    alpha=0.55
)

# Línea ideal 1:1
ax.plot(
    x_fit,
    x_fit,
    linestyle="--",
    linewidth=1.5,
    label="1:1"
)

# Ajuste lineal
ax.plot(
    x_fit,
    y_fit,
    linewidth=1.5,
    label="Linear fit"
)

ax.set_xlabel(
    "MERRA-2 reference [g m$^{-2}$ yr$^{-1}$]"
)

ax.set_ylabel(
    "Global-Krigger [g m$^{-2}$ yr$^{-1}$]"
)

ax.set_title(
    "Global-Krigger vs MERRA-2"
)

ax.legend()

ax.text(
    0.05,
    0.95,
    f"r = {r:.3f}\n"
    f"y = {slope:.3f}x + {intercept:.3f}",
    transform=ax.transAxes,
    verticalalignment="top"
)

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ----------------------------------------------------------------------
# Confirmación
# ----------------------------------------------------------------------
print()
print("Scatter generado correctamente")
print("--------------------------------")
print(f"Correlación r       : {r:.6f}")
print(f"Pendiente           : {slope:.6f}")
print(f"Intercepto          : {intercept:.6f}")
print()
print("Guardado en:")
print(output_file)
