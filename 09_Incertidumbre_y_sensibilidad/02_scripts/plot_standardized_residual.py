import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
BASE = Path.home() / "MERRA2_dust" / "processed"
COMP = BASE / "comparison_results"

z_file = (
    COMP
    / "MERRA2_N100_seed42_log10_1deg_circular_standardized_residual.txt"
)

output_dir = COMP / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "GK_standardized_residual_z.png"


# ----------------------------------------------------------------------
# Leer campo z
# ----------------------------------------------------------------------
z = np.loadtxt(z_file, delimiter=",")


# ----------------------------------------------------------------------
# Coordenadas
# ----------------------------------------------------------------------
lon = np.arange(-80.0, -20.0 + 1.0, 1.0)
lat = np.arange(-60.0, -35.0 + 1.0, 1.0)

LON, LAT = np.meshgrid(lon, lat)


# ----------------------------------------------------------------------
# Escala centrada en cero
# ----------------------------------------------------------------------
max_abs_z = np.max(np.abs(z))

norm = TwoSlopeNorm(
    vmin=-max_abs_z,
    vcenter=0.0,
    vmax=max_abs_z
)


# ----------------------------------------------------------------------
# Gráfico
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

m = ax.pcolormesh(
    LON,
    LAT,
    z,
    shading="auto",
    cmap="RdBu_r",
    norm=norm
)

cbar = fig.colorbar(m, ax=ax)
cbar.set_label("Standardized residual z")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(
    "Standardized residual: "
    "[log10(MERRA-2) - Global-Krigger] / sigma"
)

ax.set_xlim(-80, -20)
ax.set_ylim(-60, -35)

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
print("Mapa de residuo estandarizado generado")
print("--------------------------------------")
print(f"z mínimo     : {z.min():.6f}")
print(f"z máximo     : {z.max():.6f}")
print(f"Máximo |z|  : {np.max(np.abs(z)):.6f}")
print()
print("Guardado en:")
print(output_file)
