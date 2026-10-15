import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from pathlib import Path

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
BASE = Path.home() / "MERRA2_dust" / "processed"
KRIG = BASE / "kriging_results"
COMP = BASE / "comparison_results"

reference_file = BASE / "MERRA2_reference_2001_2020_1deg.txt"
gk_file = KRIG / "MERRA2_N100_seed42_log10_1deg_circular_BTF.txt"
difference_file = COMP / "MERRA2_N100_seed42_log10_1deg_circular_difference.txt"

output_dir = COMP / "figures"
output_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Leer campos
# ----------------------------------------------------------------------
reference = np.loadtxt(reference_file, delimiter=",")
gk = np.loadtxt(gk_file, delimiter=",")
difference = np.loadtxt(difference_file, delimiter=",")


# ----------------------------------------------------------------------
# Coordenadas
# ----------------------------------------------------------------------
lon = np.arange(-80.0, -20.0 + 1.0, 1.0)
lat = np.arange(-60.0, -35.0 + 1.0, 1.0)

LON, LAT = np.meshgrid(lon, lat)


# ----------------------------------------------------------------------
# Escala común para MERRA-2 y Global-Krigger
# ----------------------------------------------------------------------
vmin = min(reference.min(), gk.min())
vmax = max(reference.max(), gk.max())

norm_flux = LogNorm(
    vmin=vmin,
    vmax=vmax
)


# ----------------------------------------------------------------------
# 1. MERRA-2
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

m = ax.pcolormesh(
    LON,
    LAT,
    reference,
    shading="auto",
    cmap="viridis",
    norm=norm_flux
)

cbar = fig.colorbar(m, ax=ax)
cbar.set_label("Dust deposition [g m$^{-2}$ yr$^{-1}$]")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("MERRA-2 reference field (2001–2020)")

ax.set_xlim(-80, -20)
ax.set_ylim(-60, -35)

fig.tight_layout()

fig.savefig(
    output_dir / "MERRA2_reference_1deg.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ----------------------------------------------------------------------
# 2. Global-Krigger
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

m = ax.pcolormesh(
    LON,
    LAT,
    gk,
    shading="auto",
    cmap="viridis",
    norm=norm_flux
)

cbar = fig.colorbar(m, ax=ax)
cbar.set_label("Dust deposition [g m$^{-2}$ yr$^{-1}$]")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Global-Krigger reconstruction")

ax.set_xlim(-80, -20)
ax.set_ylim(-60, -35)

fig.tight_layout()

fig.savefig(
    output_dir / "GlobalKrigger_BTF_1deg.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ----------------------------------------------------------------------
# 3. Campo de diferencias
# ----------------------------------------------------------------------
max_abs_diff = np.max(np.abs(difference))

norm_diff = TwoSlopeNorm(
    vmin=-max_abs_diff,
    vcenter=0.0,
    vmax=max_abs_diff
)

fig, ax = plt.subplots(figsize=(10, 6))

m = ax.pcolormesh(
    LON,
    LAT,
    difference,
    shading="auto",
    cmap="RdBu_r",
    norm=norm_diff
)

cbar = fig.colorbar(m, ax=ax)
cbar.set_label(
    "GK - MERRA-2 [g m$^{-2}$ yr$^{-1}$]"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Difference: Global-Krigger - MERRA-2")

ax.set_xlim(-80, -20)
ax.set_ylim(-60, -35)

fig.tight_layout()

fig.savefig(
    output_dir / "GK_minus_MERRA2_1deg.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ----------------------------------------------------------------------
# Confirmación
# ----------------------------------------------------------------------
print()
print("Mapas generados correctamente")
print("-----------------------------")
print(output_dir / "MERRA2_reference_1deg.png")
print(output_dir / "GlobalKrigger_BTF_1deg.png")
print(output_dir / "GK_minus_MERRA2_1deg.png")
