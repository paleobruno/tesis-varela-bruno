import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
BASE = Path.home() / "MERRA2_dust" / "processed"
PHYS = BASE / "physical_analysis"

INPUT_FILE = PHYS / "MERRA2_zonal_dust_profile.txt"

OUTPUT_DIR = PHYS / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FIG = OUTPUT_DIR / "MERRA2_downwind_exponential_fit.png"
OUTPUT_TXT = PHYS / "MERRA2_downwind_exponential_fit.txt"


# ----------------------------------------------------------------------
# Leer perfil zonal
# ----------------------------------------------------------------------
data = np.genfromtxt(
    INPUT_FILE,
    delimiter=",",
    names=True
)

lon = data["longitude_deg"]
dust = data["mean_dust_g_m2_yr"]


# ----------------------------------------------------------------------
# Encontrar automáticamente el máximo
# ----------------------------------------------------------------------
imax = np.argmax(dust)

peak_lon = lon[imax]
peak_dust = dust[imax]


# ----------------------------------------------------------------------
# Seleccionar solamente el sector downwind
#
# Como el transporte dominante es hacia el este:
# downwind = longitudes >= longitud del máximo
# ----------------------------------------------------------------------
mask = lon >= peak_lon

lon_downwind = lon[mask]
dust_downwind = dust[mask]


# ----------------------------------------------------------------------
# Convertir longitud a distancia zonal aproximada
#
# Banda latitudinal usada anteriormente:
# -51° a -35°
#
# Usamos su latitud media como referencia.
# ----------------------------------------------------------------------
lat_reference = (-51.0 + -35.0) / 2.0

earth_radius_km = 6371.0

delta_lon_rad = np.deg2rad(
    lon_downwind - peak_lon
)

x_km = (
    earth_radius_km
    * np.cos(np.deg2rad(lat_reference))
    * delta_lon_rad
)


# ----------------------------------------------------------------------
# Modelo exponencial
#
# F(x) = F0 * exp(-x / L)
# ----------------------------------------------------------------------
def exponential_model(x, F0, L):
    return F0 * np.exp(-x / L)


# ----------------------------------------------------------------------
# Ajuste
# ----------------------------------------------------------------------
initial_guess = [
    peak_dust,
    2000.0
]

parameters, covariance = curve_fit(
    exponential_model,
    x_km,
    dust_downwind,
    p0=initial_guess,
    bounds=(
        [0.0, 1.0],
        [np.inf, np.inf]
    )
)

F0_fit, L_fit = parameters

parameter_errors = np.sqrt(
    np.diag(covariance)
)

F0_error, L_error = parameter_errors


# ----------------------------------------------------------------------
# Predicción del modelo
# ----------------------------------------------------------------------
dust_fit = exponential_model(
    x_km,
    F0_fit,
    L_fit
)


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------
residuals = dust_downwind - dust_fit

rmse = np.sqrt(
    np.mean(residuals**2)
)

mae = np.mean(
    np.abs(residuals)
)

ss_res = np.sum(
    residuals**2
)

ss_tot = np.sum(
    (dust_downwind - np.mean(dust_downwind))**2
)

r2 = 1.0 - ss_res / ss_tot


# ----------------------------------------------------------------------
# Guardar resultados
# ----------------------------------------------------------------------
output_data = np.column_stack(
    (
        lon_downwind,
        x_km,
        dust_downwind,
        dust_fit,
        residuals
    )
)

header = (
    "longitude_deg,"
    "distance_downwind_km,"
    "MERRA2_dust_g_m2_yr,"
    "exponential_fit_g_m2_yr,"
    "residual_g_m2_yr"
)

np.savetxt(
    OUTPUT_TXT,
    output_data,
    delimiter=",",
    header=header,
    comments="",
    fmt="%.8f"
)


# ----------------------------------------------------------------------
# Gráfico
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.scatter(
    x_km,
    dust_downwind,
    s=35,
    label="MERRA-2 zonal mean",
    zorder=3
)

x_smooth = np.linspace(
    0.0,
    x_km.max(),
    500
)

ax.plot(
    x_smooth,
    exponential_model(
        x_smooth,
        F0_fit,
        L_fit
    ),
    linewidth=2,
    label="Exponential fit"
)

ax.set_xlabel(
    "Downwind distance from maximum [km]"
)

ax.set_ylabel(
    "Dust deposition [g m$^{-2}$ yr$^{-1}$]"
)

ax.set_title(
    "Downwind exponential decay of MERRA-2 dust deposition"
)

ax.text(
    0.97,
    0.95,
    f"$F(x)=F_0e^{{-x/L}}$\n"
    f"$F_0$ = {F0_fit:.3f} g m$^{{-2}}$ yr$^{{-1}}$\n"
    f"$L$ = {L_fit:.0f} ± {L_error:.0f} km\n"
    f"$R^2$ = {r2:.3f}\n"
    f"RMSE = {rmse:.3f}",
    transform=ax.transAxes,
    horizontalalignment="right",
    verticalalignment="top"
)

ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_FIG,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ----------------------------------------------------------------------
# Mostrar resultados
# ----------------------------------------------------------------------
print()
print("Ajuste exponencial downwind terminado")
print("------------------------------------")
print(f"Longitud del máximo : {peak_lon:.1f}°")
print(f"Latitud de referencia : {lat_reference:.1f}°")
print()
print(f"F0 : {F0_fit:.6f} ± {F0_error:.6f} g m^-2 yr^-1")
print(f"L  : {L_fit:.2f} ± {L_error:.2f} km")
print()
print(f"R²   : {r2:.6f}")
print(f"RMSE : {rmse:.6f} g m^-2 yr^-1")
print(f"MAE  : {mae:.6f} g m^-2 yr^-1")
print()
print("Archivos guardados en:")
print(OUTPUT_TXT)
print(OUTPUT_FIG)
