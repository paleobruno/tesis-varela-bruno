import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
ERA5_FILE = (
    Path.home()
    / "ERA5_wind"
    / "processed"
    / "ERA5_wind_climatology_2001_2020.nc"
)

OUTPUT_DIR = (
    Path.home()
    / "ERA5_wind"
    / "processed"
    / "vertical_similarity"
)

FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Abrir ERA5
# ----------------------------------------------------------------------
ds = xr.open_dataset(ERA5_FILE)

levels = ds["pressure_level"].values.astype(float)

u = ds["u_clim"].values
v = ds["v_clim"].values

nlev = len(levels)

print()
print(f"Número de niveles: {nlev}")
print(f"Desde {levels.max():.0f} hasta {levels.min():.0f} hPa")


# ----------------------------------------------------------------------
# Matrices que calcularemos
# ----------------------------------------------------------------------
pattern_similarity = np.zeros((nlev, nlev))
direction_similarity = np.zeros((nlev, nlev))
vector_rmse = np.zeros((nlev, nlev))


# ----------------------------------------------------------------------
# Funciones
# ----------------------------------------------------------------------
def pearson(a, b):
    """
    Correlación espacial de Pearson entre dos campos.
    """

    a = a.ravel()
    b = b.ravel()

    mask = np.isfinite(a) & np.isfinite(b)

    return np.corrcoef(
        a[mask],
        b[mask]
    )[0, 1]


def mean_direction_cosine(u1, v1, u2, v2):
    """
    Coseno medio del ángulo entre los vectores de viento.

     1  -> misma dirección
     0  -> perpendiculares
    -1  -> direcciones opuestas
    """

    speed1 = np.sqrt(u1**2 + v1**2)
    speed2 = np.sqrt(u2**2 + v2**2)

    denominator = speed1 * speed2

    valid = (
        np.isfinite(denominator)
        & (denominator > 1e-6)
    )

    cosine = (
        u1[valid] * u2[valid]
        + v1[valid] * v2[valid]
    ) / denominator[valid]

    # Evitar errores numéricos ligeramente fuera de [-1, 1]
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.mean(cosine)


def calculate_vector_rmse(u1, v1, u2, v2):
    """
    RMSE de la diferencia vectorial entre ambos niveles.

    sqrt( mean[ (du)^2 + (dv)^2 ] )
    """

    squared_difference = (
        (u1 - u2)**2
        + (v1 - v2)**2
    )

    return np.sqrt(
        np.nanmean(squared_difference)
    )


# ----------------------------------------------------------------------
# Comparación de todos los pares de niveles
# ----------------------------------------------------------------------
for i in range(nlev):

    for j in range(nlev):

        # --------------------------------------------------------------
        # 1. Similitud del patrón espacial
        #
        # Calculamos r para u y v separadamente y luego promediamos.
        # De esta forma u no domina automáticamente solo por tener
        # valores de mayor magnitud que v.
        # --------------------------------------------------------------
        r_u = pearson(
            u[i],
            u[j]
        )

        r_v = pearson(
            v[i],
            v[j]
        )

        pattern_similarity[i, j] = (
            r_u + r_v
        ) / 2.0

        # --------------------------------------------------------------
        # 2. Similitud de dirección
        # --------------------------------------------------------------
        direction_similarity[i, j] = (
            mean_direction_cosine(
                u[i],
                v[i],
                u[j],
                v[j]
            )
        )

        # --------------------------------------------------------------
        # 3. Diferencia vectorial absoluta
        # --------------------------------------------------------------
        vector_rmse[i, j] = (
            calculate_vector_rmse(
                u[i],
                v[i],
                u[j],
                v[j]
            )
        )


# ----------------------------------------------------------------------
# Guardar matrices
# ----------------------------------------------------------------------
np.savetxt(
    OUTPUT_DIR / "pattern_similarity_matrix.csv",
    pattern_similarity,
    delimiter=",",
    fmt="%.6f"
)

np.savetxt(
    OUTPUT_DIR / "direction_similarity_matrix.csv",
    direction_similarity,
    delimiter=",",
    fmt="%.6f"
)

np.savetxt(
    OUTPUT_DIR / "vector_rmse_matrix.csv",
    vector_rmse,
    delimiter=",",
    fmt="%.6f"
)

np.savetxt(
    OUTPUT_DIR / "pressure_levels_hPa.txt",
    levels,
    fmt="%.1f"
)


# ----------------------------------------------------------------------
# Comparación entre niveles consecutivos
#
# Esto será especialmente útil para detectar transiciones verticales.
# ----------------------------------------------------------------------
adjacent_lines = []

header = (
    "level_1_hPa,level_2_hPa,"
    "pattern_similarity,"
    "direction_similarity,"
    "vector_RMSE_m_s,"
    "mean_speed_1_m_s,"
    "mean_speed_2_m_s\n"
)

adjacent_lines.append(header)

print()
print("Comparación entre niveles consecutivos")
print("======================================")
print()
print(
    "Nivel 1   Nivel 2   Pattern sim.   Direction sim.   "
    "Vector RMSE [m/s]"
)
print(
    "---------------------------------------------------------------"
)

for i in range(nlev - 1):

    j = i + 1

    speed_i = np.sqrt(
        u[i]**2 + v[i]**2
    )

    speed_j = np.sqrt(
        u[j]**2 + v[j]**2
    )

    mean_speed_i = np.nanmean(speed_i)
    mean_speed_j = np.nanmean(speed_j)

    adjacent_lines.append(
        f"{levels[i]:.0f},"
        f"{levels[j]:.0f},"
        f"{pattern_similarity[i,j]:.6f},"
        f"{direction_similarity[i,j]:.6f},"
        f"{vector_rmse[i,j]:.6f},"
        f"{mean_speed_i:.6f},"
        f"{mean_speed_j:.6f}\n"
    )

    print(
        f"{levels[i]:7.0f}   "
        f"{levels[j]:7.0f}      "
        f"{pattern_similarity[i,j]:8.4f}         "
        f"{direction_similarity[i,j]:8.4f}          "
        f"{vector_rmse[i,j]:8.3f}"
    )


with open(
    OUTPUT_DIR / "adjacent_level_similarity.csv",
    "w"
) as f:

    f.writelines(adjacent_lines)


# ----------------------------------------------------------------------
# Función para dibujar matrices
# ----------------------------------------------------------------------
def plot_matrix(
    matrix,
    title,
    colorbar_label,
    filename,
    vmin=None,
    vmax=None,
    cmap="viridis"
):

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    im = ax.imshow(
        matrix,
        origin="upper",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    tick_positions = np.arange(nlev)

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    ax.set_xticklabels(
        [f"{p:.0f}" for p in levels],
        rotation=90
    )

    ax.set_yticklabels(
        [f"{p:.0f}" for p in levels]
    )

    ax.set_xlabel("Pressure level [hPa]")
    ax.set_ylabel("Pressure level [hPa]")

    ax.set_title(title)

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(colorbar_label)

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ----------------------------------------------------------------------
# Dibujar las tres matrices
# ----------------------------------------------------------------------
plot_matrix(
    pattern_similarity,
    "ERA5 vertical similarity: spatial wind pattern",
    "Mean Pearson correlation of u and v",
    "ERA5_pattern_similarity_matrix.png",
    vmin=-1,
    vmax=1,
    cmap="RdBu_r"
)

plot_matrix(
    direction_similarity,
    "ERA5 vertical similarity: wind direction",
    "Mean vector cosine similarity",
    "ERA5_direction_similarity_matrix.png",
    vmin=-1,
    vmax=1,
    cmap="RdBu_r"
)

plot_matrix(
    vector_rmse,
    "ERA5 vertical difference: wind vectors",
    "Vector RMSE [m s$^{-1}$]",
    "ERA5_vector_RMSE_matrix.png",
    cmap="viridis"
)


# ----------------------------------------------------------------------
# Final
# ----------------------------------------------------------------------
print()
print("Análisis vertical terminado")
print("---------------------------")
print()
print("Resultados guardados en:")
print(OUTPUT_DIR)
print()
print("Figuras:")
print(FIG_DIR / "ERA5_pattern_similarity_matrix.png")
print(FIG_DIR / "ERA5_direction_similarity_matrix.png")
print(FIG_DIR / "ERA5_vector_RMSE_matrix.png")

ds.close()
