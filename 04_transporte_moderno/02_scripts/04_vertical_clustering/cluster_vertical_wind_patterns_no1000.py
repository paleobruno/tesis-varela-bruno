import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.cluster.hierarchy import dendrogram


# ======================================================================
# 1. RUTAS
# ======================================================================
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
    / "vertical_clustering_pattern_no1000"
)

FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 2. ABRIR ERA5
# ======================================================================
ds = xr.open_dataset(ERA5_FILE)

levels_all = ds["pressure_level"].values.astype(float)

# Excluir 1000 hPa
keep = levels_all != 1000.0

levels = levels_all[keep]

u = ds["u_clim"].values[keep, :, :]
v = ds["v_clim"].values[keep, :, :]

lat = ds["latitude"].values
lon = ds["longitude"].values

nlev = len(levels)
nlat = len(lat)
nlon = len(lon)

print()
print("Datos ERA5")
print("===========")
print(f"Niveles     : {nlev}")
print(f"Latitudes   : {nlat}")
print(f"Longitudes  : {nlon}")
print()


# ======================================================================
# 3. PESOS DE ÁREA
#
# En una grilla lat-lon:
#
#       dA ~ cos(phi) dphi dlambda
#
# Por eso una celda a 60°S representa menos superficie que una
# celda a 35°S.
# ======================================================================
lat_weights = np.cos(
    np.deg2rad(lat)
)

weight_grid = np.repeat(
    lat_weights[:, None],
    nlon,
    axis=1
)

# Normalizamos para que la suma de pesos sea 1
weight_grid = (
    weight_grid
    / np.sum(weight_grid)
)

w = weight_grid.ravel()


# ======================================================================
# 4. MEDIA Y DESVIACIÓN ESTÁNDAR ESPACIAL PONDERADAS
# ======================================================================
def weighted_mean(field, weights):

    field_flat = field.ravel()

    return np.sum(
        weights * field_flat
    )


def weighted_std(field, weights):

    field_flat = field.ravel()

    mean = np.sum(
        weights * field_flat
    )

    variance = np.sum(
        weights
        * (field_flat - mean)**2
    )

    return np.sqrt(variance)


# ======================================================================
# 5. CONSTRUIR VECTOR DE CARACTERÍSTICAS PARA CADA NIVEL
#
# Para cada presión p:
#
#   u' = (u - <u>) / sigma_u
#   v' = (v - <v>) / sigma_v
#
# Luego:
#
# X_p = [
#        sqrt(w_1) u'_1, ..., sqrt(w_N) u'_N,
#        sqrt(w_1) v'_1, ..., sqrt(w_N) v'_N
#       ] / sqrt(2)
#
# Así preservamos el patrón espacial completo de u y v,
# pero eliminamos media y escala.
# ======================================================================
features = []

diagnostic_lines = []

diagnostic_lines.append(
    "pressure_hPa,"
    "mean_u_m_s,std_u_m_s,"
    "mean_v_m_s,std_v_m_s\n"
)

sqrt_w = np.sqrt(w)


for k in range(nlev):

    u_field = u[k]
    v_field = v[k]

    mean_u = weighted_mean(
        u_field,
        w
    )

    std_u = weighted_std(
        u_field,
        w
    )

    mean_v = weighted_mean(
        v_field,
        w
    )

    std_v = weighted_std(
        v_field,
        w
    )

    if std_u == 0 or std_v == 0:
        raise ValueError(
            f"Desviación estándar cero en "
            f"{levels[k]:.0f} hPa"
        )

    u_norm = (
        u_field.ravel() - mean_u
    ) / std_u

    v_norm = (
        v_field.ravel() - mean_v
    ) / std_v

    feature_vector = np.concatenate(
        (
            sqrt_w * u_norm,
            sqrt_w * v_norm
        )
    ) / np.sqrt(2.0)

    features.append(
        feature_vector
    )

    diagnostic_lines.append(
        f"{levels[k]:.0f},"
        f"{mean_u:.8f},"
        f"{std_u:.8f},"
        f"{mean_v:.8f},"
        f"{std_v:.8f}\n"
    )


features = np.asarray(features)


with open(
    OUTPUT_DIR / "level_normalization_diagnostics.csv",
    "w"
) as f:

    f.writelines(
        diagnostic_lines
    )


# ======================================================================
# 6. DISTANCIA ENTRE DOS NIVELES
#
# Después de nuestra construcción:
#
# ||X_p - X_q||²
#
# equivale a una distancia cuadrática ponderada por área
# entre los patrones espaciales de u y v.
# ======================================================================
distance_matrix = np.zeros(
    (nlev, nlev)
)

for i in range(nlev):

    for j in range(nlev):

        difference = (
            features[i]
            - features[j]
        )

        distance_matrix[i, j] = np.sqrt(
            np.sum(
                difference**2
            )
        )


np.savetxt(
    OUTPUT_DIR / "pattern_distance_matrix.csv",
    distance_matrix,
    delimiter=",",
    fmt="%.8f"
)


# ======================================================================
# 7. CLUSTERING JERÁRQUICO DE WARD
#    CON RESTRICCIÓN DE CONTIGÜIDAD VERTICAL
#
# Cada cluster guarda:
#
#   id        -> identificador para el dendrograma
#   members   -> índices de los niveles incluidos
#   centroid  -> centroide en espacio de características
#   size      -> número de niveles
#
# Solo permitimos fusionar clusters vecinos.
# ======================================================================
clusters = []

for i in range(nlev):

    clusters.append(
        {
            "id": i,
            "members": [i],
            "centroid": features[i].copy(),
            "size": 1
        }
    )


next_cluster_id = nlev

linkage_rows = []

merge_history = []

# Guardaremos cómo quedan las capas al tener K grupos
snapshots = {}

requested_K = [
    2,
    3,
    4,
    5,
    6
]

# Estado inicial
if nlev in requested_K:
    snapshots[nlev] = [
        c["members"].copy()
        for c in clusters
    ]


# ======================================================================
# 8. FUNCIÓN DE COSTO DE WARD
#
# Delta(A,B) =
#
# n_A n_B
# ---------  ||mu_A - mu_B||²
# n_A+n_B
#
# Es el aumento de la suma de cuadrados intra-cluster
# causado por fusionar A y B.
# ======================================================================
def ward_delta(cluster_a, cluster_b):

    n_a = cluster_a["size"]
    n_b = cluster_b["size"]

    mu_a = cluster_a["centroid"]
    mu_b = cluster_b["centroid"]

    centroid_difference = (
        mu_a - mu_b
    )

    squared_distance = np.sum(
        centroid_difference**2
    )

    delta = (
        (n_a * n_b)
        / (n_a + n_b)
        * squared_distance
    )

    return delta


# ======================================================================
# 9. ALGORITMO AGLOMERATIVO
# ======================================================================
while len(clusters) > 1:

    best_position = None
    best_delta = np.inf

    # --------------------------------------------------------------
    # SOLO comparamos clusters verticalmente adyacentes
    # --------------------------------------------------------------
    for i in range(
        len(clusters) - 1
    ):

        A = clusters[i]
        B = clusters[i + 1]

        delta = ward_delta(
            A,
            B
        )

        if delta < best_delta:

            best_delta = delta
            best_position = i


    # --------------------------------------------------------------
    # Clusters que se fusionarán
    # --------------------------------------------------------------
    A = clusters[
        best_position
    ]

    B = clusters[
        best_position + 1
    ]


    # --------------------------------------------------------------
    # Nuevo centroide
    #
    # mu_AB =
    #
    # n_A mu_A + n_B mu_B
    # -------------------
    #       n_A+n_B
    # --------------------------------------------------------------
    new_size = (
        A["size"]
        + B["size"]
    )

    new_centroid = (
        A["size"] * A["centroid"]
        + B["size"] * B["centroid"]
    ) / new_size

    new_members = (
        A["members"]
        + B["members"]
    )


    # --------------------------------------------------------------
    # Altura equivalente Ward para dendrograma
    #
    # d_Ward = sqrt(2 Delta)
    # --------------------------------------------------------------
    ward_height = np.sqrt(
        2.0 * best_delta
    )


    linkage_rows.append(
        [
            A["id"],
            B["id"],
            ward_height,
            new_size
        ]
    )


    # --------------------------------------------------------------
    # Información física del bloque fusionado
    # --------------------------------------------------------------
    pressure_members = levels[
        new_members
    ]

    p_max = np.max(
        pressure_members
    )

    p_min = np.min(
        pressure_members
    )

    merge_history.append(
        (
            len(clusters),
            len(clusters) - 1,
            A["id"],
            B["id"],
            p_max,
            p_min,
            best_delta,
            ward_height
        )
    )


    # --------------------------------------------------------------
    # Crear nuevo cluster
    # --------------------------------------------------------------
    new_cluster = {
        "id": next_cluster_id,
        "members": new_members,
        "centroid": new_centroid,
        "size": new_size
    }

    next_cluster_id += 1


    # --------------------------------------------------------------
    # Sustituir A y B por AB, conservando orden vertical
    # --------------------------------------------------------------
    clusters = (
        clusters[:best_position]
        + [new_cluster]
        + clusters[
            best_position + 2:
        ]
    )


    # --------------------------------------------------------------
    # Guardar snapshot cuando quedan K capas
    # --------------------------------------------------------------
    K = len(clusters)

    if K in requested_K:

        snapshots[K] = [
            c["members"].copy()
            for c in clusters
        ]


# ======================================================================
# 10. MATRIZ DE LINKAGE
# ======================================================================
linkage_matrix = np.asarray(
    linkage_rows,
    dtype=float
)

np.savetxt(
    OUTPUT_DIR / "constrained_ward_linkage.csv",
    linkage_matrix,
    delimiter=",",
    header=(
        "cluster_A,"
        "cluster_B,"
        "ward_distance,"
        "new_cluster_size"
    ),
    comments="",
    fmt="%.8f"
)


# ======================================================================
# 11. HISTORIAL DE FUSIONES
# ======================================================================
history_lines = [
    "clusters_before,"
    "clusters_after,"
    "cluster_A,"
    "cluster_B,"
    "pressure_max_hPa,"
    "pressure_min_hPa,"
    "delta_SSE,"
    "ward_distance\n"
]

for row in merge_history:

    history_lines.append(
        f"{row[0]},"
        f"{row[1]},"
        f"{row[2]},"
        f"{row[3]},"
        f"{row[4]:.1f},"
        f"{row[5]:.1f},"
        f"{row[6]:.8f},"
        f"{row[7]:.8f}\n"
    )


with open(
    OUTPUT_DIR / "merge_history.csv",
    "w"
) as f:

    f.writelines(
        history_lines
    )


# ======================================================================
# 12. GUARDAR SOLUCIONES PARA K = 2 ... 6
# ======================================================================
solution_lines = []

solution_lines.append(
    "Clustering vertical restringido - patrones ERA5 (975-200 hPa)\n"
)

solution_lines.append(
    "===============================================\n\n"
)

solution_lines.append(
    "Cada capa contiene exclusivamente niveles "
    "contiguos en presión.\n\n"
)


for K in sorted(
    snapshots.keys()
):

    solution_lines.append(
        f"K = {K} capas\n"
    )

    solution_lines.append(
        "------------\n"
    )

    for number, members in enumerate(
        snapshots[K],
        start=1
    ):

        pressures = levels[
            members
        ]

        solution_lines.append(
            f"Capa {number}: "
            f"{np.max(pressures):.0f}"
            f" - "
            f"{np.min(pressures):.0f}"
            f" hPa"
            f"   ({len(members)} niveles)\n"
        )

    solution_lines.append(
        "\n"
    )


with open(
    OUTPUT_DIR / "candidate_vertical_layers.txt",
    "w"
) as f:

    f.writelines(
        solution_lines
    )


# ======================================================================
# 13. DENDROGRAMA
# ======================================================================
fig, ax = plt.subplots(
    figsize=(13, 6)
)

dendrogram(
    linkage_matrix,
    labels=[
        f"{p:.0f}"
        for p in levels
    ],
    leaf_rotation=90,
    leaf_font_size=9,
    ax=ax
)

ax.set_xlabel(
    "Pressure level [hPa]"
)

ax.set_ylabel(
    "Constrained Ward distance"
)

ax.set_title(
    "Hierarchical clustering of ERA5 wind patterns (975-200 hPa)\n"
    "Area-weighted, standardized and vertically contiguous"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIG_DIR
    / "ERA5_vertical_pattern_dendrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ======================================================================
# 14. COSTO DE FUSIÓN
#
# Permite detectar posibles "saltos" al intentar unir
# regímenes dinámicos diferentes.
# ======================================================================
clusters_after = np.array(
    [
        row[1]
        for row in merge_history
    ]
)

delta_sse = np.array(
    [
        row[6]
        for row in merge_history
    ]
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    clusters_after,
    delta_sse,
    marker="o"
)

ax.set_xlabel(
    "Number of clusters after merge"
)

ax.set_ylabel(
    "Increase in within-cluster SSE"
)

ax.set_title(
    "Ward merge cost"
)

# El clustering comienza con muchos grupos y termina con uno
ax.invert_xaxis()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIG_DIR
    / "ERA5_vertical_pattern_merge_cost.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ======================================================================
# 15. MOSTRAR RESULTADOS
# ======================================================================
print()
print("Clustering jerárquico terminado")
print("===============================")
print()

print(
    "".join(
        solution_lines
    )
)

print("Resultados guardados en:")
print(OUTPUT_DIR)

print()
print("Figuras:")
print(
    FIG_DIR
    / "ERA5_vertical_pattern_dendrogram.png"
)

print(
    FIG_DIR
    / "ERA5_vertical_pattern_merge_cost.png"
)


# ======================================================================
# 16. CERRAR DATASET
# ======================================================================
ds.close()
