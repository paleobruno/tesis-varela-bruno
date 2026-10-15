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
    / "vertical_clustering_dynamic"
)

FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 2. ABRIR ERA5
# ======================================================================
ds = xr.open_dataset(ERA5_FILE)

levels = ds["pressure_level"].values.astype(float)

u = ds["u_clim"].values
v = ds["v_clim"].values

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
# En coordenadas latitud-longitud:
#
#       dA ~ cos(phi) dphi dlambda
#
# Normalizamos para que la suma de todos los pesos sea 1.
# ======================================================================
lat_weights = np.cos(
    np.deg2rad(lat)
)

weight_grid = np.repeat(
    lat_weights[:, None],
    nlon,
    axis=1
)

weight_grid = (
    weight_grid
    / np.sum(weight_grid)
)

w = weight_grid.ravel()

sqrt_w = np.sqrt(w)


# ======================================================================
# 4. VECTOR DINÁMICO DE CADA NIVEL
#
# IMPORTANTE:
#
# A diferencia del clustering de patrón:
#
# NO restamos la media
# NO dividimos por la desviación estándar
#
# X_p =
#
# [
# sqrt(w_1) u_1, ..., sqrt(w_N) u_N,
# sqrt(w_1) v_1, ..., sqrt(w_N) v_N
# ]
#
# Por tanto se preservan:
#
# - patrón espacial
# - dirección
# - intensidad
# ======================================================================
features = []

diagnostic_lines = [
    "pressure_hPa,"
    "mean_u_m_s,"
    "mean_v_m_s,"
    "mean_speed_m_s,"
    "rms_speed_m_s\n"
]


for k in range(nlev):

    u_flat = u[k].ravel()
    v_flat = v[k].ravel()

    feature_vector = np.concatenate(
        (
            sqrt_w * u_flat,
            sqrt_w * v_flat
        )
    )

    features.append(
        feature_vector
    )


    # --------------------------------------------------------------
    # Diagnósticos físicos
    # --------------------------------------------------------------
    mean_u = np.sum(
        w * u_flat
    )

    mean_v = np.sum(
        w * v_flat
    )

    speed = np.sqrt(
        u_flat**2 + v_flat**2
    )

    mean_speed = np.sum(
        w * speed
    )

    rms_speed = np.sqrt(
        np.sum(
            w * speed**2
        )
    )

    diagnostic_lines.append(
        f"{levels[k]:.0f},"
        f"{mean_u:.8f},"
        f"{mean_v:.8f},"
        f"{mean_speed:.8f},"
        f"{rms_speed:.8f}\n"
    )


features = np.asarray(features)


with open(
    OUTPUT_DIR / "level_dynamic_diagnostics.csv",
    "w"
) as f:

    f.writelines(
        diagnostic_lines
    )


# ======================================================================
# 5. DISTANCIA DINÁMICA ENTRE NIVELES
#
# ||X_p - X_q|| =
#
# sqrt[
# sum w_ij (
#   (u_p-u_q)^2
#   +
#   (v_p-v_q)^2
# )
# ]
#
# Tiene unidades de m/s.
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
    OUTPUT_DIR / "dynamic_distance_matrix.csv",
    distance_matrix,
    delimiter=",",
    fmt="%.8f"
)


# ======================================================================
# 6. CLUSTERS INICIALES
#
# Cada nivel comienza siendo un cluster independiente.
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

snapshots = {}

requested_K = [
    2,
    3,
    4,
    5,
    6
]


# ======================================================================
# 7. COSTO DE WARD
#
# Delta(A,B) =
#
# n_A n_B
# ---------  ||mu_A - mu_B||²
# n_A+n_B
#
# Ahora mu_A y mu_B contienen los campos físicos originales,
# por lo que la magnitud del viento contribuye al costo.
# ======================================================================
def ward_delta(cluster_a, cluster_b):

    n_a = cluster_a["size"]
    n_b = cluster_b["size"]

    mu_a = cluster_a["centroid"]
    mu_b = cluster_b["centroid"]

    squared_distance = np.sum(
        (mu_a - mu_b)**2
    )

    delta = (
        (n_a * n_b)
        / (n_a + n_b)
        * squared_distance
    )

    return delta


# ======================================================================
# 8. CLUSTERING AGLOMERATIVO
#    CON CONTIGÜIDAD VERTICAL
# ======================================================================
while len(clusters) > 1:

    best_position = None
    best_delta = np.inf


    # --------------------------------------------------------------
    # Solamente podemos fusionar clusters vecinos en presión
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
    # Clusters elegidos
    # --------------------------------------------------------------
    A = clusters[
        best_position
    ]

    B = clusters[
        best_position + 1
    ]


    # --------------------------------------------------------------
    # Nuevo cluster
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
    # Altura para dendrograma compatible con Ward
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


    new_cluster = {
        "id": next_cluster_id,
        "members": new_members,
        "centroid": new_centroid,
        "size": new_size
    }

    next_cluster_id += 1


    # --------------------------------------------------------------
    # Reemplazar A y B manteniendo el orden vertical
    # --------------------------------------------------------------
    clusters = (
        clusters[:best_position]
        + [new_cluster]
        + clusters[
            best_position + 2:
        ]
    )


    # --------------------------------------------------------------
    # Guardar soluciones de interés
    # --------------------------------------------------------------
    K = len(clusters)

    if K in requested_K:

        snapshots[K] = [
            c["members"].copy()
            for c in clusters
        ]


# ======================================================================
# 9. LINKAGE MATRIX
# ======================================================================
linkage_matrix = np.asarray(
    linkage_rows,
    dtype=float
)

np.savetxt(
    OUTPUT_DIR / "constrained_dynamic_ward_linkage.csv",
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
# 10. HISTORIAL DE FUSIONES
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
# 11. SOLUCIONES K = 2 ... 6
# ======================================================================
solution_lines = []

solution_lines.append(
    "Clustering vertical dinámico restringido - ERA5\n"
)

solution_lines.append(
    "===============================================\n\n"
)

solution_lines.append(
    "Se preservan patrón, dirección e intensidad "
    "del viento.\n"
)

solution_lines.append(
    "Cada capa contiene únicamente niveles "
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
# 12. DENDROGRAMA
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
    "Constrained Ward distance [m s$^{-1}$]"
)

ax.set_title(
    "Hierarchical clustering of ERA5 wind dynamics\n"
    "Area-weighted and vertically contiguous"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIG_DIR
    / "ERA5_vertical_dynamic_dendrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ======================================================================
# 13. COSTO DE FUSIÓN
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
    "Increase in within-cluster SSE [(m/s)$^2$]"
)

ax.set_title(
    "Dynamic Ward merge cost"
)

ax.invert_xaxis()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    FIG_DIR
    / "ERA5_vertical_dynamic_merge_cost.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ======================================================================
# 14. MOSTRAR RESULTADOS
# ======================================================================
print()
print("Clustering dinámico terminado")
print("=============================")
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
    / "ERA5_vertical_dynamic_dendrogram.png"
)

print(
    FIG_DIR
    / "ERA5_vertical_dynamic_merge_cost.png"
)


# ======================================================================
# 15. CERRAR
# ======================================================================
ds.close()
