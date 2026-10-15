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
    / "regional_vertical_clustering_no1000"
)

FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 2. REGIONES
#
# Se usa la misma banda latitudinal para reducir diferencias debidas
# solamente al desplazamiento meridional del flujo.
#
# Patagonia:
#     75 W - 60 W
#     55 S - 38 S
#
# Atlántico Sur:
#     55 W - 35 W
#     55 S - 38 S
#
# Dejamos 60-55 W como zona de transición.
# ======================================================================

REGIONS = {
    "Patagonia": {
        "lon_min": -75.0,
        "lon_max": -60.0,
        "lat_min": -55.0,
        "lat_max": -38.0,
    },

    "South_Atlantic": {
        "lon_min": -55.0,
        "lon_max": -35.0,
        "lat_min": -55.0,
        "lat_max": -38.0,
    },
}


# ======================================================================
# 3. ABRIR ERA5
# ======================================================================

ds = xr.open_dataset(ERA5_FILE)

levels_all = ds["pressure_level"].values.astype(float)

# Excluir 1000 hPa
keep_levels = levels_all != 1000.0

levels = levels_all[keep_levels]

u_all = ds["u_clim"].values[keep_levels, :, :]
v_all = ds["v_clim"].values[keep_levels, :, :]

lat_all = ds["latitude"].values
lon_all = ds["longitude"].values

nlev = len(levels)

print()
print("============================================================")
print("COMPARACIÓN REGIONAL DE REGÍMENES VERTICALES ERA5")
print("975-200 hPa")
print("============================================================")
print()


# ======================================================================
# 4. FUNCIONES AUXILIARES
# ======================================================================

def weighted_mean(field, weights):
    return np.sum(weights * field)


def weighted_std(field, weights):
    mean = weighted_mean(field, weights)

    variance = np.sum(
        weights * (field - mean)**2
    )

    return np.sqrt(variance)


# ======================================================================
# 5. CONSTRUIR FEATURES
#
# mode = "pattern":
#
#       u' = (u - <u>) / sigma_u
#       v' = (v - <v>) / sigma_v
#
# Conserva geometría espacial, elimina media y amplitud.
#
#
# mode = "dynamic":
#
#       u, v originales
#
# Conserva patrón + dirección + intensidad.
# ======================================================================

def build_features(u, v, lat, mode):

    nlat = len(lat)
    nlon = u.shape[2]

    # --------------------------------------------------------------
    # Pesos de área
    # --------------------------------------------------------------

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

    features = []


    for k in range(nlev):

        u_flat = u[k].ravel()
        v_flat = v[k].ravel()


        if mode == "pattern":

            mean_u = weighted_mean(
                u_flat,
                w
            )

            mean_v = weighted_mean(
                v_flat,
                w
            )

            std_u = weighted_std(
                u_flat,
                w
            )

            std_v = weighted_std(
                v_flat,
                w
            )

            u_used = (
                (u_flat - mean_u)
                / std_u
            )

            v_used = (
                (v_flat - mean_v)
                / std_v
            )


        elif mode == "dynamic":

            u_used = u_flat
            v_used = v_flat


        else:

            raise ValueError(
                "mode debe ser 'pattern' o 'dynamic'"
            )


        feature_vector = np.concatenate(
            (
                sqrt_w * u_used,
                sqrt_w * v_used
            )
        )

        features.append(
            feature_vector
        )


    return np.asarray(features)


# ======================================================================
# 6. CLUSTERING WARD RESTRINGIDO
# ======================================================================

def constrained_ward(features):

    clusters = []

    for i in range(nlev):

        clusters.append(
            {
                "id": i,
                "members": [i],
                "centroid": features[i].copy(),
                "size": 1,
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
        6,
    ]


    def ward_delta(A, B):

        n_a = A["size"]
        n_b = B["size"]

        squared_distance = np.sum(
            (
                A["centroid"]
                - B["centroid"]
            )**2
        )

        return (
            (n_a * n_b)
            / (n_a + n_b)
            * squared_distance
        )


    # --------------------------------------------------------------
    # Fusiones
    # --------------------------------------------------------------

    while len(clusters) > 1:

        best_position = None
        best_delta = np.inf


        # Solo clusters verticalmente contiguos
        for i in range(
            len(clusters) - 1
        ):

            delta = ward_delta(
                clusters[i],
                clusters[i + 1]
            )

            if delta < best_delta:

                best_delta = delta
                best_position = i


        A = clusters[
            best_position
        ]

        B = clusters[
            best_position + 1
        ]


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


        ward_height = np.sqrt(
            2.0 * best_delta
        )


        linkage_rows.append(
            [
                A["id"],
                B["id"],
                ward_height,
                new_size,
            ]
        )


        pressure_members = levels[
            new_members
        ]

        merge_history.append(
            {
                "clusters_before":
                    len(clusters),

                "clusters_after":
                    len(clusters) - 1,

                "pressure_max":
                    np.max(pressure_members),

                "pressure_min":
                    np.min(pressure_members),

                "delta":
                    best_delta,

                "distance":
                    ward_height,
            }
        )


        new_cluster = {
            "id": next_cluster_id,
            "members": new_members,
            "centroid": new_centroid,
            "size": new_size,
        }

        next_cluster_id += 1


        clusters = (
            clusters[:best_position]
            + [new_cluster]
            + clusters[
                best_position + 2:
            ]
        )


        K = len(clusters)

        if K in requested_K:

            snapshots[K] = [
                c["members"].copy()
                for c in clusters
            ]


    return (
        np.asarray(
            linkage_rows,
            dtype=float
        ),
        merge_history,
        snapshots,
    )


# ======================================================================
# 7. FORMATEAR SOLUCIÓN
# ======================================================================

def format_layers(snapshot):

    output = []

    for members in snapshot:

        p = levels[members]

        output.append(
            (
                int(np.max(p)),
                int(np.min(p)),
                len(members),
            )
        )

    return output


# ======================================================================
# 8. PROCESAR CADA REGIÓN
# ======================================================================

summary_lines = []

summary_lines.append(
    "Regional vertical clustering ERA5 - 975-200 hPa\n"
)

summary_lines.append(
    "================================================\n\n"
)


all_results = {}


for region_name, bounds in REGIONS.items():

    # --------------------------------------------------------------
    # Selección espacial mediante máscaras booleanas
    # --------------------------------------------------------------

    lat_mask = (
        (lat_all >= bounds["lat_min"])
        & (lat_all <= bounds["lat_max"])
    )

    lon_mask = (
        (lon_all >= bounds["lon_min"])
        & (lon_all <= bounds["lon_max"])
    )


    lat = lat_all[
        lat_mask
    ]

    lon = lon_all[
        lon_mask
    ]


    u = u_all[
        :,
        lat_mask,
        :
    ][
        :,
        :,
        lon_mask
    ]

    v = v_all[
        :,
        lat_mask,
        :
    ][
        :,
        :,
        lon_mask
    ]


    print()
    print(
        "============================================================"
    )

    print(
        region_name
    )

    print(
        f"Latitud : "
        f"{lat.min():.2f} a {lat.max():.2f}"
    )

    print(
        f"Longitud: "
        f"{lon.min():.2f} a {lon.max():.2f}"
    )

    print(
        f"Grilla  : "
        f"{len(lat)} x {len(lon)}"
    )

    print(
        "============================================================"
    )


    summary_lines.append(
        f"\n{region_name}\n"
    )

    summary_lines.append(
        "-" * len(region_name)
        + "\n"
    )


    all_results[
        region_name
    ] = {}


    # --------------------------------------------------------------
    # Patrón y dinámica
    # --------------------------------------------------------------

    for mode in [
        "pattern",
        "dynamic",
    ]:

        features = build_features(
            u,
            v,
            lat,
            mode
        )

        (
            linkage_matrix,
            merge_history,
            snapshots,
        ) = constrained_ward(
            features
        )


        all_results[
            region_name
        ][mode] = {
            "linkage":
                linkage_matrix,

            "history":
                merge_history,

            "snapshots":
                snapshots,
        }


        print()
        print(
            f"--- {mode.upper()} ---"
        )


        summary_lines.append(
            f"\n{mode.upper()}\n"
        )


        # ----------------------------------------------------------
        # Mostrar K = 2 ... 6
        # ----------------------------------------------------------

        for K in [
            2,
            3,
            4,
            5,
            6,
        ]:

            layers = format_layers(
                snapshots[K]
            )


            text_layers = []

            for pmax, pmin, n in layers:

                text_layers.append(
                    f"{pmax}-{pmin}"
                )


            print(
                f"K={K}: "
                + " | ".join(
                    text_layers
                )
            )


            summary_lines.append(
                f"K={K}: "
                + " | ".join(
                    text_layers
                )
                + "\n"
            )


        # ----------------------------------------------------------
        # Últimos costos Ward
        # ----------------------------------------------------------

        print()
        print(
            "Últimas fusiones Ward:"
        )

        for row in merge_history[-5:]:

            print(
                f"K "
                f"{row['clusters_before']}"
                f" -> "
                f"{row['clusters_after']}: "
                f"Delta = "
                f"{row['delta']:.6f}"
            )


        # ----------------------------------------------------------
        # Dendrograma
        # ----------------------------------------------------------

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
            ax=ax,
        )


        if mode == "pattern":

            ylabel = (
                "Constrained Ward distance"
            )

            title_mode = (
                "spatial wind patterns"
            )

        else:

            ylabel = (
                "Constrained Ward distance "
                "[m s$^{-1}$]"
            )

            title_mode = (
                "wind dynamics"
            )


        ax.set_xlabel(
            "Pressure level [hPa]"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            f"{region_name}: "
            f"{title_mode}\n"
            "ERA5 2001-2020, "
            "975-200 hPa"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            FIG_DIR
            / (
                f"{region_name}_"
                f"{mode}_dendrogram.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


        # ----------------------------------------------------------
        # Costos Ward
        # ----------------------------------------------------------

        clusters_after = np.array(
            [
                row["clusters_after"]
                for row in merge_history
            ]
        )

        costs = np.array(
            [
                row["delta"]
                for row in merge_history
            ]
        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            clusters_after,
            costs,
            marker="o",
        )

        ax.invert_xaxis()

        ax.set_xlabel(
            "Number of clusters after merge"
        )

        ax.set_ylabel(
            "Increase in within-cluster SSE"
        )

        ax.set_title(
            f"{region_name}: "
            f"{mode} Ward merge cost"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            FIG_DIR
            / (
                f"{region_name}_"
                f"{mode}_merge_cost.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ======================================================================
# 9. GUARDAR RESUMEN
# ======================================================================

SUMMARY_FILE = (
    OUTPUT_DIR
    / "regional_vertical_clusters_summary.txt"
)

with open(
    SUMMARY_FILE,
    "w"
) as f:

    f.writelines(
        summary_lines
    )


# ======================================================================
# 10. FINAL
# ======================================================================

print()
print(
    "============================================================"
)

print(
    "ANÁLISIS REGIONAL TERMINADO"
)

print(
    "============================================================"
)

print()

print(
    "Resumen:"
)

print(
    SUMMARY_FILE
)

print()

print(
    "Figuras:"
)

print(
    FIG_DIR
)

print()


ds.close()
