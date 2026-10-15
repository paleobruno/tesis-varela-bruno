#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import ast

import numpy as np
import xarray as xr


# ============================================================
# RUTAS
# ============================================================

ORIGINAL_SCRIPT = Path.home() / (
    "MERRA2_dust/scripts/"
    "build_MERRA2_3D_local_climatology.py"
)

CHECKPOINT = Path(
    "/mnt/z/MERRA2_dust/checkpoints/"
    "cumulative_through_2020_09.npz"
)

OUTPUT_FILE = Path(
    "/mnt/z/MERRA2_dust/processed/"
    "MERRA2_3D_dust_climatology_2001_2020.nc"
)

START_DATE = "2001-01-01"


# ============================================================
# LEER CONSTANTES DEL SCRIPT ORIGINAL
# ============================================================

def read_literal(name):
    """
    Lee de forma segura una constante literal definida
    en el script original, sin ejecutar dicho script.
    """

    source = ORIGINAL_SCRIPT.read_text()

    tree = ast.parse(source)

    for node in tree.body:

        if isinstance(node, ast.Assign):

            for target in node.targets:

                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                ):

                    return ast.literal_eval(
                        node.value
                    )

        elif isinstance(node, ast.AnnAssign):

            if (
                isinstance(node.target, ast.Name)
                and node.target.id == name
            ):

                return ast.literal_eval(
                    node.value
                )

    raise RuntimeError(
        f"No se encontró la constante {name} "
        "en el script original."
    )


REGIMES = read_literal("REGIMES")
G0 = float(read_literal("G0"))
P_TOP_HPA = float(
    read_literal("P_TOP_HPA")
)


# ============================================================
# CARGAR CHECKPOINT
# ============================================================

def load_checkpoint(path):

    with np.load(
        path,
        allow_pickle=False
    ) as data:

        accumulator = {

            "n_days":
                int(data["n_days"]),

            "n_samples":
                int(data["n_samples"]),

            "last_date":
                str(
                    data["last_date"].item()
                ),

            "lat":
                data["lat"].copy(),

            "lon":
                data["lon"].copy(),

            "lev":
                data["lev"].copy(),

            "qd_sum":
                data["qd_sum"].copy(),

            "delp_sum":
                data["delp_sum"].copy(),

            "pmid_sum":
                data["pmid_sum"].copy(),

            "dust_mass_layer_sum":
                data[
                    "dust_mass_layer_sum"
                ].copy(),

            "dust_column_mass_sum":
                data[
                    "dust_column_mass_sum"
                ].copy(),

            "ps_sum":
                data["ps_sum"].copy(),

            "max_closure_error":
                float(
                    data[
                        "max_closure_error"
                    ]
                ),

            "max_ps_error":
                float(
                    data[
                        "max_ps_error"
                    ]
                ),

            "regime_sums": {
                name:
                    data[
                        f"regime__{name}"
                    ].copy()

                for name in REGIMES
            },
        }

    return accumulator


# ============================================================
# CONSTRUIR Y GUARDAR CLIMATOLOGÍA
# ============================================================

def write_final_netcdf(
    accumulator
):

    n_days = accumulator[
        "n_days"
    ]

    n_samples = accumulator[
        "n_samples"
    ]

    last_date = accumulator[
        "last_date"
    ]

    # ========================================================
    # 1. VALIDACIONES
    # ========================================================

    if n_days != 7213:

        raise RuntimeError(
            f"Se esperaban 7213 días, "
            f"pero hay {n_days}."
        )

    if n_samples != 57704:

        raise RuntimeError(
            f"Se esperaban 57704 tiempos, "
            f"pero hay {n_samples}."
        )

    if n_samples != n_days * 8:

        raise RuntimeError(
            "El número de muestras no corresponde "
            "a 8 tiempos diarios."
        )

    if last_date != "2020-09-30":

        raise RuntimeError(
            f"Fecha final inesperada: "
            f"{last_date}"
        )

    print()
    print("=" * 72)
    print("CHECKPOINT VALIDADO")
    print("=" * 72)

    print(
        f"Período:             "
        f"{START_DATE} a {last_date}"
    )

    print(
        f"Días procesados:     "
        f"{n_days}"
    )

    print(
        f"Tiempos procesados:  "
        f"{n_samples}"
    )


    # ========================================================
    # 2. MEDIAS CLIMATOLÓGICAS
    # ========================================================

    qd_mean = (
        accumulator["qd_sum"]
        / n_samples
    )

    delp_mean = (
        accumulator["delp_sum"]
        / n_samples
    )

    pmid_mean = (
        accumulator["pmid_sum"]
        / n_samples
    )

    dust_mass_layer_mean = (
        accumulator[
            "dust_mass_layer_sum"
        ]
        / n_samples
    )

    dust_column_mass_mean = (
        accumulator[
            "dust_column_mass_sum"
        ]
        / n_samples
    )

    ps_mean = (
        accumulator["ps_sum"]
        / n_samples
    )

    regime_means = {
        name:
            accumulator[
                "regime_sums"
            ][name]
            / n_samples

        for name in REGIMES
    }


    # ========================================================
    # 3. FRACCIONES DE MASA
    # ========================================================

    regime_fractions = {}

    for name in REGIMES:

        regime_fractions[name] = (
            np.divide(
                regime_means[name],
                dust_column_mass_mean,
                out=np.zeros_like(
                    dust_column_mass_mean,
                    dtype=np.float64
                ),
                where=(
                    dust_column_mass_mean
                    > 0.0
                )
            )
        )


    # ========================================================
    # 4. CONSERVACIÓN DE MASA
    # ========================================================

    column_from_layers = np.sum(
        dust_mass_layer_mean,
        axis=0
    )

    layer_closure_error = np.max(
        np.abs(
            column_from_layers
            - dust_column_mass_mean
        )
    )

    column_from_regimes = (
        np.zeros_like(
            dust_column_mass_mean,
            dtype=np.float64
        )
    )

    for name in REGIMES:

        column_from_regimes += (
            regime_means[name]
        )

    regime_closure_error = np.max(
        np.abs(
            column_from_regimes
            - dust_column_mass_mean
        )
    )

    print()
    print("=" * 72)
    print(
        "VALIDACIÓN DE LA CLIMATOLOGÍA FINAL"
    )
    print("=" * 72)

    print(
        "Error máximo columna vs 72 capas: "
        f"{layer_closure_error:.3e} kg/m²"
    )

    print(
        "Error máximo columna vs regímenes: "
        f"{regime_closure_error:.3e} kg/m²"
    )

    if layer_closure_error > 1e-10:

        raise RuntimeError(
            "Falló el cierre de masa entre "
            "las 72 capas y la columna."
        )

    if regime_closure_error > 1e-10:

        raise RuntimeError(
            "Falló el cierre de masa entre "
            "los regímenes y la columna."
        )


    # ========================================================
    # 5. DATASET XARRAY
    # ========================================================

    coords = {

        "lev":
            accumulator["lev"],

        "lat":
            accumulator["lat"],

        "lon":
            accumulator["lon"],
    }

    data_vars = {

        "dust_mixing_ratio": (
            ("lev", "lat", "lon"),
            qd_mean
        ),

        "pressure_thickness": (
            ("lev", "lat", "lon"),
            delp_mean
        ),

        "pressure_midpoint": (
            ("lev", "lat", "lon"),
            pmid_mean
        ),

        "dust_mass_layer": (
            ("lev", "lat", "lon"),
            dust_mass_layer_mean
        ),

        "dust_column_mass": (
            ("lat", "lon"),
            dust_column_mass_mean
        ),

        "surface_pressure": (
            ("lat", "lon"),
            ps_mean
        ),
    }

    for name in REGIMES:

        data_vars[name] = (
            ("lat", "lon"),
            regime_means[name]
        )

        fraction_name = (
            name.replace(
                "mass_",
                "fraction_"
            )
        )

        data_vars[fraction_name] = (
            ("lat", "lon"),
            regime_fractions[name]
        )

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords=coords
    )


    # ========================================================
    # 6. ATRIBUTOS DE COORDENADAS
    # ========================================================

    ds_out["lon"].attrs = {
        "long_name":
            "longitude",

        "standard_name":
            "longitude",

        "units":
            "degrees_east",
    }

    ds_out["lat"].attrs = {
        "long_name":
            "latitude",

        "standard_name":
            "latitude",

        "units":
            "degrees_north",
    }

    ds_out["lev"].attrs = {
        "long_name":
            "MERRA-2 native model layer index",

        "description":
            "Layer 1 is the model top; "
            "layer 72 is nearest the surface.",
    }


    # ========================================================
    # 7. ATRIBUTOS DE VARIABLES
    # ========================================================

    ds_out[
        "dust_mixing_ratio"
    ].attrs = {

        "long_name":
            "Climatological mean total dust mixing ratio",

        "description":
            "DU001 + DU002 + DU003 + DU004 + DU005",

        "units":
            "kg kg-1",
    }

    ds_out[
        "pressure_thickness"
    ].attrs = {

        "long_name":
            "Climatological mean layer pressure thickness",

        "units":
            "Pa",
    }

    ds_out[
        "pressure_midpoint"
    ].attrs = {

        "long_name":
            "Climatological mean pressure at layer midpoint",

        "units":
            "hPa",

        "description":
            "Reconstructed from DELP using "
            "a nominal model-top pressure of 0.01 hPa.",
    }

    ds_out[
        "dust_mass_layer"
    ].attrs = {

        "long_name":
            "Climatological mean dust mass per unit area "
            "in each model layer",

        "units":
            "kg m-2",

        "formula":
            "mean(q_d * DELP / g)",
    }

    ds_out[
        "dust_column_mass"
    ].attrs = {

        "long_name":
            "Climatological mean atmospheric dust column mass",

        "units":
            "kg m-2",
    }

    ds_out[
        "surface_pressure"
    ].attrs = {

        "long_name":
            "Climatological mean surface pressure",

        "units":
            "Pa",
    }

    for (
        name,
        (
            p_upper,
            p_lower
        )
    ) in REGIMES.items():

        ds_out[name].attrs = {

            "long_name":
                "Climatological mean dust mass "
                f"between {p_upper:g} and "
                f"{p_lower:g} hPa",

            "units":
                "kg m-2",

            "pressure_upper_hPa":
                float(p_upper),

            "pressure_lower_hPa":
                float(p_lower),
        }

        fraction_name = (
            name.replace(
                "mass_",
                "fraction_"
            )
        )

        ds_out[
            fraction_name
        ].attrs = {

            "long_name":
                "Fraction of atmospheric dust column mass "
                f"between {p_upper:g} and "
                f"{p_lower:g} hPa",

            "units":
                "1",

            "description":
                "Multiply by 100 to obtain percent.",
        }


    # ========================================================
    # 8. ATRIBUTOS GLOBALES
    # ========================================================

    ds_out.attrs = {

        "title":
            "MERRA-2 3D dust climatology over southern "
            "South America and the South Atlantic",

        "source":
            "NASA MERRA-2 M2I3NVAER V5.12.4",

        "product":
            "inst3_3d_aer_Nv",

        "temporal_resolution_source":
            "3-hourly instantaneous",

        "climatology_period":
            f"{START_DATE} to {last_date}",

        "number_of_days":
            int(n_days),

        "number_of_time_samples":
            int(n_samples),

        "domain":
            "80W-20W, 60S-35S",

        "gravity_m_s2":
            float(G0),

        "nominal_model_top_pressure_hPa":
            float(P_TOP_HPA),

        "dust_definition":
            "DU001 + DU002 + DU003 + DU004 + DU005",

        "vertical_mass_formula":
            "m_d = q_d * DELP / g",

        "vertical_regimes_hPa":
            "0-200; 200-425; 425-725; 725-975; >975",

        "regime_method":
            "Layers crossing regime boundaries were split "
            "according to their exact pressure overlap.",

        "processing_note":
            "All temporal products are arithmetic means "
            f"of the {n_samples} native 3-hourly "
            "instantaneous samples from "
            f"{START_DATE} through {last_date}. "
            "Dust mass was calculated at each instant "
            "before temporal averaging.",

        "maximum_daily_mass_closure_error_kg_m2":
            float(
                accumulator[
                    "max_closure_error"
                ]
            ),

        "maximum_PS_DELP_discrepancy_Pa":
            float(
                accumulator[
                    "max_ps_error"
                ]
            ),

        "final_layer_closure_error_kg_m2":
            float(
                layer_closure_error
            ),

        "final_regime_closure_error_kg_m2":
            float(
                regime_closure_error
            ),

        "created":
            datetime.now().isoformat(
                timespec="seconds"
            ),
    }


    # ========================================================
    # 9. COMPRESIÓN NETCDF
    # ========================================================

    encoding = {}

    for variable in ds_out.data_vars:

        encoding[variable] = {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
        }


    # ========================================================
    # 10. ESCRITURA SEGURA
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_output = (
        OUTPUT_FILE.parent
        / (
            OUTPUT_FILE.stem
            + ".tmp.nc"
        )
    )

    if temp_output.exists():

        temp_output.unlink()

    print()
    print(
        "Escribiendo NetCDF final..."
    )

    ds_out.to_netcdf(
        temp_output,
        engine="netcdf4",
        encoding=encoding
    )

    ds_out.close()

    temp_output.replace(
        OUTPUT_FILE
    )

    print()
    print("=" * 72)
    print("NETCDF FINAL CREADO")
    print("=" * 72)

    print(
        OUTPUT_FILE
    )

    size_mb = (
        OUTPUT_FILE.stat().st_size
        / 1024**2
    )

    print()

    print(
        f"Tamaño final: "
        f"{size_mb:.2f} MB"
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        f"Cargando checkpoint:\n"
        f"{CHECKPOINT}"
    )

    accumulator = (
        load_checkpoint(
            CHECKPOINT
        )
    )

    write_final_netcdf(
        accumulator
    )
