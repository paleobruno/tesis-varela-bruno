from pathlib import Path
from datetime import datetime
import re
import subprocess
import sys

import numpy as np
import xarray as xr


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path.home() / "MERRA2_dust"

LINK_FILE = (
    BASE_DIR
    / "links"
    / "subset_M2I3NVAER_5.12.4_20260914_051028_.txt"
)

# Carpeta temporal:
# cada archivo diario se descargará aquí,
# se procesará y luego se eliminará.
TEMP_DIR = BASE_DIR / "raw_3d" / "temp"

# Checkpoints anuales.
CHECKPOINT_DIR = (
    BASE_DIR
    / "processed"
    / "vertical_regimes"
    / "checkpoints"
)

# Archivo climatológico final.
OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "MERRA2_3D_dust_climatology_2001_2020.nc"
)

# Cookies de Earthdata.
COOKIE_FILE = Path.home() / ".urs_cookies"


# ============================================================
# CONSTANTES FÍSICAS
# ============================================================

G0 = 9.80665  # m s^-2

# Borde superior nominal de MERRA-2.
#
# PS - sum(DELP) mostró fluctuaciones de unas centésimas
# de hPa debido al redondeo float32. Para reconstruir
# los bordes verticales usamos un valor fijo y físicamente
# representativo.
P_TOP_HPA = 0.01


# ============================================================
# REGÍMENES VERTICALES
# ============================================================

# Límites definidos a partir de nuestro análisis dinámico.
#
# Las tuplas son:
# (presión superior, presión inferior), en hPa.

REGIMES = {
    "mass_p_lt_200":     (0.0,   200.0),
    "mass_200_425":      (200.0, 425.0),
    "mass_425_725":      (425.0, 725.0),
    "mass_725_975":      (725.0, 975.0),
    "mass_p_gt_975":     (975.0, 2000.0),
}


# ============================================================
# CREAR CARPETAS
# ============================================================

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LEER LISTA DE URLs
# ============================================================

def load_daily_urls(link_file):
    """
    Lee la lista generada por NASA GES DISC.

    Devuelve una lista:

        [(datetime, url), ...]

    conteniendo únicamente los 7305 archivos diarios
    M2I3NVAER entre 2001 y 2020.
    """

    records = []

    with open(
        link_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            # strip() elimina tanto '\n' como '\r'.
            # Esto evita que aparezca '%0D' al final
            # de las URLs.
            url = line.strip()

            if not url:
                continue

            # Ignorar README y cualquier enlace
            # que no corresponda a un subset diario.
            if "SUB.nc" not in url:
                continue

            match = re.search(
                r"(\d{8})\.SUB\.nc",
                url
            )

            if match is None:
                continue

            date_string = match.group(1)

            date = datetime.strptime(
                date_string,
                "%Y%m%d"
            )

            records.append(
                (date, url)
            )

    records.sort(
        key=lambda item: item[0]
    )

    return records


# ============================================================
# CARGAR Y VALIDAR LISTA
# ============================================================

DAILY_URLS = load_daily_urls(
    LINK_FILE
)

print()
print("=" * 72)
print("MERRA-2 3D DUST CLIMATOLOGY 2001-2020")
print("=" * 72)

print(
    f"URLs diarias encontradas: "
    f"{len(DAILY_URLS)}"
)

if len(DAILY_URLS) != 7305:

    raise RuntimeError(
        "Se esperaban 7305 archivos diarios, "
        f"pero se encontraron {len(DAILY_URLS)}."
    )

dates = [
    date
    for date, url in DAILY_URLS
]

if len(set(dates)) != 7305:

    raise RuntimeError(
        "La lista contiene fechas duplicadas."
    )

print(
    f"Primera fecha: "
    f"{dates[0]:%Y-%m-%d}"
)

print(
    f"Última fecha:  "
    f"{dates[-1]:%Y-%m-%d}"
)

print(
    "Lista de URLs validada correctamente."
)

print()

# ============================================================
# DESCARGA Y VALIDACIÓN DE ARCHIVOS DIARIOS
# ============================================================

from urllib.parse import urlparse, parse_qs


def filename_from_url(url):
    """
    Extrae el nombre indicado por LABEL en la URL de GES DISC.

    Esto es preferible a construir el nombre manualmente,
    porque el prefijo MERRA2_300 / MERRA2_400 cambia según
    el período de MERRA-2.
    """

    query = parse_qs(
        urlparse(url).query
    )

    labels = query.get(
        "LABEL",
        []
    )

    if len(labels) != 1:

        raise RuntimeError(
            "No fue posible extraer LABEL "
            "desde la URL."
        )

    return labels[0]


def download_daily_file(date, url):
    """
    Descarga un único subset diario a TEMP_DIR.

    Si el archivo ya existe y es válido, lo reutiliza.

    Devuelve la ruta local.
    """

    filename = filename_from_url(
        url
    )

    local_path = (
        TEMP_DIR
        / filename
    )

    # --------------------------------------------------------
    # Si quedó un archivo de una ejecución anterior,
    # comprobar si es utilizable antes de descargar de nuevo.
    # --------------------------------------------------------

    if local_path.exists():

        try:

            validate_daily_file(
                local_path,
                date
            )

            print(
                f"  Archivo existente válido: "
                f"{filename}"
            )

            return local_path

        except Exception:

            print(
                "  Archivo temporal incompleto "
                "o inválido. Se descargará nuevamente."
            )

            local_path.unlink(
                missing_ok=True
            )

    # --------------------------------------------------------
    # Comando wget
    # --------------------------------------------------------

    command = [
        "wget",

        "--save-cookies",
        str(COOKIE_FILE),

        "--keep-session-cookies",

        "--auth-no-challenge=on",

        "--netrc",

        "-O",
        str(local_path),
    ]

    # Solo intentar cargar cookies si ya existen.
    # En la primera descarga pueden no existir todavía.
    if COOKIE_FILE.exists():

        command.extend([
            "--load-cookies",
            str(COOKIE_FILE),
        ])

    command.append(
        url
    )

    print(
        f"  Descargando "
        f"{date:%Y-%m-%d} ..."
    )

    try:

        subprocess.run(
            command,
            check=True
        )

    except subprocess.CalledProcessError as exc:

        # Eliminar cualquier archivo parcial.
        local_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"Falló la descarga de "
            f"{date:%Y-%m-%d}"
        ) from exc

    # --------------------------------------------------------
    # Validar inmediatamente después de descargar.
    # --------------------------------------------------------

    validate_daily_file(
        local_path,
        date
    )

    return local_path


def validate_daily_file(path, expected_date):
    """
    Comprueba que el archivo descargado sea realmente
    un NetCDF MERRA-2 diario utilizable.

    Se verifican:

      - variables necesarias;
      - dimensiones;
      - 8 tiempos diarios;
      - 72 niveles;
      - grilla 51 x 97;
      - fecha correcta.
    """

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    # Un HTML de login o una descarga fallida suele ser
    # diminuto comparado con un NetCDF real.
    if path.stat().st_size < 1_000_000:

        raise RuntimeError(
            f"Archivo sospechosamente pequeño: "
            f"{path.stat().st_size} bytes"
        )

    required_variables = {
        "DELP",
        "DU001",
        "DU002",
        "DU003",
        "DU004",
        "DU005",
        "PS",
    }

    try:

        with xr.open_dataset(
            path
        ) as ds:

            missing = (
                required_variables
                - set(ds.data_vars)
            )

            if missing:

                raise RuntimeError(
                    "Faltan variables: "
                    + ", ".join(
                        sorted(missing)
                    )
                )

            expected_sizes = {
                "time": 8,
                "lev": 72,
                "lat": 51,
                "lon": 97,
            }

            for dim, expected_size in expected_sizes.items():

                actual_size = ds.sizes.get(
                    dim
                )

                if actual_size != expected_size:

                    raise RuntimeError(
                        f"Dimensión {dim}: "
                        f"esperada {expected_size}, "
                        f"obtenida {actual_size}"
                    )

            # ------------------------------------------------
            # Verificar fecha del primer timestep.
            # ------------------------------------------------

            first_time = np.datetime64(
                ds["time"].values[0],
                "D"
            )

            expected_np_date = np.datetime64(
                expected_date.date()
            )

            if first_time != expected_np_date:

                raise RuntimeError(
                    "La fecha interna del archivo "
                    "no coincide con la URL."
                )

            # ------------------------------------------------
            # Verificar unidades esenciales.
            # ------------------------------------------------

            if ds["DELP"].attrs.get(
                "units",
                ""
            ) != "Pa":

                raise RuntimeError(
                    "DELP no está expresado en Pa."
                )

            for variable in [
                "DU001",
                "DU002",
                "DU003",
                "DU004",
                "DU005",
            ]:

                units = ds[
                    variable
                ].attrs.get(
                    "units",
                    ""
                )

                if units != "kg kg-1":

                    raise RuntimeError(
                        f"{variable} tiene unidades "
                        f"inesperadas: {units}"
                    )

            if ds["PS"].attrs.get(
                "units",
                ""
            ) != "Pa":

                raise RuntimeError(
                    "PS no está expresado en Pa."
                )

    except Exception as exc:

        raise RuntimeError(
            f"El archivo diario no pasó "
            f"la validación: {path.name}"
        ) from exc


def remove_daily_file(path):
    """
    Elimina el NetCDF diario después de que sus datos
    hayan sido incorporados correctamente al acumulador.
    """

    if path.exists():

        path.unlink()

# ============================================================
# NÚCLEO FÍSICO DEL PROCESAMIENTO
# ============================================================

def pressure_overlap(
    p_upper,
    p_lower,
    regime_upper,
    regime_lower
):
    """
    Calcula el espesor de presión que una capa atmosférica
    comparte con un régimen vertical.

    Parámetros
    ----------
    p_upper, p_lower : ndarray
        Bordes superior e inferior de cada capa [hPa].

    regime_upper, regime_lower : float
        Límites del régimen [hPa].

    Devuelve
    --------
    overlap : ndarray
        Espesor de presión compartido [hPa].
    """

    return np.maximum(
        0.0,
        np.minimum(
            p_lower,
            regime_lower
        )
        - np.maximum(
            p_upper,
            regime_upper
        )
    )


def process_daily_file(path):
    """
    Procesa un archivo diario MERRA-2 de 8 tiempos.

    Para cada día calcula y suma sobre sus 8 tiempos:

      - mezcla total de polvo q_d;
      - DELP;
      - presión media de cada capa;
      - masa de polvo por capa;
      - masa total de la columna;
      - masa en cada régimen vertical;
      - presión superficial.

    Las sumas diarias se incorporarán posteriormente
    al acumulador climatológico 2001-2020.
    """

    with xr.open_dataset(
        path
    ) as ds:

        # ====================================================
        # 1. LEER VARIABLES
        # ====================================================

        delp = (
            ds["DELP"]
            .values
            .astype(
                np.float64
            )
        )

        ps = (
            ds["PS"]
            .values
            .astype(
                np.float64
            )
        )

        # ----------------------------------------------------
        # Mezcla total de polvo:
        #
        # q_d = DU001 + ... + DU005
        #
        # [kg dust / kg air]
        # ----------------------------------------------------

        qd = np.zeros_like(
            delp,
            dtype=np.float64
        )

        for variable in [
            "DU001",
            "DU002",
            "DU003",
            "DU004",
            "DU005",
        ]:

            qd += (
                ds[variable]
                .values
                .astype(
                    np.float64
                )
            )

        # ====================================================
        # 2. VERIFICACIONES BÁSICAS
        # ====================================================

        if not np.all(
            np.isfinite(delp)
        ):

            raise RuntimeError(
                "DELP contiene valores no finitos."
            )

        if not np.all(
            np.isfinite(qd)
        ):

            raise RuntimeError(
                "El campo de polvo contiene "
                "valores no finitos."
            )

        if not np.all(
            np.isfinite(ps)
        ):

            raise RuntimeError(
                "PS contiene valores no finitos."
            )

        if np.any(
            delp < 0.0
        ):

            raise RuntimeError(
                "Se encontraron valores negativos "
                "de DELP."
            )

        if np.any(
            qd < 0.0
        ):

            raise RuntimeError(
                "Se encontraron razones de mezcla "
                "de polvo negativas."
            )

        # ====================================================
        # 3. RECONSTRUIR PRESIÓN DE CADA CAPA
        # ====================================================

        # DELP viene en Pa.
        # Trabajamos los bordes en hPa.

        delp_hpa = (
            delp / 100.0
        )

        ntime, nlev, nlat, nlon = (
            delp.shape
        )

        # ----------------------------------------------------
        # edges tiene dimensión:
        #
        # (time, 73, lat, lon)
        #
        # porque 72 capas tienen 73 bordes.
        # ----------------------------------------------------

        pressure_edges = np.empty(
            (
                ntime,
                nlev + 1,
                nlat,
                nlon
            ),
            dtype=np.float64
        )

        pressure_edges[:, 0, :, :] = (
            P_TOP_HPA
        )

        pressure_edges[:, 1:, :, :] = (
            P_TOP_HPA
            + np.cumsum(
                delp_hpa,
                axis=1
            )
        )

        p_upper = (
            pressure_edges[
                :, :-1, :, :
            ]
        )

        p_lower = (
            pressure_edges[
                :, 1:, :, :
            ]
        )

        # Presión media aproximada de cada capa.

        p_mid = (
            0.5
            * (
                p_upper
                + p_lower
            )
        )

        # ====================================================
        # 4. MASA DE POLVO POR CAPA
        # ====================================================

        # Hidrostática:
        #
        # dm_air / dA = DELP / g
        #
        # Por tanto:
        #
        # dm_dust / dA
        #     = q_d * DELP / g
        #
        # unidades:
        #
        # kg dust / m²

        dust_mass_layer = (
            qd
            * delp
            / G0
        )

        # Masa total de la columna para cada tiempo:
        #
        # (time, lat, lon)

        dust_column_mass = np.sum(
            dust_mass_layer,
            axis=1
        )

        # ====================================================
        # 5. MASA POR RÉGIMEN VERTICAL
        # ====================================================

        regime_sums = {}

        for (
            regime_name,
            (
                regime_upper,
                regime_lower
            )
        ) in REGIMES.items():

            overlap_hpa = pressure_overlap(
                p_upper,
                p_lower,
                regime_upper,
                regime_lower
            )

            overlap_pa = (
                overlap_hpa
                * 100.0
            )

            # -----------------------------------------------
            # Si una capa cruza una frontera entre regímenes,
            # solo asignamos la fracción correspondiente de
            # su DELP.
            #
            # Esto evita clasificar toda la capa por su
            # presión media.
            # -----------------------------------------------

            dust_mass_regime = np.sum(
                qd
                * overlap_pa
                / G0,
                axis=1
            )

            # Sumar los ocho tiempos del día.

            regime_sums[
                regime_name
            ] = np.sum(
                dust_mass_regime,
                axis=0
            )

        # ====================================================
        # 6. CIERRE DE MASA DEL DÍA
        # ====================================================

        total_from_regimes = np.zeros(
            (
                ntime,
                nlat,
                nlon
            ),
            dtype=np.float64
        )

        for (
            regime_upper,
            regime_lower
        ) in REGIMES.values():

            overlap_pa = (
                pressure_overlap(
                    p_upper,
                    p_lower,
                    regime_upper,
                    regime_lower
                )
                * 100.0
            )

            total_from_regimes += np.sum(
                qd
                * overlap_pa
                / G0,
                axis=1
            )

        closure_error = np.max(
            np.abs(
                total_from_regimes
                - dust_column_mass
            )
        )

        # Un error muy pequeño es normal por aritmética
        # de punto flotante.

        if closure_error > 1e-10:

            raise RuntimeError(
                "El cierre de masa vertical falló. "
                f"Error máximo = "
                f"{closure_error:.3e} kg/m²"
            )

        # ====================================================
        # 7. DIAGNÓSTICO DE PRESIÓN SUPERFICIAL
        # ====================================================

        reconstructed_ps = (
            pressure_edges[
                :, -1, :, :
            ]
            * 100.0
        )

        ps_error = np.max(
            np.abs(
                reconstructed_ps
                - ps
            )
        )

        # No exigimos igualdad exacta porque PS y DELP
        # están almacenados en float32 y estamos usando
        # P_TOP_HPA = 0.01 hPa fijo.

        # ====================================================
        # 8. SUMAS DEL DÍA
        # ====================================================

        daily_result = {

            # Número de observaciones temporales.
            "n_samples":
                ntime,

            # Coordenadas.
            "lat":
                ds["lat"].values.astype(
                    np.float64
                ),

            "lon":
                ds["lon"].values.astype(
                    np.float64
                ),

            "lev":
                ds["lev"].values.astype(
                    np.float64
                ),

            # -----------------------------------------------
            # Sumas sobre los 8 tiempos.
            # Dimensión: (lev, lat, lon)
            # -----------------------------------------------

            "qd_sum":
                np.sum(
                    qd,
                    axis=0
                ),

            "delp_sum":
                np.sum(
                    delp,
                    axis=0
                ),

            "pmid_sum":
                np.sum(
                    p_mid,
                    axis=0
                ),

            "dust_mass_layer_sum":
                np.sum(
                    dust_mass_layer,
                    axis=0
                ),

            # -----------------------------------------------
            # Dimensión: (lat, lon)
            # -----------------------------------------------

            "dust_column_mass_sum":
                np.sum(
                    dust_column_mass,
                    axis=0
                ),

            "ps_sum":
                np.sum(
                    ps,
                    axis=0
                ),

            # Masas de los cinco regímenes.
            "regime_sums":
                regime_sums,

            # Diagnósticos escalares.
            "closure_error":
                float(
                    closure_error
                ),

            "ps_error":
                float(
                    ps_error
                ),
        }

    return daily_result

# ============================================================
# ACUMULADOR CLIMATOLÓGICO
# ============================================================

def create_accumulator(daily_result):
    """
    Crea un acumulador vacío usando la geometría del
    primer archivo diario válido.
    """

    accumulator = {
        "n_days": 0,
        "n_samples": 0,
        "last_date": "",

        "lat":
            daily_result["lat"].copy(),

        "lon":
            daily_result["lon"].copy(),

        "lev":
            daily_result["lev"].copy(),

        # -----------------------------------------------
        # Campos 3D: (lev, lat, lon)
        # -----------------------------------------------

        "qd_sum":
            np.zeros_like(
                daily_result["qd_sum"],
                dtype=np.float64
            ),

        "delp_sum":
            np.zeros_like(
                daily_result["delp_sum"],
                dtype=np.float64
            ),

        "pmid_sum":
            np.zeros_like(
                daily_result["pmid_sum"],
                dtype=np.float64
            ),

        "dust_mass_layer_sum":
            np.zeros_like(
                daily_result["dust_mass_layer_sum"],
                dtype=np.float64
            ),

        # -----------------------------------------------
        # Campos 2D: (lat, lon)
        # -----------------------------------------------

        "dust_column_mass_sum":
            np.zeros_like(
                daily_result["dust_column_mass_sum"],
                dtype=np.float64
            ),

        "ps_sum":
            np.zeros_like(
                daily_result["ps_sum"],
                dtype=np.float64
            ),

        # -----------------------------------------------
        # Masas por régimen
        # -----------------------------------------------

        "regime_sums": {
            name:
                np.zeros_like(
                    daily_result[
                        "regime_sums"
                    ][name],
                    dtype=np.float64
                )

            for name in REGIMES
        },

        # -----------------------------------------------
        # Diagnósticos máximos encontrados
        # -----------------------------------------------

        "max_closure_error": 0.0,
        "max_ps_error": 0.0,
    }

    return accumulator


def check_coordinates(
    accumulator,
    daily_result
):
    """
    Comprueba que todos los archivos diarios utilicen
    exactamente la misma grilla horizontal y vertical.
    """

    for coordinate in [
        "lat",
        "lon",
        "lev",
    ]:

        if not np.array_equal(
            accumulator[coordinate],
            daily_result[coordinate]
        ):

            raise RuntimeError(
                "La coordenada "
                f"{coordinate} cambió entre archivos."
            )


def add_daily_result(
    accumulator,
    daily_result,
    date
):
    """
    Incorpora las sumas de un día al acumulador
    climatológico.
    """

    check_coordinates(
        accumulator,
        daily_result
    )

    # ====================================================
    # SUMAR CAMPOS 3D
    # ====================================================

    accumulator["qd_sum"] += (
        daily_result["qd_sum"]
    )

    accumulator["delp_sum"] += (
        daily_result["delp_sum"]
    )

    accumulator["pmid_sum"] += (
        daily_result["pmid_sum"]
    )

    accumulator[
        "dust_mass_layer_sum"
    ] += (
        daily_result[
            "dust_mass_layer_sum"
        ]
    )

    # ====================================================
    # SUMAR CAMPOS 2D
    # ====================================================

    accumulator[
        "dust_column_mass_sum"
    ] += (
        daily_result[
            "dust_column_mass_sum"
        ]
    )

    accumulator["ps_sum"] += (
        daily_result["ps_sum"]
    )

    # ====================================================
    # SUMAR REGÍMENES
    # ====================================================

    for name in REGIMES:

        accumulator[
            "regime_sums"
        ][name] += (
            daily_result[
                "regime_sums"
            ][name]
        )

    # ====================================================
    # CONTADORES
    # ====================================================

    accumulator["n_days"] += 1

    accumulator["n_samples"] += (
        daily_result["n_samples"]
    )

    accumulator["last_date"] = (
        date.strftime(
            "%Y-%m-%d"
        )
    )

    # ====================================================
    # DIAGNÓSTICOS
    # ====================================================

    accumulator[
        "max_closure_error"
    ] = max(
        accumulator[
            "max_closure_error"
        ],
        daily_result[
            "closure_error"
        ]
    )

    accumulator[
        "max_ps_error"
    ] = max(
        accumulator[
            "max_ps_error"
        ],
        daily_result[
            "ps_error"
        ]
    )


# ============================================================
# CHECKPOINTS
# ============================================================

def checkpoint_path(year):
    """
    Nombre del checkpoint acumulativo al terminar un año.
    """

    return (
        CHECKPOINT_DIR
        / f"cumulative_through_{year}.npz"
    )


def expected_days_through_year(year):
    """
    Número exacto de días desde 2001-01-01 hasta
    el 31 de diciembre del año indicado.
    """

    start = datetime(
        2001,
        1,
        1
    )

    end = datetime(
        year + 1,
        1,
        1
    )

    return (
        end - start
    ).days


def save_checkpoint(
    accumulator,
    year
):
    """
    Guarda un checkpoint acumulativo y comprueba antes
    que no falte ningún día ni timestep.
    """

    expected_days = (
        expected_days_through_year(
            year
        )
    )

    expected_samples = (
        expected_days
        * 8
    )

    if (
        accumulator["n_days"]
        != expected_days
    ):

        raise RuntimeError(
            f"No se puede guardar el checkpoint "
            f"{year}: se esperaban "
            f"{expected_days} días y hay "
            f"{accumulator['n_days']}."
        )

    if (
        accumulator["n_samples"]
        != expected_samples
    ):

        raise RuntimeError(
            f"No se puede guardar el checkpoint "
            f"{year}: se esperaban "
            f"{expected_samples} tiempos y hay "
            f"{accumulator['n_samples']}."
        )

    path = checkpoint_path(
        year
    )

    # Guardaremos primero en un archivo temporal.
    # Solo si termina correctamente sustituirá al
    # checkpoint definitivo.

    temp_path = path.with_suffix(
        ".tmp.npz"
    )

    payload = {
        "n_days":
            np.array(
                accumulator["n_days"],
                dtype=np.int64
            ),

        "n_samples":
            np.array(
                accumulator["n_samples"],
                dtype=np.int64
            ),

        "last_date":
            np.array(
                accumulator["last_date"]
            ),

        "lat":
            accumulator["lat"],

        "lon":
            accumulator["lon"],

        "lev":
            accumulator["lev"],

        "qd_sum":
            accumulator["qd_sum"],

        "delp_sum":
            accumulator["delp_sum"],

        "pmid_sum":
            accumulator["pmid_sum"],

        "dust_mass_layer_sum":
            accumulator[
                "dust_mass_layer_sum"
            ],

        "dust_column_mass_sum":
            accumulator[
                "dust_column_mass_sum"
            ],

        "ps_sum":
            accumulator["ps_sum"],

        "max_closure_error":
            np.array(
                accumulator[
                    "max_closure_error"
                ],
                dtype=np.float64
            ),

        "max_ps_error":
            np.array(
                accumulator[
                    "max_ps_error"
                ],
                dtype=np.float64
            ),
    }

    # Añadir los cinco acumuladores por régimen
    # como variables independientes dentro del NPZ.

    for name in REGIMES:

        payload[
            f"regime__{name}"
        ] = (
            accumulator[
                "regime_sums"
            ][name]
        )

    np.savez(
        temp_path,
        **payload
    )

    temp_path.replace(
        path
    )

    print()
    print(
        f"  CHECKPOINT GUARDADO: "
        f"{path.name}"
    )

    print(
        f"  Días acumulados:     "
        f"{accumulator['n_days']}"
    )

    print(
        f"  Tiempos acumulados:  "
        f"{accumulator['n_samples']}"
    )

    print()


def load_checkpoint(path):
    """
    Reconstruye el acumulador desde un checkpoint NPZ.
    """

    with np.load(
        path,
        allow_pickle=False
    ) as data:

        accumulator = {
            "n_days":
                int(
                    data[
                        "n_days"
                    ]
                ),

            "n_samples":
                int(
                    data[
                        "n_samples"
                    ]
                ),

            "last_date":
                str(
                    data[
                        "last_date"
                    ].item()
                ),

            "lat":
                data[
                    "lat"
                ].copy(),

            "lon":
                data[
                    "lon"
                ].copy(),

            "lev":
                data[
                    "lev"
                ].copy(),

            "qd_sum":
                data[
                    "qd_sum"
                ].copy(),

            "delp_sum":
                data[
                    "delp_sum"
                ].copy(),

            "pmid_sum":
                data[
                    "pmid_sum"
                ].copy(),

            "dust_mass_layer_sum":
                data[
                    "dust_mass_layer_sum"
                ].copy(),

            "dust_column_mass_sum":
                data[
                    "dust_column_mass_sum"
                ].copy(),

            "ps_sum":
                data[
                    "ps_sum"
                ].copy(),

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


def find_latest_checkpoint():
    """
    Busca el checkpoint anual más reciente disponible.

    Devuelve:
        (accumulator, year)

    o:
        (None, None)

    si todavía no existe ninguno.
    """

    candidates = []

    for path in CHECKPOINT_DIR.glob(
        "cumulative_through_*.npz"
    ):

        match = re.search(
            r"cumulative_through_(\d{4})\.npz$",
            path.name
        )

        if match:

            year = int(
                match.group(1)
            )

            candidates.append(
                (
                    year,
                    path
                )
            )

    if not candidates:

        return (
            None,
            None
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    latest_year, latest_path = (
        candidates[-1]
    )

    accumulator = load_checkpoint(
        latest_path
    )

    # --------------------------------------------------------
    # Verificar coherencia interna del checkpoint.
    # --------------------------------------------------------

    expected_days = (
        expected_days_through_year(
            latest_year
        )
    )

    if (
        accumulator["n_days"]
        != expected_days
    ):

        raise RuntimeError(
            "El último checkpoint tiene "
            "un número de días inconsistente."
        )

    if (
        accumulator["n_samples"]
        != expected_days * 8
    ):

        raise RuntimeError(
            "El último checkpoint tiene "
            "un número de tiempos inconsistente."
        )

    expected_last_date = (
        datetime(
            latest_year,
            12,
            31
        ).strftime(
            "%Y-%m-%d"
        )
    )

    if (
        accumulator["last_date"]
        != expected_last_date
    ):

        raise RuntimeError(
            "La última fecha del checkpoint "
            "es inconsistente."
        )

    print()
    print(
        f"Checkpoint encontrado: "
        f"{latest_path.name}"
    )

    print(
        f"Se continuará después de "
        f"{accumulator['last_date']}."
    )

    print()

    return (
        accumulator,
        latest_year
    )

# ============================================================
# BUCLE PRINCIPAL DE DESCARGA Y PROCESAMIENTO
# ============================================================

import time


MAX_DOWNLOAD_ATTEMPTS = 5
RETRY_WAIT_SECONDS = 30


def download_with_retries(
    date,
    url
):
    """
    Intenta descargar un día varias veces antes
    de detener completamente el procesamiento.
    """

    last_error = None

    for attempt in range(
        1,
        MAX_DOWNLOAD_ATTEMPTS + 1
    ):

        try:

            return download_daily_file(
                date,
                url
            )

        except Exception as exc:

            last_error = exc

            print()
            print(
                f"  ERROR EN DESCARGA "
                f"(intento {attempt}/"
                f"{MAX_DOWNLOAD_ATTEMPTS})"
            )

            print(
                f"  {exc}"
            )

            if attempt < MAX_DOWNLOAD_ATTEMPTS:

                print(
                    f"  Esperando "
                    f"{RETRY_WAIT_SECONDS} s "
                    f"antes de reintentar..."
                )

                time.sleep(
                    RETRY_WAIT_SECONDS
                )

    raise RuntimeError(
        f"No fue posible descargar "
        f"{date:%Y-%m-%d} después de "
        f"{MAX_DOWNLOAD_ATTEMPTS} intentos."
    ) from last_error


def run_processing():
    """
    Ejecuta el procesamiento completo 2001-2020.

    Si existen checkpoints anuales, continúa desde
    el primer año aún no procesado.
    """

    # ========================================================
    # 1. BUSCAR CHECKPOINT PREVIO
    # ========================================================

    accumulator, checkpoint_year = (
        find_latest_checkpoint()
    )

    if checkpoint_year is None:

        first_year = 2001

        print(
            "No existen checkpoints previos."
        )

        print(
            "El procesamiento comenzará en 2001."
        )

    else:

        first_year = (
            checkpoint_year + 1
        )

        if first_year > 2020:

            print(
                "Los años 2001-2020 ya fueron "
                "procesados completamente."
            )

            return accumulator

    print()
    print(
        f"Primer año a procesar: "
        f"{first_year}"
    )

    print()

    # ========================================================
    # 2. RECORRER AÑOS
    # ========================================================

    for year in range(
        first_year,
        2021
    ):

        year_records = [
            (date, url)

            for date, url in DAILY_URLS

            if date.year == year
        ]

        expected_days_this_year = (
            datetime(
                year + 1,
                1,
                1
            )
            - datetime(
                year,
                1,
                1
            )
        ).days

        if (
            len(year_records)
            != expected_days_this_year
        ):

            raise RuntimeError(
                f"El año {year} debería tener "
                f"{expected_days_this_year} archivos, "
                f"pero la lista contiene "
                f"{len(year_records)}."
            )

        print()
        print("=" * 72)

        print(
            f"PROCESANDO AÑO {year}"
        )

        print(
            f"Días: {len(year_records)}"
        )

        print("=" * 72)

        year_start_time = (
            time.time()
        )

        # ====================================================
        # 3. RECORRER DÍAS DEL AÑO
        # ====================================================

        for (
            day_number,
            (
                date,
                url
            )
        ) in enumerate(
            year_records,
            start=1
        ):

            print()
            print(
                f"[{year} | "
                f"{day_number:03d}/"
                f"{len(year_records):03d}] "
                f"{date:%Y-%m-%d}"
            )

            day_start_time = (
                time.time()
            )

            # ------------------------------------------------
            # Descargar
            # ------------------------------------------------

            local_path = (
                download_with_retries(
                    date,
                    url
                )
            )

            file_size_mb = (
                local_path.stat().st_size
                / 1024**2
            )

            print(
                f"  Tamaño: "
                f"{file_size_mb:.2f} MB"
            )

            # ------------------------------------------------
            # Procesar
            # ------------------------------------------------

            print(
                "  Procesando NetCDF..."
            )

            daily_result = (
                process_daily_file(
                    local_path
                )
            )

            # ------------------------------------------------
            # Crear acumulador durante el primer día de
            # la primera ejecución.
            # ------------------------------------------------

            if accumulator is None:

                accumulator = (
                    create_accumulator(
                        daily_result
                    )
                )

            # ------------------------------------------------
            # Incorporar al acumulador.
            # ------------------------------------------------

            add_daily_result(
                accumulator,
                daily_result,
                date
            )

            # ------------------------------------------------
            # IMPORTANTE:
            #
            # Solo borramos el NetCDF después de que haya
            # sido procesado e incorporado correctamente.
            # ------------------------------------------------

            remove_daily_file(
                local_path
            )

            day_elapsed = (
                time.time()
                - day_start_time
            )

            print(
                f"  OK | "
                f"{day_elapsed:.1f} s | "
                f"muestras acumuladas: "
                f"{accumulator['n_samples']}"
            )

        # ====================================================
        # 4. FINAL DEL AÑO
        # ====================================================

        year_elapsed = (
            time.time()
            - year_start_time
        )

        print()
        print(
            f"Año {year} completado en "
            f"{year_elapsed / 3600.0:.2f} h"
        )

        print(
            "Error máximo de cierre de masa: "
            f"{accumulator['max_closure_error']:.3e} "
            "kg/m²"
        )

        print(
            "Máxima discrepancia PS-DELP: "
            f"{accumulator['max_ps_error']:.3f} Pa"
        )

        # ----------------------------------------------------
        # Checkpoint acumulativo hasta el 31 de diciembre.
        # ----------------------------------------------------

        save_checkpoint(
            accumulator,
            year
        )

    # ========================================================
    # 5. COMPROBACIÓN FINAL
    # ========================================================

    if (
        accumulator["n_days"]
        != 7305
    ):

        raise RuntimeError(
            "El procesamiento terminó pero "
            f"n_days = {accumulator['n_days']} "
            "en lugar de 7305."
        )

    if (
        accumulator["n_samples"]
        != 58440
    ):

        raise RuntimeError(
            "El procesamiento terminó pero "
            f"n_samples = "
            f"{accumulator['n_samples']} "
            "en lugar de 58440."
        )

    print()
    print("=" * 72)
    print("PROCESAMIENTO 2001-2020 COMPLETADO")
    print("=" * 72)

    print(
        f"Días procesados:    "
        f"{accumulator['n_days']}"
    )

    print(
        f"Tiempos procesados: "
        f"{accumulator['n_samples']}"
    )

    print()

    return accumulator

# ============================================================
# CONSTRUIR Y GUARDAR CLIMATOLOGÍA FINAL
# ============================================================

def write_final_netcdf(
    accumulator
):
    """
    Convierte las sumas acumuladas en medias climatológicas
    2001-2020 y escribe el NetCDF final.
    """

    # ========================================================
    # 1. VALIDAR CONTADORES
    # ========================================================

    n_days = accumulator[
        "n_days"
    ]

    n_samples = accumulator[
        "n_samples"
    ]

    if n_days != 7305:

        raise RuntimeError(
            f"Se esperaban 7305 días, "
            f"pero hay {n_days}."
        )

    if n_samples != 58440:

        raise RuntimeError(
            f"Se esperaban 58440 tiempos, "
            f"pero hay {n_samples}."
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

    # IMPORTANTE:
    #
    # Esto es:
    #
    #   mean(q_d * DELP / g)
    #
    # y NO:
    #
    #   mean(q_d) * mean(DELP) / g
    #
    # Por tanto conserva correctamente la covariación
    # temporal entre polvo y espesor de presión.

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
    # 3. FRACCIÓN DE MASA EN CADA RÉGIMEN
    # ========================================================

    regime_fractions = {}

    for name in REGIMES:

        regime_fractions[
            name
        ] = np.divide(
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

    # ========================================================
    # 4. CONTROLES FINALES DE CONSERVACIÓN DE MASA
    # ========================================================

    # --------------------------------------------------------
    # A. Suma de las 72 capas
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # B. Suma de los cinco regímenes
    # --------------------------------------------------------

    column_from_regimes = np.zeros_like(
        dust_column_mass_mean,
        dtype=np.float64
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
    print("VALIDACIÓN DE LA CLIMATOLOGÍA FINAL")
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
    # 5. CREAR DATASET XARRAY
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

        # ----------------------------------------------------
        # Campos verticales
        # ----------------------------------------------------

        "dust_mixing_ratio": (
            (
                "lev",
                "lat",
                "lon"
            ),
            qd_mean
        ),

        "pressure_thickness": (
            (
                "lev",
                "lat",
                "lon"
            ),
            delp_mean
        ),

        "pressure_midpoint": (
            (
                "lev",
                "lat",
                "lon"
            ),
            pmid_mean
        ),

        "dust_mass_layer": (
            (
                "lev",
                "lat",
                "lon"
            ),
            dust_mass_layer_mean
        ),

        # ----------------------------------------------------
        # Campos integrados verticalmente
        # ----------------------------------------------------

        "dust_column_mass": (
            (
                "lat",
                "lon"
            ),
            dust_column_mass_mean
        ),

        "surface_pressure": (
            (
                "lat",
                "lon"
            ),
            ps_mean
        ),
    }

    # ========================================================
    # 6. AÑADIR REGÍMENES
    # ========================================================

    for (
        name,
        (
            p_upper,
            p_lower
        )
    ) in REGIMES.items():

        data_vars[name] = (
            (
                "lat",
                "lon"
            ),
            regime_means[name]
        )

        fraction_name = (
            name.replace(
                "mass_",
                "fraction_"
            )
        )

        data_vars[
            fraction_name
        ] = (
            (
                "lat",
                "lon"
            ),
            regime_fractions[name]
        )

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords=coords
    )

    # ========================================================
    # 7. ATRIBUTOS DE COORDENADAS
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
    # 8. ATRIBUTOS DE VARIABLES
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
    # 9. ATRIBUTOS GLOBALES
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
            "2001-01-01 to 2020-12-31",

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
            "of the 58440 native 3-hourly instantaneous "
            "samples. Dust mass was calculated at each "
            "instant before temporal averaging.",

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
    # 10. COMPRESIÓN NETCDF
    # ========================================================

    encoding = {}

    for variable in ds_out.data_vars:

        encoding[
            variable
        ] = {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
        }

    # ========================================================
    # 11. ESCRITURA SEGURA
    # ========================================================

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

    print(
        "NETCDF FINAL CREADO"
    )

    print("=" * 72)

    print(
        OUTPUT_FILE
    )

    print()

    size_mb = (
        OUTPUT_FILE.stat().st_size
        / 1024**2
    )

    print(
        f"Tamaño final: "
        f"{size_mb:.2f} MB"
    )

    print()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    accumulator = (
        run_processing()
    )

    write_final_netcdf(
        accumulator
    )
