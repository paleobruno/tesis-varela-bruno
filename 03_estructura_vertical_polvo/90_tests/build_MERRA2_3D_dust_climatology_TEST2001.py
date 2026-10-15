import ee
import numpy as np
import xarray as xr

import io
import os
import time
import requests
from datetime import date


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

PROJECT_ID = "bruno-paleodust-tesis"

DATASET_ID = "NASA/GSFC/MERRA/aer_nv/2"

START_YEAR = 2001
END_YEAR = 2001

G0 = 9.80665          # m s^-2
P_TOP_HPA = 0.01      # borde superior aproximado del modelo


# ============================================================
# DOMINIO
# ============================================================

# Mismo dominio utilizado en nuestros otros productos:
#
# lon: -80 ... -20
# lat: -60 ... -35
#
# Resolución nativa MERRA-2:
# 0.625° longitud x 0.5° latitud

LON_MIN = -80.0
LON_MAX = -20.0

LAT_MIN = -60.0
LAT_MAX = -35.0

DLON = 0.625
DLAT = 0.5

NX = 97
NY = 51

# Grilla exacta de salida.
#
# El origen corresponde al borde NW del primer píxel.
#
# Centros:
# lon = -80, -79.375, ..., -20
# lat = -35, -35.5, ..., -60

CRS = "EPSG:4326"

CRS_TRANSFORM = [
    DLON, 0.0, LON_MIN - DLON / 2.0,
    0.0, -DLAT, LAT_MAX + DLAT / 2.0
]


# ============================================================
# DIRECTORIOS
# ============================================================

BASE_DIR = os.path.expanduser(
    "~/MERRA2_dust/processed"
)

CACHE_DIR = os.path.join(
    BASE_DIR,
    "MERRA2_3D_dust_annual_cache"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "MERRA2_3D_dust_climatology_TEST2001.nc"
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


# ============================================================
# REGÍMENES VERTICALES
# ============================================================

# Límites expresados como presión [hPa].
#
# p < 200       : sobre nuestro dominio principal
# 200-425       : régimen superior
# 425-725       : régimen medio
# 725-975       : régimen inferior
# p > 975       : capa cercana a superficie

REGIMES = {
    "lt200": (0.0, 200.0),
    "200_425": (200.0, 425.0),
    "425_725": (425.0, 725.0),
    "725_975": (725.0, 975.0),
    "gt975": (975.0, 2000.0),
}


# ============================================================
# NOMBRES DE BANDAS DERIVADAS
# ============================================================

QD_BANDS = [
    f"qd_{k:02d}"
    for k in range(1, 73)
]

MASS_BANDS = [
    f"mass_{k:02d}"
    for k in range(1, 73)
]

DELP_BANDS = [
    f"delp_{k:02d}"
    for k in range(1, 73)
]

PMID_BANDS = [
    f"pmid_{k:02d}"
    for k in range(1, 73)
]

REGIME_BANDS = [
    f"regime_{name}"
    for name in REGIMES
]

EXTRA_BANDS = [
    "column_mass",
    "surface_pressure"
]

ALL_OUTPUT_BANDS = (
    QD_BANDS
    + MASS_BANDS
    + DELP_BANDS
    + PMID_BANDS
    + REGIME_BANDS
    + EXTRA_BANDS
)


# ============================================================
# FUNCIÓN EARTH ENGINE:
# DERIVAR VARIABLES FÍSICAS EN CADA INSTANTE
# ============================================================

def derive_dust_fields(img):
    """
    Para cada imagen instantánea MERRA-2 calcula:

      q_d(k) = DU001 + ... + DU005

      m_d(k) = q_d(k) * DELP(k) / g

    También reconstruye presión media de cada capa y reparte
    la masa vertical entre nuestros cinco regímenes.

    IMPORTANTE:
    estas operaciones se hacen ANTES del promedio temporal.
    """

    img = ee.Image(img)

    output_images = []

    # Borde superior de presión [Pa]
    p_edge = ee.Image.constant(
        P_TOP_HPA * 100.0
    )

    # Acumuladores de masa por régimen
    regime_mass = {
        name: ee.Image.constant(0.0)
        for name in REGIMES
    }

    column_mass = ee.Image.constant(0.0)

    # --------------------------------------------------------
    # Recorrer las 72 capas
    # --------------------------------------------------------

    for k in range(1, 73):

        # ----------------------------------------------------
        # Mixing ratio total de polvo
        # ----------------------------------------------------

        q_dust = (
            img.select(f"DU001_{k}")
            .add(img.select(f"DU002_{k}"))
            .add(img.select(f"DU003_{k}"))
            .add(img.select(f"DU004_{k}"))
            .add(img.select(f"DU005_{k}"))
        )

        delp = img.select(
            f"DELP_{k}"
        )

        # ----------------------------------------------------
        # Bordes de presión de la capa [Pa]
        # ----------------------------------------------------

        p_upper = p_edge

        p_lower = (
            p_upper
            .add(delp)
        )

        p_mid_hpa = (
            p_upper
            .add(p_lower)
            .multiply(0.5)
            .divide(100.0)
        )

        # ----------------------------------------------------
        # Masa de polvo por unidad de superficie
        #
        # kg/m2
        # ----------------------------------------------------

        mass_layer = (
            q_dust
            .multiply(delp)
            .divide(G0)
        )

        column_mass = (
            column_mass
            .add(mass_layer)
        )

        # ----------------------------------------------------
        # Guardar variables de esta capa
        # ----------------------------------------------------

        output_images.append(
            q_dust.rename(
                f"qd_{k:02d}"
            )
        )

        output_images.append(
            mass_layer.rename(
                f"mass_{k:02d}"
            )
        )

        output_images.append(
            delp.rename(
                f"delp_{k:02d}"
            )
        )

        output_images.append(
            p_mid_hpa.rename(
                f"pmid_{k:02d}"
            )
        )

        # ----------------------------------------------------
        # Solapamiento de esta capa con cada régimen
        # ----------------------------------------------------

        for name, (
            regime_upper_hpa,
            regime_lower_hpa
        ) in REGIMES.items():

            regime_upper_pa = (
                regime_upper_hpa * 100.0
            )

            regime_lower_pa = (
                regime_lower_hpa * 100.0
            )

            # overlap =
            #
            # min(p_lower, regime_lower)
            # -
            # max(p_upper, regime_upper)

            overlap_pa = (
                p_lower
                .min(
                    ee.Image.constant(
                        regime_lower_pa
                    )
                )
                .subtract(
                    p_upper.max(
                        ee.Image.constant(
                            regime_upper_pa
                        )
                    )
                )
                .max(0.0)
            )

            mass_overlap = (
                q_dust
                .multiply(overlap_pa)
                .divide(G0)
            )

            regime_mass[name] = (
                regime_mass[name]
                .add(mass_overlap)
            )

        # Borde superior para la siguiente capa
        p_edge = p_lower

    # --------------------------------------------------------
    # Añadir masas integradas de cada régimen
    # --------------------------------------------------------

    for name in REGIMES:

        output_images.append(
            regime_mass[name].rename(
                f"regime_{name}"
            )
        )

    # --------------------------------------------------------
    # Masa total de la columna
    # --------------------------------------------------------

    output_images.append(
        column_mass.rename(
            "column_mass"
        )
    )

    # --------------------------------------------------------
    # Presión superficial reconstruida
    # --------------------------------------------------------

    surface_pressure = (
        p_edge
        .divide(100.0)
        .rename("surface_pressure")
    )

    output_images.append(
        surface_pressure
    )

    # --------------------------------------------------------
    # Combinar todas las bandas
    # --------------------------------------------------------

    result = ee.Image.cat(
        *output_images
    )

    return (
        result
        .toFloat()
        .copyProperties(
            img,
            ["system:time_start"]
        )
    )


# ============================================================
# DESCARGA DE UNA IMAGEN COMO NUMPY
# ============================================================

def download_image_as_npy(
    image,
    band_names,
    output_path,
    max_retries=3
):
    """
    Descarga una imagen pequeña de Earth Engine directamente
    como array NumPy estructurado.

    Se utiliza nuestra grilla fija de 97 x 51 píxeles.
    """

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            print(
                f"    Generando URL "
                f"(intento {attempt}/{max_retries})..."
            )

            url = image.getDownloadURL(
                {
                    "bands": band_names,
                    "crs": CRS,
                    "crs_transform": CRS_TRANSFORM,
                    "dimensions": [NX, NY],
                    "format": "NPY",
                }
            )

            print(
                "    Descargando producto anual..."
            )

            response = requests.get(
                url,
                timeout=1800
            )

            if not response.ok:
                print()
                print("Respuesta completa de Earth Engine:")
                print(response.text)
                print()

            response.raise_for_status()

            data = np.load(
                io.BytesIO(
                    response.content
                )
            )

            # Comprobaciones básicas
            if data.shape != (NY, NX):

                raise RuntimeError(
                    "Dimensiones inesperadas: "
                    f"{data.shape}; "
                    f"esperado {(NY, NX)}"
                )

            missing = [
                name
                for name in band_names
                if name not in data.dtype.names
            ]

            if missing:

                raise RuntimeError(
                    "Faltan bandas en la descarga: "
                    + ", ".join(missing[:10])
                )

            np.save(
                output_path,
                data
            )

            return data

        except Exception as exc:

            print()
            print(
                f"    ERROR en intento {attempt}:"
            )
            print(
                f"    {exc}"
            )

            if attempt == max_retries:
                raise

            print(
                "    Esperando 15 segundos "
                "antes de reintentar..."
            )

            time.sleep(15)


# ============================================================
# CARGAR CACHE ANUAL
# ============================================================

def load_cached_year(path):
    """
    Intenta cargar y validar un archivo anual ya procesado.
    """

    try:

        data = np.load(path)

        if data.shape != (NY, NX):
            return None

        if data.dtype.names is None:
            return None

        missing = [
            name
            for name in ALL_OUTPUT_BANDS
            if name not in data.dtype.names
        ]

        if missing:
            return None

        return data

    except Exception:
        return None


# ============================================================
# CONSTRUIR DATASET NETCDF
# ============================================================

def build_netcdf(
    climatology,
    total_samples
):
    """
    Convierte los campos climatológicos en un xarray.Dataset
    y los guarda como NetCDF.
    """

    # --------------------------------------------------------
    # Coordenadas
    # --------------------------------------------------------

    lon = (
        LON_MIN
        + np.arange(NX) * DLON
    )

    # La descarga viene norte -> sur.
    # En el NetCDF usaremos latitud creciente.
    lat = (
        LAT_MIN
        + np.arange(NY) * DLAT
    )

    layer = np.arange(
        1,
        73,
        dtype=np.int16
    )

    # --------------------------------------------------------
    # Función para invertir eje latitud
    # --------------------------------------------------------

    def orient(field):

        return (
            np.asarray(
                field,
                dtype=np.float32
            )[::-1, :]
        )

    # --------------------------------------------------------
    # Construir campos 3D
    # --------------------------------------------------------

    qd_3d = np.stack(
        [
            orient(
                climatology[
                    f"qd_{k:02d}"
                ]
            )
            for k in range(1, 73)
        ],
        axis=0
    )

    mass_3d = np.stack(
        [
            orient(
                climatology[
                    f"mass_{k:02d}"
                ]
            )
            for k in range(1, 73)
        ],
        axis=0
    )

    delp_3d = np.stack(
        [
            orient(
                climatology[
                    f"delp_{k:02d}"
                ]
            )
            for k in range(1, 73)
        ],
        axis=0
    )

    pmid_3d = np.stack(
        [
            orient(
                climatology[
                    f"pmid_{k:02d}"
                ]
            )
            for k in range(1, 73)
        ],
        axis=0
    )

    column_mass = orient(
        climatology[
            "column_mass"
        ]
    )

    surface_pressure = orient(
        climatology[
            "surface_pressure"
        ]
    )

    # --------------------------------------------------------
    # Dataset base
    # --------------------------------------------------------

    ds = xr.Dataset(
        coords={
            "layer": layer,
            "lat": lat.astype(
                np.float32
            ),
            "lon": lon.astype(
                np.float32
            ),
        }
    )

    # --------------------------------------------------------
    # Variables 3D
    # --------------------------------------------------------

    ds[
        "dust_mixing_ratio_clim"
    ] = (
        ("layer", "lat", "lon"),
        qd_3d
    )

    ds[
        "dust_mass_layer_clim"
    ] = (
        ("layer", "lat", "lon"),
        mass_3d
    )

    ds[
        "delp_clim"
    ] = (
        ("layer", "lat", "lon"),
        delp_3d
    )

    ds[
        "pressure_mid_clim"
    ] = (
        ("layer", "lat", "lon"),
        pmid_3d
    )

    # --------------------------------------------------------
    # Variables 2D principales
    # --------------------------------------------------------

    ds[
        "dust_column_mass_clim"
    ] = (
        ("lat", "lon"),
        column_mass
    )

    ds[
        "surface_pressure_clim"
    ] = (
        ("lat", "lon"),
        surface_pressure
    )

    # --------------------------------------------------------
    # Masa y fracción por régimen
    # --------------------------------------------------------

    for name in REGIMES:

        regime_mass = orient(
            climatology[
                f"regime_{name}"
            ]
        )

        ds[
            f"dust_mass_{name}"
        ] = (
            ("lat", "lon"),
            regime_mass
        )

        fraction = np.divide(
            regime_mass,
            column_mass,
            out=np.full_like(
                regime_mass,
                np.nan,
                dtype=np.float32
            ),
            where=column_mass > 0.0
        )

        ds[
            f"dust_fraction_{name}"
        ] = (
            ("lat", "lon"),
            fraction.astype(
                np.float32
            )
        )

    # --------------------------------------------------------
    # Atributos de coordenadas
    # --------------------------------------------------------

    ds["lat"].attrs = {
        "long_name": "latitude",
        "units": "degrees_north"
    }

    ds["lon"].attrs = {
        "long_name": "longitude",
        "units": "degrees_east"
    }

    ds["layer"].attrs = {
        "long_name": (
            "MERRA-2 native model layer"
        ),
        "description": (
            "1 = model top; "
            "72 = lowest model layer"
        )
    }

    # --------------------------------------------------------
    # Atributos variables 3D
    # --------------------------------------------------------

    ds[
        "dust_mixing_ratio_clim"
    ].attrs = {
        "long_name": (
            "Climatological total dust "
            "mass mixing ratio"
        ),
        "units": "kg kg-1",
        "definition": (
            "DU001 + DU002 + DU003 + "
            "DU004 + DU005"
        )
    }

    ds[
        "dust_mass_layer_clim"
    ].attrs = {
        "long_name": (
            "Climatological atmospheric "
            "dust mass per unit area "
            "in each model layer"
        ),
        "units": "kg m-2",
        "formula": "q_dust * DELP / g"
    }

    ds[
        "delp_clim"
    ].attrs = {
        "long_name": (
            "Climatological pressure "
            "thickness of model layer"
        ),
        "units": "Pa"
    }

    ds[
        "pressure_mid_clim"
    ].attrs = {
        "long_name": (
            "Climatological midpoint "
            "pressure of model layer"
        ),
        "units": "hPa"
    }

    ds[
        "dust_column_mass_clim"
    ].attrs = {
        "long_name": (
            "Climatological total "
            "atmospheric dust column mass"
        ),
        "units": "kg m-2"
    }

    ds[
        "surface_pressure_clim"
    ].attrs = {
        "long_name": (
            "Climatological reconstructed "
            "surface pressure"
        ),
        "units": "hPa"
    }

    # --------------------------------------------------------
    # Atributos regímenes
    # --------------------------------------------------------

    regime_labels = {
        "lt200": "p < 200 hPa",
        "200_425": "200 <= p < 425 hPa",
        "425_725": "425 <= p < 725 hPa",
        "725_975": "725 <= p < 975 hPa",
        "gt975": "p >= 975 hPa",
    }

    for name in REGIMES:

        ds[
            f"dust_mass_{name}"
        ].attrs = {
            "long_name": (
                "Climatological dust mass "
                f"in pressure regime {regime_labels[name]}"
            ),
            "units": "kg m-2",
            "method": (
                "Pressure-overlap integration "
                "performed at every 3-hourly "
                "time step before averaging"
            )
        }

        ds[
            f"dust_fraction_{name}"
        ].attrs = {
            "long_name": (
                "Fraction of total dust "
                "column mass in pressure "
                f"regime {regime_labels[name]}"
            ),
            "units": "1",
            "valid_min": 0.0,
            "valid_max": 1.0
        }

    # --------------------------------------------------------
    # Atributos globales
    # --------------------------------------------------------

    ds.attrs = {
        "title": (
            "MERRA-2 3D Dust Climatology "
            "2001-2020 - Southern South America "
            "and South Atlantic"
        ),

        "source_dataset": (
            "NASA/GSFC/MERRA/aer_nv/2 "
            "(MERRA-2 M2I3NVAER)"
        ),

        "source_doi": (
            "10.5067/LTVB4GPCOTK2"
        ),

        "climatology_period": (
            "2001-01-01 to 2020-12-31"
        ),

        "temporal_sampling": (
            "3-hourly instantaneous fields: "
            "00, 03, 06, 09, 12, 15, 18, 21 UTC"
        ),

        "n_time_samples": int(
            total_samples
        ),

        "gravity_m_s2": G0,

        "model_top_pressure_hPa": (
            P_TOP_HPA
        ),

        "horizontal_resolution": (
            "0.5 degree latitude x "
            "0.625 degree longitude"
        ),

        "domain": (
            "80W-20W, 60S-35S"
        ),

        "vertical_layers": 72,

        "processing_method": (
            "Total dust mixing ratio was computed "
            "as DU001+DU002+DU003+DU004+DU005. "
            "Layer dust mass q_dust*DELP/g and "
            "pressure-regime overlap masses were "
            "computed at each individual 3-hourly "
            "time step before temporal averaging. "
            "Annual climatologies were subsequently "
            "combined using the actual number of "
            "3-hourly samples in each year."
        ),
    }

    # --------------------------------------------------------
    # Diagnóstico de cierre
    # --------------------------------------------------------

    regime_sum = np.zeros_like(
        column_mass,
        dtype=np.float64
    )

    for name in REGIMES:

        regime_sum += ds[
            f"dust_mass_{name}"
        ].values

    closure = (
        regime_sum
        - column_mass
    )

    max_abs_closure = np.nanmax(
        np.abs(closure)
    )

    print()
    print(
        "Máximo error absoluto de cierre "
        "de regímenes:"
    )

    print(
        f"{max_abs_closure:.6e} kg/m²"
    )

    # --------------------------------------------------------
    # Guardar NetCDF
    # --------------------------------------------------------

    print()
    print(
        "Guardando NetCDF final..."
    )

    ds.to_netcdf(
        OUTPUT_FILE
    )

    return ds


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # Inicializar Earth Engine
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "MERRA-2 3D DUST CLIMATOLOGY 2001-2020"
    )
    print("=" * 72)
    print()

    print(
        "Inicializando Earth Engine..."
    )

    ee.Initialize(
        project=PROJECT_ID
    )

    # --------------------------------------------------------
    # Acumuladores climatológicos
    # --------------------------------------------------------

    weighted_sum = {
        name: np.zeros(
            (NY, NX),
            dtype=np.float64
        )
        for name in ALL_OUTPUT_BANDS
    }

    total_samples = 0

    # --------------------------------------------------------
    # Procesar año por año
    # --------------------------------------------------------

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        start_date = (
            f"{year}-01-01"
        )

        end_date = (
            f"{year + 1}-01-01"
        )

        collection = (
            ee.ImageCollection(
                DATASET_ID
            )
            .filterDate(
                start_date,
                end_date
            )
        )

        n_images = (
            collection
            .size()
            .getInfo()
        )

        expected_images = (
            (
                date(
                    year + 1,
                    1,
                    1
                )
                -
                date(
                    year,
                    1,
                    1
                )
            ).days
            * 8
        )

        print()
        print("-" * 72)
        print(
            f"AÑO {year}"
        )
        print("-" * 72)

        print(
            f"Imágenes encontradas: "
            f"{n_images}"
        )

        print(
            f"Imágenes esperadas:    "
            f"{expected_images}"
        )

        if n_images != expected_images:

            print(
                "ADVERTENCIA: el número de "
                "imágenes no coincide con "
                "el esperado."
            )

        if n_images == 0:

            raise RuntimeError(
                f"No hay datos para {year}."
            )

        # ----------------------------------------------------
        # Cache local
        # ----------------------------------------------------

        cache_file = os.path.join(
            CACHE_DIR,
            f"MERRA2_3D_dust_{year}_annualmean.npy"
        )

        annual_data = None

        if os.path.exists(
            cache_file
        ):

            print(
                "Encontrado cache anual."
            )

            annual_data = (
                load_cached_year(
                    cache_file
                )
            )

            if annual_data is not None:

                print(
                    "Cache válido. "
                    "No se descargará nuevamente."
                )

            else:

                print(
                    "Cache inválido. "
                    "Se volverá a calcular."
                )

        # ----------------------------------------------------
        # Calcular promedio anual en Earth Engine
        # ----------------------------------------------------

        if annual_data is None:

            print(
                "Construyendo variables "
                "derivadas en Earth Engine..."
            )

            derived_collection = (
                collection.map(
                    derive_dust_fields
                )
            )

            print(
                "Calculando promedio anual..."
            )

            annual_mean = (
                derived_collection
                .mean()
                .select(
                    ALL_OUTPUT_BANDS
                )
                .toFloat()
            )

            annual_data = (
                download_image_as_npy(
                    annual_mean,
                    ALL_OUTPUT_BANDS,
                    cache_file
                )
            )

            print(
                "Producto anual guardado:"
            )

            print(
                f"    {cache_file}"
            )

        # ----------------------------------------------------
        # Acumular ponderado por número de imágenes
        # ----------------------------------------------------

        print(
            "Incorporando año al "
            "promedio climatológico..."
        )

        for band_name in ALL_OUTPUT_BANDS:

            weighted_sum[
                band_name
            ] += (
                np.asarray(
                    annual_data[
                        band_name
                    ],
                    dtype=np.float64
                )
                * n_images
            )

        total_samples += n_images

        print(
            f"Muestras acumuladas: "
            f"{total_samples}"
        )

    # ========================================================
    # CLIMATOLOGÍA FINAL
    # ========================================================

    print()
    print("=" * 72)
    print(
        "CONSTRUYENDO CLIMATOLOGÍA FINAL"
    )
    print("=" * 72)

    climatology = {}

    for band_name in ALL_OUTPUT_BANDS:

        climatology[
            band_name
        ] = (
            weighted_sum[
                band_name
            ]
            / total_samples
        )

    # ========================================================
    # NETCDF
    # ========================================================

    ds = build_netcdf(
        climatology,
        total_samples
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print()
    print("=" * 72)
    print(
        "PROCESAMIENTO COMPLETADO"
    )
    print("=" * 72)

    print()
    print(
        f"Número total de muestras "
        f"3-horarias: {total_samples}"
    )

    print()
    print(
        "Dimensiones finales:"
    )

    print(
        f"    layer = {ds.sizes['layer']}"
    )

    print(
        f"    lat   = {ds.sizes['lat']}"
    )

    print(
        f"    lon   = {ds.sizes['lon']}"
    )

    print()
    print(
        "NetCDF guardado en:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Caches anuales guardados en:"
    )

    print(
        CACHE_DIR
    )

    print()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
