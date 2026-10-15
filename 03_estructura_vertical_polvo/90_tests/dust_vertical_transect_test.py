import ee
import os
import csv


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ID = "bruno-paleodust-tesis"

LAT = -45.0

# Patagonia -> Atlántico Sur
LONGITUDES = [
    -70.0, -65.0, -60.0, -55.0, -50.0,
    -45.0, -40.0, -35.0, -30.0
]

DATE_START = "2001-01-01"
DATE_END = "2001-01-02"

G0 = 9.80665
P_TOP_HPA = 0.01

OUTPUT_DIR = os.path.expanduser(
    "~/MERRA2_dust/processed/vertical_regimes"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "dust_vertical_transect_45S_2001-01-01_0000UTC.csv"
)

# Regímenes obtenidos a partir del análisis vertical ERA5
REGIMES = {
    "above_200":     (0.0, 200.0),
    "upper_200_425": (200.0, 425.0),
    "middle_425_725": (425.0, 725.0),
    "lower_725_975": (725.0, 975.0),
    "below_975":     (975.0, 2000.0),
}


# ============================================================
# FUNCIONES
# ============================================================

def pressure_overlap(p_upper, p_lower, r_upper, r_lower):
    """
    Solapamiento en presión [hPa] entre una capa MERRA-2
    y uno de nuestros regímenes verticales.
    """

    overlap_upper = max(p_upper, r_upper)
    overlap_lower = min(p_lower, r_lower)

    return max(0.0, overlap_lower - overlap_upper)


def analyze_profile(values):
    """
    A partir de las bandas DU001-DU005 y DELP de un punto,
    reconstruye el perfil de presión y calcula la masa de polvo
    contenida en cada régimen.
    """

    mass_regime = {
        name: 0.0
        for name in REGIMES
    }

    total_mass = 0.0
    p_edge_hpa = P_TOP_HPA

    for k in range(1, 73):

        # Mixing ratio total de polvo
        q_dust = sum(
            values[f"DU00{i}_{k}"]
            for i in range(1, 6)
        )

        delp_pa = values[f"DELP_{k}"]

        # Bordes de presión de la capa
        p_upper = p_edge_hpa
        p_lower = p_upper + delp_pa / 100.0

        # Masa de polvo completa de la capa
        mass_layer = q_dust * delp_pa / G0
        total_mass += mass_layer

        # Repartición por solapamiento
        for name, (r_upper, r_lower) in REGIMES.items():

            overlap_hpa = pressure_overlap(
                p_upper,
                p_lower,
                r_upper,
                r_lower
            )

            overlap_pa = overlap_hpa * 100.0

            mass_regime[name] += (
                q_dust * overlap_pa / G0
            )

        p_edge_hpa = p_lower

    return total_mass, p_edge_hpa, mass_regime


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Inicializar Earth Engine
    # --------------------------------------------------------

    ee.Initialize(project=PROJECT_ID)

    # --------------------------------------------------------
    # 2. Seleccionar imagen
    # --------------------------------------------------------

    collection = (
        ee.ImageCollection("NASA/GSFC/MERRA/aer_nv/2")
        .filterDate(DATE_START, DATE_END)
    )

    img = ee.Image(collection.first())

    date = (
        ee.Date(img.get("system:time_start"))
        .format("YYYY-MM-dd HH:mm")
        .getInfo()
    )

    # --------------------------------------------------------
    # 3. Bandas necesarias
    # --------------------------------------------------------

    bands = []

    for k in range(1, 73):

        for dust_bin in range(1, 6):
            bands.append(f"DU00{dust_bin}_{k}")

        bands.append(f"DELP_{k}")

    img_selected = img.select(bands)

    # --------------------------------------------------------
    # 4. Crear puntos de la transecta
    # --------------------------------------------------------

    features = []

    for lon in LONGITUDES:

        feature = ee.Feature(
            ee.Geometry.Point([lon, LAT]),
            {
                "longitude": lon,
                "latitude": LAT
            }
        )

        features.append(feature)

    points = ee.FeatureCollection(features)

    # --------------------------------------------------------
    # 5. Extraer todos los perfiles en una sola consulta
    # --------------------------------------------------------

    sampled = img_selected.sampleRegions(
        collection=points,
        scale=50000,
        geometries=False
    )

    samples = sampled.getInfo()["features"]

    # Ordenar por longitud
    samples = sorted(
        samples,
        key=lambda x: x["properties"]["longitude"]
    )

    # --------------------------------------------------------
    # 6. Analizar perfiles
    # --------------------------------------------------------

    results = []

    for feature in samples:

        values = feature["properties"]

        lon = values["longitude"]
        lat = values["latitude"]

        total_mass, surface_pressure, mass_regime = (
            analyze_profile(values)
        )

        row = {
            "longitude": lon,
            "latitude": lat,
            "surface_pressure_hPa": surface_pressure,
            "total_dust_g_m2": total_mass * 1000.0
        }

        for name, mass in mass_regime.items():

            row[f"{name}_g_m2"] = mass * 1000.0

            if total_mass > 0:
                fraction = 100.0 * mass / total_mass
            else:
                fraction = 0.0

            row[f"{name}_percent"] = fraction

        results.append(row)

    # --------------------------------------------------------
    # 7. Mostrar tabla resumida
    # --------------------------------------------------------

    print()
    print("=" * 112)
    print("TRANSECTA VERTICAL DE POLVO MERRA-2: PATAGONIA -> ATLÁNTICO SUR")
    print("=" * 112)

    print(f"Latitud: {abs(LAT):.1f}°S")
    print(f"Fecha:   {date} UTC")

    print()
    print(
        f"{'Lon':>7}"
        f"{'Ps [hPa]':>11}"
        f"{'Dust [g/m²]':>14}"
        f"{'>200':>11}"
        f"{'200-425':>12}"
        f"{'425-725':>12}"
        f"{'725-975':>12}"
        f"{'>975':>11}"
    )

    print("-" * 112)

    for row in results:

        print(
            f"{row['longitude']:7.1f}"
            f"{row['surface_pressure_hPa']:11.1f}"
            f"{row['total_dust_g_m2']:14.6f}"
            f"{row['above_200_percent']:11.2f}"
            f"{row['upper_200_425_percent']:12.2f}"
            f"{row['middle_425_725_percent']:12.2f}"
            f"{row['lower_725_975_percent']:12.2f}"
            f"{row['below_975_percent']:11.2f}"
        )

    print("-" * 112)

    # --------------------------------------------------------
    # 8. Guardar CSV
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    fieldnames = list(results[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("Archivo guardado en:")
    print(OUTPUT_FILE)
    print()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
