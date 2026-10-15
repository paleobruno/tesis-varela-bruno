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
    "dust_vertical_transect_45S_2001-01-01_dailymean.csv"
)

# Regímenes verticales
REGIMES = {
    "above_200":       (0.0, 200.0),
    "upper_200_425":   (200.0, 425.0),
    "middle_425_725":  (425.0, 725.0),
    "lower_725_975":   (725.0, 975.0),
    "below_975":       (975.0, 2000.0),
}


# ============================================================
# FUNCIONES
# ============================================================

def pressure_overlap(p_upper, p_lower, r_upper, r_lower):
    """
    Solapamiento en presión [hPa] entre una capa MERRA-2
    y un régimen vertical.
    """

    overlap_upper = max(p_upper, r_upper)
    overlap_lower = min(p_lower, r_lower)

    return max(0.0, overlap_lower - overlap_upper)


def analyze_profile(values):
    """
    Calcula, para un perfil instantáneo:
      - carga total de polvo
      - presión superficial reconstruida
      - masa de polvo en cada régimen
    """

    mass_regime = {
        name: 0.0
        for name in REGIMES
    }

    total_mass = 0.0
    p_edge_hpa = P_TOP_HPA

    for k in range(1, 73):

        # Mixing ratio total:
        # DU001 + DU002 + ... + DU005
        q_dust = sum(
            values[f"DU00{i}_{k}"]
            for i in range(1, 6)
        )

        delp_pa = values[f"DELP_{k}"]

        # Bordes de la capa
        p_upper = p_edge_hpa
        p_lower = p_upper + delp_pa / 100.0

        # Masa total de polvo en la capa
        mass_layer = q_dust * delp_pa / G0

        total_mass += mass_layer

        # Dividir la masa según solapamiento con regímenes
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
    # 2. Colección de un día completo
    # --------------------------------------------------------

    collection = (
        ee.ImageCollection("NASA/GSFC/MERRA/aer_nv/2")
        .filterDate(DATE_START, DATE_END)
        .sort("system:time_start")
    )

    n_images = collection.size().getInfo()

    print()
    print(f"Número de imágenes encontradas: {n_images}")

    if n_images == 0:
        raise RuntimeError(
            "No se encontraron imágenes para el período seleccionado."
        )

    # --------------------------------------------------------
    # 3. Bandas necesarias
    # --------------------------------------------------------

    bands = []

    for k in range(1, 73):

        for dust_bin in range(1, 6):
            bands.append(f"DU00{dust_bin}_{k}")

        bands.append(f"DELP_{k}")

    # --------------------------------------------------------
    # 4. Crear puntos de la transecta
    # --------------------------------------------------------

    features = []

    for lon in LONGITUDES:

        features.append(
            ee.Feature(
                ee.Geometry.Point([lon, LAT]),
                {
                    "longitude": lon,
                    "latitude": LAT
                }
            )
        )

    points = ee.FeatureCollection(features)

    # --------------------------------------------------------
    # 5. Acumuladores diarios
    # --------------------------------------------------------

    accumulators = {}

    for lon in LONGITUDES:

        accumulators[lon] = {
            "surface_pressure_sum": 0.0,
            "total_mass_sum": 0.0,
            "n": 0,
            "mass_regime_sum": {
                name: 0.0
                for name in REGIMES
            }
        }

    # --------------------------------------------------------
    # 6. Recorrer las 8 imágenes de 3 horas
    # --------------------------------------------------------

    image_list = collection.toList(n_images)

    for index in range(n_images):

        img = ee.Image(image_list.get(index))

        time_string = (
            ee.Date(img.get("system:time_start"))
            .format("YYYY-MM-dd HH:mm")
            .getInfo()
        )

        print(
            f"Procesando imagen "
            f"{index + 1}/{n_images}: {time_string} UTC"
        )

        sampled = (
            img
            .select(bands)
            .sampleRegions(
                collection=points,
                scale=50000,
                geometries=False
            )
        )

        samples = sampled.getInfo()["features"]

        for feature in samples:

            values = feature["properties"]

            lon = float(values["longitude"])

            total_mass, surface_pressure, mass_regime = (
                analyze_profile(values)
            )

            acc = accumulators[lon]

            acc["surface_pressure_sum"] += surface_pressure
            acc["total_mass_sum"] += total_mass
            acc["n"] += 1

            for name in REGIMES:
                acc["mass_regime_sum"][name] += (
                    mass_regime[name]
                )

    # --------------------------------------------------------
    # 7. Construir promedio diario
    # --------------------------------------------------------

    results = []

    for lon in LONGITUDES:

        acc = accumulators[lon]
        n = acc["n"]

        if n == 0:
            continue

        mean_surface_pressure = (
            acc["surface_pressure_sum"] / n
        )

        mean_total_mass = (
            acc["total_mass_sum"] / n
        )

        row = {
            "longitude": lon,
            "latitude": LAT,
            "n_times": n,
            "surface_pressure_hPa": mean_surface_pressure,
            "total_dust_g_m2": mean_total_mass * 1000.0
        }

        for name in REGIMES:

            mean_mass = (
                acc["mass_regime_sum"][name] / n
            )

            row[f"{name}_g_m2"] = (
                mean_mass * 1000.0
            )

            if mean_total_mass > 0:

                fraction = (
                    100.0
                    * mean_mass
                    / mean_total_mass
                )

            else:
                fraction = 0.0

            row[f"{name}_percent"] = fraction

        results.append(row)

    # --------------------------------------------------------
    # 8. Mostrar tabla
    # --------------------------------------------------------

    print()
    print("=" * 112)
    print(
        "TRANSECTA DIARIA DE POLVO MERRA-2: "
        "PATAGONIA -> ATLÁNTICO SUR"
    )
    print("=" * 112)

    print(f"Latitud: {abs(LAT):.1f}°S")
    print(f"Día:     {DATE_START}")
    print(f"Salidas utilizadas: {n_images}")

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
    # 9. Guardar CSV
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
