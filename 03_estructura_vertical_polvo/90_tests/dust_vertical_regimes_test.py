import ee

# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ID = "bruno-paleodust-tesis"

LON = -68.0
LAT = -45.0

DATE_START = "2001-01-01"
DATE_END = "2001-01-02"

G0 = 9.80665       # m s^-2
P_TOP_HPA = 0.01   # presión aproximada del tope del modelo

# Regímenes verticales definidos a partir del clustering ERA5
REGIMES = {
    "Sobre 200 hPa":    (0.0, 200.0),
    "Superior 200-425": (200.0, 425.0),
    "Medio 425-725":    (425.0, 725.0),
    "Inferior 725-975": (725.0, 975.0),
    "Bajo 975 hPa":     (975.0, 2000.0),
}


# ============================================================
# FUNCIONES
# ============================================================

def pressure_overlap(p_upper, p_lower, r_upper, r_lower):
    """
    Calcula el solapamiento, en hPa, entre una capa atmosférica
    [p_upper, p_lower] y un régimen [r_upper, r_lower].
    """

    overlap_upper = max(p_upper, r_upper)
    overlap_lower = min(p_lower, r_lower)

    return max(0.0, overlap_lower - overlap_upper)


def main():

    # --------------------------------------------------------
    # 1. Inicializar Earth Engine
    # --------------------------------------------------------

    ee.Initialize(project=PROJECT_ID)

    # --------------------------------------------------------
    # 2. Seleccionar una imagen MERRA-2
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

    point = ee.Geometry.Point([LON, LAT])

    # --------------------------------------------------------
    # 3. Construir lista de bandas
    # --------------------------------------------------------

    bands = []

    for k in range(1, 73):

        for dust_bin in range(1, 6):
            bands.append(f"DU00{dust_bin}_{k}")

        bands.append(f"DELP_{k}")

    # --------------------------------------------------------
    # 4. Extraer perfil vertical en el punto
    # --------------------------------------------------------

    values = (
        img
        .select(bands)
        .reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=50000,
            maxPixels=1000000
        )
        .getInfo()
    )

    # --------------------------------------------------------
    # 5. Inicializar acumuladores
    # --------------------------------------------------------

    mass_regime = {
        name: 0.0
        for name in REGIMES
    }

    total_mass = 0.0

    p_edge_hpa = P_TOP_HPA

    # --------------------------------------------------------
    # 6. Recorrer las 72 capas MERRA-2
    # --------------------------------------------------------

    for k in range(1, 73):

        # Mixing ratio total de polvo:
        # suma de los cinco tamaños de partícula
        q_dust = sum(
            values[f"DU00{i}_{k}"]
            for i in range(1, 6)
        )

        # Espesor de presión de la capa
        delp_pa = values[f"DELP_{k}"]

        # Bordes de presión
        p_upper = p_edge_hpa
        p_lower = p_upper + delp_pa / 100.0

        # Masa total de polvo de la capa
        mass_layer = q_dust * delp_pa / G0

        total_mass += mass_layer

        # ----------------------------------------------------
        # Repartir la capa entre los regímenes que atraviesa
        # ----------------------------------------------------

        for name, (r_upper, r_lower) in REGIMES.items():

            overlap_hpa = pressure_overlap(
                p_upper,
                p_lower,
                r_upper,
                r_lower
            )

            overlap_pa = overlap_hpa * 100.0

            mass_overlap = (
                q_dust
                * overlap_pa
                / G0
            )

            mass_regime[name] += mass_overlap

        p_edge_hpa = p_lower

    # --------------------------------------------------------
    # 7. Resultados
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DISTRIBUCIÓN VERTICAL DEL POLVO MERRA-2")
    print("=" * 70)

    print(f"Punto: {abs(LAT):.1f}°S, {abs(LON):.1f}°W")
    print(f"Fecha: {date} UTC")

    print()
    print(
        f"Presión superficial reconstruida: "
        f"{p_edge_hpa:.2f} hPa"
    )

    print()
    print(
        f"Carga total de polvo: "
        f"{total_mass:.6e} kg/m²"
    )

    print(
        f"                     "
        f"{total_mass * 1000:.6f} g/m²"
    )

    print()
    print("-" * 70)

    print(
        f"{'Régimen':<25}"
        f"{'Masa [g/m²]':>20}"
        f"{'Fracción [%]':>20}"
    )

    print("-" * 70)

    for name, mass in mass_regime.items():

        mass_g = mass * 1000.0

        if total_mass > 0:
            fraction = 100.0 * mass / total_mass
        else:
            fraction = 0.0

        print(
            f"{name:<25}"
            f"{mass_g:20.6e}"
            f"{fraction:20.3f}"
        )

    print("-" * 70)

    sum_regimes = sum(mass_regime.values())

    print()
    print(
        f"Suma de los regímenes: "
        f"{sum_regimes:.6e} kg/m²"
    )

    print(
        f"Diferencia respecto al total: "
        f"{sum_regimes - total_mass:.3e} kg/m²"
    )

    print()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()
