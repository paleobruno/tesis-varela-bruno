from pathlib import Path
from datetime import datetime
import re
import netrc
from getpass import getpass

from harmony import Client
from harmony.config import Environment


JOB_ID = "5ace3c7a-e897-4284-9753-8ccc83393901"

OUTPUT_FILE = (
    Path.home()
    / "MERRA2_dust"
    / "links"
    / "harmony_M2I3NVAER_2001_2020.txt"
)


def get_login_credentials():
    try:
        nrc = netrc.netrc()
        login, _, password = nrc.authenticators(
            "urs.earthdata.nasa.gov"
        )
        return login, password

    except Exception:
        login = input("Earthdata Login username: ")
        password = getpass("Earthdata Login password: ")
        return login, password


def main():
    login, password = get_login_credentials()

    client = Client(
        auth=(login, password),
        env=Environment.PROD
    )

    print(f"Consultando trabajo Harmony: {JOB_ID}")

    job = client.status(JOB_ID)

    if job["status"] not in (
        "successful",
        "complete_with_errors"
    ):
        print("Esperando que termine el trabajo...")
        client.wait_for_processing(
            JOB_ID,
            show_progress=True
        )

    print("Obteniendo URLs completas...")

    urls = list(
        client.result_urls(JOB_ID)
    )

    print(
        f"URLs devueltas por Harmony: {len(urls)}"
    )

    records = []

    for url in urls:
        match = re.search(
            r"(\d{8})_subsetted\.nc4(?:\?|$)",
            url
        )

        if match is None:
            continue

        date = datetime.strptime(
            match.group(1),
            "%Y%m%d"
        )

        if (
            datetime(2001, 1, 1)
            <= date
            <= datetime(2020, 12, 31)
        ):
            records.append(
                (date, url)
            )

    records.sort(
        key=lambda item: item[0]
    )

    dates = [
        date
        for date, url in records
    ]

    if len(records) != 7305:
        raise RuntimeError(
            f"Se esperaban 7305 archivos diarios, "
            f"pero se encontraron {len(records)}."
        )

    if len(set(dates)) != 7305:
        raise RuntimeError(
            "Hay fechas duplicadas en los resultados."
        )

    if dates[0] != datetime(2001, 1, 1):
        raise RuntimeError(
            f"Fecha inicial incorrecta: {dates[0]}"
        )

    if dates[-1] != datetime(2020, 12, 31):
        raise RuntimeError(
            f"Fecha final incorrecta: {dates[-1]}"
        )

    OUTPUT_FILE.write_text(
        "\n".join(
            url
            for date, url in records
        )
        + "\n",
        encoding="utf-8"
    )

    print()
    print("LISTA VALIDADA CORRECTAMENTE")
    print(f"Archivos:      {len(records)}")
    print(f"Primera fecha: {dates[0]:%Y-%m-%d}")
    print(f"Última fecha:  {dates[-1]:%Y-%m-%d}")
    print(f"Guardada en:   {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
