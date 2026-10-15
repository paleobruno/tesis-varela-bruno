#!/bin/bash

LOG=$(cat ~/MERRA2_dust/current_log.txt)
PID=$(cat ~/MERRA2_dust/current_run.pid)

while true; do
    clear

    echo "===== MERRA-2 3D ====="
    echo

    if ps -p "$PID" > /dev/null; then
        echo "Estado: CORRIENDO (PID $PID)"
    else
        echo "Estado: DETENIDO"
    fi

    echo
    echo "Último checkpoint:"
    ls -1 /mnt/z/MERRA2_dust/checkpoints/cumulative_through_*.npz 2>/dev/null \
        | sort | tail -1 | xargs -r basename

    echo
    echo "Archivos temporales actuales:"
    ls ~/MERRA2_dust/tmp/*.nc4 2>/dev/null | wc -l

    echo
    echo "Últimos eventos:"
    grep -E "Descargando|Archivo existente válido|CHECKPOINT GUARDADO|Procesando" "$LOG" \
        | tail -10

    sleep 10
done
