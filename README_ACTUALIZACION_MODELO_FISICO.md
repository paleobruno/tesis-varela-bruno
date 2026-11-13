# Actualización: modelo físico de polvo atmosférico

**Tesis:** reconstrucción geoestadística de deposición de polvo, Patagonia y Atlántico Sur.

Este paquete es **aditivo**: copiar las carpetas `03_estructura_vertical_polvo/`, `04_transporte_moderno/` y `09_Incertidumbre_y_sensibilidad/` sobre la copia local del repositorio, **sin borrar el contenido existente**. Incluye una selección curada, no todos los diagnósticos del ZIP `physical_model(1).zip`.

## Criterio editorial

- **Centrales:** seis mapas de carga/fracción vertical, gradientes y superposiciones espaciales, posición del máximo y costa, escalas exponenciales, sensibilidad del ajuste, perfiles del modelo por tramos y un caso austral asimétrico.
- **Complementarias:** perfiles observados específicos, validación del viento ERA5 y ejemplos de ajustes por régimen y muestreo aleatorio para interpolación del residuo.
- **No incluidos:** decenas de gráficos casi repetidos por cada latitud y cada semilla; archivos `Zone.Identifier`; ficheros NetCDF pesados; resultados de kriging en forma de tablas masivas. Se conservan en el ZIP de trabajo original.

## Cómo leer las figuras

Las figuras descriptivas del polvo vertical no validan por sí solas la contribución a deposición. Los mapas de gradientes y superposiciones son **diagnósticos**, no trayectorias lagrangianas. Los parámetros ajustados a perfiles de MERRA-2 corresponden al clima moderno, no se deben presentar directamente como estimaciones para LGM/PIH. Los ajustes australes asimétricos son una representación empírica alternativa, no una solución de la EDO de advección-remoción. La figura de muestreo muestra posiciones usadas, pero **no demuestra** mejora de kriging hasta comparar métricas de validación.

## Rutas

- `03_estructura_vertical_polvo/modelo_fisico/figuras`: cargas $B_r$ y fracciones $f_r$.
- `04_transporte_moderno/modelo_fisico/gradientes`: derivadas zonales por régimen con escala comparable.
- `04_transporte_moderno/modelo_fisico/superposiciones`: mapas de carga y derivada.
- `04_transporte_moderno/modelo_fisico/fuente_costa`: máximo de deposición y posición de costa.
- `04_transporte_moderno/modelo_fisico/ajustes`: longitudes características estimadas.
- `04_transporte_moderno/modelo_fisico/perfiles_observados`: ejemplos de perfiles MERRA-2.
- `04_transporte_moderno/modelo_fisico/ajustes_tramos`: comparación de perfiles con el modelo por tramos.
- `04_transporte_moderno/modelo_fisico/ajustes_australes`: alternativa empírica austral.
- `04_transporte_moderno/modelo_fisico/validacion_viento`: velocidad de ERA5 para el transporte zonal.
- `04_transporte_moderno/modelo_fisico/ajustes_por_regimen`: comparación por régimen inferior y medio.
- `09_Incertidumbre_y_sensibilidad/modelo_fisico`: sensibilidad de parámetros y ejemplo de muestreo para kriging.

El archivo `CATALOGO_FIGURAS.csv` da la ruta exacta de cada imagen y su correspondencia con el ZIP original.

## Subida sin modificar la historia

1. Descarga tu repositorio en ZIP o clónalo y descomprime este paquete encima de su **raíz**.
2. Comprueba los cambios con `git status` y `git diff --stat`.
3. Ejecuta `git add 03_estructura_vertical_polvo 04_transporte_moderno 09_Incertidumbre_y_sensibilidad CATALOGO_FIGURAS.csv README_ACTUALIZACION_MODELO_FISICO.md`.
4. Confirma con `git commit -m "Add curated physical dust model figures and diagnostics"` y publica con `git push origin main`.

**Reproducibilidad:** este ZIP no incluye el código fuente que generó los gráficos porque no aparece en el archivo suministrado. Para alcanzar reproducibilidad integral hay que incorporar los scripts/notebooks de generación y las versiones/fechas de los datos de MERRA-2 y ERA5. Tampoco se ha modificado el repositorio remoto.
