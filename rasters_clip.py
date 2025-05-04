import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd

# Definir rutas
ruta_capa_mascara = r"E:\Memoria\GIS\AREA DE ESTUDIO\Nueva_area\Cuencas_cordilleranas.shp"
ruta_rasters = r"E:\Memoria\GIS\Predictores"
ruta_salida = r"C:\Users\yuuky\Desktop\Nueva carpeta"

# Leer la capa máscara (shapefile)
mascara = gpd.read_file(ruta_capa_mascara)
geometrias = mascara.geometry.values  # Array de geometrías

# Procesar cada raster .sdat en la carpeta de entrada
for archivo in os.listdir(ruta_rasters):
    if archivo.endswith(".sdat"):
        ruta_raster = os.path.join(ruta_rasters, archivo)
        with rasterio.open(ruta_raster) as src:
            # Recortar el raster usando las geometrías de la capa máscara
            recorte, transformacion = mask(src, geometrias, crop=True)
            metadata = src.meta.copy()
            metadata.update({
                "driver": "GTiff",  # Guardamos en formato TIFF
                "height": recorte.shape[1],
                "width": recorte.shape[2],
                "transform": transformacion
            })
            
            # Generar el nombre del archivo de salida conservando el nombre original
            nombre_salida = os.path.splitext(archivo)[0] + ".tif"
            ruta_final = os.path.join(ruta_salida, nombre_salida)
            
            # Escribir el raster recortado
            with rasterio.open(ruta_final, "w", **metadata) as dst:
                dst.write(recorte)
            
            print(f"Procesado: {archivo} -> {nombre_salida}")
