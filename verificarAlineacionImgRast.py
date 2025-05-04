import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Directorios
imagenes_dir = r"E:\Memoria\CNN\pruebasMartes\OG_Fixed\TIF_fixed\RM"
rasters_dir = r"E:\Memoria\CNN\dem_recortados"  # Ajustar según la carpeta donde guardaste los rasters recortados

# Lista de imágenes y rasters disponibles
imagenes = sorted([f for f in os.listdir(imagenes_dir) if f.endswith(".tif")])
rasters = sorted([f for f in os.listdir(rasters_dir) if f.endswith(".tif")])

# Seleccionar un subconjunto de imágenes para visualizar (por ejemplo, 5 aleatorias)
num_muestras = min(5, len(imagenes))  # Evitar errores si hay menos de 5 imágenes
muestras = np.random.choice(imagenes, num_muestras, replace=False)

for img_name in muestras:
    raster_name = f"Terrain Ruggedness Index (TRI)_{img_name}"  # Ajustar el formato si los nombres son diferentes

    img_path = os.path.join(imagenes_dir, img_name)
    raster_path = os.path.join(rasters_dir, raster_name)

    if not os.path.exists(raster_path):
        print(f"⚠ Raster no encontrado para {img_name}, omitiendo...")
        continue

    # Cargar imagen
    with rasterio.open(img_path) as img_src:
        img = img_src.read([1, 2, 3])  # Leer los 3 canales RGB
        img_extent = img_src.bounds  # Obtener los límites geográficos

    # Cargar raster recortado
    with rasterio.open(raster_path) as raster_src:
        raster = raster_src.read(1)  # Leer la primera banda (variable morfométrica)
        raster_extent = raster_src.bounds

    # Crear la figura de comparación
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Imagen RGB original
    ax[0].imshow(np.moveaxis(img, 0, -1))  # Reordenar ejes para visualización RGB
    ax[0].set_title(f"Imagen Original: {img_name}")
    ax[0].axis("off")

    # Raster recortado superpuesto en una escala de color
    ax[1].imshow(raster, cmap="viridis")  # Colormap para visualizar los valores del raster
    ax[1].set_title(f"Raster Recortado: {raster_name}")
    ax[1].axis("off")

    plt.show()
