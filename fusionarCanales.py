import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# 📂 Directorios
imagenes_dir = r"E:\Memoria\CNN\Originales\TIFF\RM"
morfometria_dir = r"E:\Memoria\CNN\PreProcesamiento\rasters_recortados2"
output_dir = r"E:\Memoria\CNN\PreProcesamiento\fusionadas_numpy2"

# 🔹 Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# 📋 Obtener lista de imágenes TIFF
imagenes_tiff = sorted([f for f in os.listdir(imagenes_dir) if f.endswith(".tif")])
morfometricos_tiff = sorted([f for f in os.listdir(morfometria_dir) if f.endswith(".tif")])

# 📜 Diccionario para guardar nombres de canales por imagen
canales_por_imagen = {}

# 🔄 Procesar cada imagen
for img_name in imagenes_tiff:
    img_path = os.path.join(imagenes_dir, img_name)
    img_base_name = os.path.splitext(img_name)[0]  # Ejemplo: "car1", "des4", "rm2"

    with rasterio.open(img_path) as src:
        # Cargar imagen RGB
        imagen_base = src.read([1, 2, 3])  # Bandas RGB
        img_size = (src.height, src.width)
        perfil_imagen = src.profile  # Guardar metadatos
    
    # 🔎 Buscar rasters que correspondan al img_base_name actual
    rasters_asociados = []
    for f in morfometricos_tiff:
        # Extraer el image_id del archivo morfométrico (última parte después de '_')
        morf_base = os.path.splitext(f)[0]
        morf_image_id = morf_base.split('_')[-1]
        if morf_image_id == img_base_name:
            rasters_asociados.append(os.path.join(morfometria_dir, f))
    
    if not rasters_asociados:
        print(f"⚠ No se encontraron rasters para {img_name}. Saltando...")
        continue

    print(f"\n📌 Fusionando {len(rasters_asociados)} rasters con {img_name}...")

    # 📜 Lista para registrar el nombre de cada canal
    nombres_canales = ["Red", "Green", "Blue"]

    # 🔄 Cargar y redimensionar los rasters morfométricos
    rasters_morfometricos = []
    for raster_path in rasters_asociados:
        with rasterio.open(raster_path) as src:
            raster_resampled = src.read(
                1, out_shape=img_size, resampling=Resampling.bilinear
            )

            # 🛠 Corrección de valores NODATA
            nodata_value = src.nodata
            if nodata_value is not None:
                raster_resampled = np.where(raster_resampled == nodata_value, np.nan, raster_resampled)

            # Opcional: reemplazar NaN con la media de valores válidos
            if np.isnan(raster_resampled).any():
                media_valida = np.nanmean(raster_resampled)
                raster_resampled = np.where(np.isnan(raster_resampled), media_valida, raster_resampled)

            rasters_morfometricos.append(raster_resampled)
            nombres_canales.append(os.path.basename(raster_path))  # Guardar nombre del raster

    # 📌 Convertir la imagen base a formato (H, W, C)
    imagen_base = np.moveaxis(imagen_base, 0, -1)  # De (3, H, W) a (H, W, 3)
    
    # 📌 Apilar los rasgos morfométricos
    rasters_morfometricos = np.stack(rasters_morfometricos, axis=-1)  # (H, W, N)
    
    # 📌 Fusionar imagen con rasgos
    imagen_fusionada = np.concatenate([imagen_base, rasters_morfometricos], axis=-1)  # (H, W, C+N)
    
    # 📌 Guardar la imagen fusionada en formato NumPy
    output_path = os.path.join(output_dir, f"{img_base_name}_fusionada.npy")
    np.save(output_path, imagen_fusionada)

    # 📜 Guardar nombres de los canales en un diccionario
    canales_por_imagen[img_base_name] = nombres_canales

    print(f"✅ Imagen fusionada guardada en: {output_path}")

# 📜 Guardar los nombres de los canales en un archivo de texto
canales_output_path = os.path.join(output_dir, "nombres_canales.txt")
with open(canales_output_path, "w") as f:
    for img, canales in canales_por_imagen.items():
        f.write(f"{img}: {', '.join(canales)}\n")

print(f"\n📜 Archivo con nombres de canales guardado en: {canales_output_path}")
print("\n🚀 ¡Proceso de fusión completado!")