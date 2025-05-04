import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# 🔹 Directorios
mascaras_originales_dir = r"E:\Memoria\CNN\pruebasMartes\OG_Fixed\TIF_fixed\Masks"
output_dir = r"E:\Memoria\CNN\masks_redimensionadas"

# Crear carpeta de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obtener lista de máscaras originales
mascaras_files = sorted([f for f in os.listdir(mascaras_originales_dir) if f.endswith(".tif")])

for mask_name in mascaras_files:
    mask_path = os.path.join(mascaras_originales_dir, mask_name)
    output_path = os.path.join(output_dir, mask_name.replace(".tif", ".npy"))

    # Cargar máscara
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # Leer la única banda (grayscale)
        mask_resampled = src.read(
            1, out_shape=(512, 512), resampling=Resampling.nearest  # Nearest neighbor para preservar etiquetas
        )

    # Asegurar valores enteros y tipo correcto
    mask_resampled = mask_resampled.astype(np.uint8)

    # Guardar en formato NumPy
    np.save(output_path, mask_resampled)
    print(f"✅ Máscara redimensionada guardada: {output_path}")

print("🚀 Todas las máscaras han sido redimensionadas correctamente.")
