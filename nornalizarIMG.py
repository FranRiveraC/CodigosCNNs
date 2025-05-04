import os
import numpy as np

# Directorios
input_dir = r"E:\Memoria\CNN\fusionadas_redimensionadas"  # Carpeta con imágenes fusionadas (sin normalizar)
output_dir = r"E:\Memoria\CNN\fusionadas_normalizadas"  # Carpeta donde guardaremos las imágenes normalizadas

# Crear la carpeta de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Definir los índices de las bandas a normalizar (Slope, TRI, VRM)
bandas_a_normalizar = [3, 4, 5]

# Función de normalización Min-Max
def normalize_band(band):
    min_val = np.min(band)
    max_val = np.max(band)
    return (band - min_val) / (max_val - min_val)  # Normalización a [0,1]

# Procesar cada imagen en la carpeta
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".npy"):
        filepath = os.path.join(input_dir, filename)
        data = np.load(filepath)

        # Aplicar normalización solo a las bandas especificadas
        for i in bandas_a_normalizar:
            data[..., i] = normalize_band(data[..., i])

        # Guardar la imagen normalizada
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, data)
        print(f"✅ Imagen guardada: {output_path}")

print("🚀 Proceso de normalización completado.")
