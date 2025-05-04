import os
import numpy as np
import scipy.ndimage

# Directorios
fusionadas_dir = r"E:\Memoria\CNN\fusionadas_sin_redimensionar"  # Carpeta con imágenes fusionadas
output_dir = r"E:\Memoria\CNN\fusionadas_redimensionadas"  # Carpeta de salida

# Crear carpeta de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obtener lista de archivos numpy fusionados
fusionadas_files = sorted([f for f in os.listdir(fusionadas_dir) if f.endswith(".npy")])

# Tamaño deseado
target_size = (512, 512)

for file_name in fusionadas_files:
    file_path = os.path.join(fusionadas_dir, file_name)

    # Cargar la imagen fusionada
    fused_img = np.load(file_path)

    # Obtener dimensiones actuales
    h, w, c = fused_img.shape

    # Redimensionar cada canal usando SciPy
    resized_bands = [
        scipy.ndimage.zoom(fused_img[..., i], (target_size[0] / h, target_size[1] / w), order=1)
        for i in range(c)
    ]

    # Apilar bandas redimensionadas
    resized_fused_img = np.stack(resized_bands, axis=-1)

    # Guardar la imagen redimensionada
    output_path = os.path.join(output_dir, file_name)
    np.save(output_path, resized_fused_img)

    print(f"✅ Redimensionada y guardada: {output_path}")

print("🚀 ¡Todas las imágenes fusionadas han sido redimensionadas correctamente a 512x512!")
