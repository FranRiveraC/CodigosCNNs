import os
import numpy as np
import matplotlib.pyplot as plt

# Directorio con las imágenes fusionadas y redimensionadas
redimensionadas_dir = r"E:\Memoria\CNN\fusionadas_redimensionadas"

# Obtener archivos numpy redimensionados
redimensionadas_files = sorted([f for f in os.listdir(redimensionadas_dir) if f.endswith(".npy")])

if not redimensionadas_files:
    print("❌ No se encontraron archivos .npy en la carpeta de redimensionadas.")
    exit()

# Seleccionar una imagen de muestra
sample_file = os.path.join(redimensionadas_dir, redimensionadas_files[0])
fused_img = np.load(sample_file)

# Verificar dimensiones
print(f"📌 Imagen: {sample_file}")
print(f"Dimensiones: {fused_img.shape}")  # Debería ser (512, 512, 6)

# Separar los canales correctamente
rgb = fused_img[..., :3] / 255.0  # Normalizar RGB a [0,1]
slope = fused_img[..., 3]
tri = fused_img[..., 4]
vrm = fused_img[..., 5]  # Cambio de TPI a VRM

# Normalizar los canales adicionales para mejorar la visualización
def normalize(array):
    array[array < -10000] = np.nan  # Reemplazar valores inválidos (-99999) con NaN
    min_val, max_val = np.nanmin(array), np.nanmax(array)  # Ignorar NaN en cálculos
    return (array - min_val) / (max_val - min_val) if max_val > min_val else array

slope_normalized = normalize(slope)
tri_normalized = normalize(tri)
vrm_normalized = normalize(vrm)

# Mostrar los canales de la imagen
plt.figure(figsize=(14, 6))

# Mostrar RGB (normalizado)
plt.subplot(1, 4, 1)
plt.imshow(rgb)
plt.title("RGB Original")

# Mostrar Slope
plt.subplot(1, 4, 2)
plt.imshow(slope_normalized, cmap="terrain")
plt.title("Slope (Normalizado)")

# Mostrar TRI
plt.subplot(1, 4, 3)
plt.imshow(tri_normalized, cmap="terrain")
plt.title("TRI (Normalizado)")

# Mostrar VRM
plt.subplot(1, 4, 4)
plt.imshow(vrm_normalized, cmap="terrain")
plt.title("VRM (Normalizado)")

plt.tight_layout()
plt.show()
