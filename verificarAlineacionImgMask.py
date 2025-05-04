import os
import numpy as np
import matplotlib.pyplot as plt
import random

# 🔹 Directorios
imagenes_dir = r"E:\Memoria\CNN\fusionadas_redimensionadas"  # Carpeta con imágenes de 6 canales en formato .npy
mascaras_dir = r"E:\Memoria\CNN\masks_redimensionadas"  # Carpeta con máscaras redimensionadas en formato .npy

# Obtener nombres de archivos
imagenes_files = {os.path.splitext(f)[0]: f for f in os.listdir(imagenes_dir) if f.endswith(".npy")}
mascaras_files = {os.path.splitext(f)[0].replace("_ground_truth", ""): f for f in os.listdir(mascaras_dir) if f.endswith(".npy")}

# **Verificar coincidencias**
imagenes_set = set(imagenes_files.keys())
mascaras_set = set(mascaras_files.keys())

if imagenes_set != mascaras_set:
    print(f"⚠️ ERROR: No coinciden todas las imágenes y máscaras.")
    print(f"Imágenes sin máscara: {imagenes_set - mascaras_set}")
    print(f"Máscaras sin imagen: {mascaras_set - imagenes_set}")
else:
    print("✅ Los nombres de imágenes y máscaras coinciden correctamente.")

# 📌 Seleccionar 10 imágenes aleatorias
muestras = random.sample(list(imagenes_files.keys()), min(10, len(imagenes_files)))  # Evitar error si hay menos de 10

# **Visualización interactiva**
for nombre in muestras:
    imagen_path = os.path.join(imagenes_dir, imagenes_files[nombre])
    mascara_path = os.path.join(mascaras_dir, mascaras_files[nombre])

    # Cargar imagen y máscara
    imagen = np.load(imagen_path)  # Shape esperado: (512, 512, 6)
    mascara = np.load(mascara_path)  # Shape esperado: (512, 512)

    # **Verificar dimensiones**
    if imagen.shape[:2] != mascara.shape:
        print(f"⚠️ Dimensiones diferentes en {nombre}: Imagen {imagen.shape[:2]}, Máscara {mascara.shape}")
    else:
        print(f"✅ Dimensiones correctas en {nombre}")

    # **Normalizar imagen para visualización**
    imagen = imagen.astype(np.float32)  # Asegurar tipo float
    imagen = np.clip(imagen, 0, 255)  # Evitar valores negativos o fuera de rango
    imagen /= imagen.max()  # Normalizar a [0,1] para que matplotlib la muestre bien

    # **Mostrar imágenes una por una**
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # 📌 Mostrar imagen en RGB (bandas 3,2,1)
    axes[0].imshow(imagen[:, :, [3, 2, 1]])  # RGB
    axes[0].set_title(f"Imagen {nombre} (RGB)")
    axes[0].axis("off")

    # 📌 Mostrar máscara en escala de grises
    axes[1].imshow(mascara, cmap="gray")
    axes[1].set_title("Máscara")
    axes[1].axis("off")

    # 📌 Superponer máscara sobre la imagen
    axes[2].imshow(imagen[:, :, [3, 2, 1]])  # Imagen de fondo en RGB
    axes[2].imshow(mascara, cmap="jet", alpha=0.5)  # Superposición
    axes[2].set_title("Superposición")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()  # 🔹 Aquí el script **se pausa** hasta que cierres la imagen
