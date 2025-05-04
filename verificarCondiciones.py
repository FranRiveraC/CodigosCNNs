import os
import numpy as np
from collections import Counter

# 🔹 Directorios base
dataset_dir = r"E:\Memoria\CNN\Dataset_Split_npu"
splits = ["train", "val", "test"]

# 🔹 Variables para estadística global
image_shapes = set()
mask_shapes = set()
class_pixel_counts = {split: Counter() for split in splits}
image_min_max = {split: [] for split in splits}

# 🔹 Revisar cada conjunto
for split in splits:
    print(f"\n🔎 Revisando {split.upper()}...\n")

    img_dir = os.path.join(dataset_dir, split, "images")
    mask_dir = os.path.join(dataset_dir, split, "masks")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".npy")])

    # 📌 Verificar cantidad de archivos
    print(f"📂 {len(img_files)} imágenes en {split}, {len(mask_files)} máscaras en {split}")

    # 📌 Verificar nombres coincidentes
    img_base_names = {f.replace(".npy", "") for f in img_files}
    mask_base_names = {f.replace("_ground_truth.npy", "") for f in mask_files}

    if img_base_names != mask_base_names:
        print(f"⚠️ ERROR: No coinciden nombres de imágenes y máscaras en {split}.")
    else:
        print("✅ Imágenes y máscaras correctamente emparejadas.")

    # 📌 Revisar cada imagen y máscara
    for img_name in img_base_names:
        img_path = os.path.join(img_dir, img_name + ".npy")
        mask_path = os.path.join(mask_dir, img_name + "_ground_truth.npy")

        # 🔹 Cargar imagen y máscara
        img = np.load(img_path)  # Imagen con 6 canales
        mask = np.load(mask_path)  # Máscara con clases 0-3

        # 🔹 Verificar dimensiones
        image_shapes.add(img.shape)
        mask_shapes.add(mask.shape)

        if img.shape != (512, 512, 6):
            print(f"⚠️ Dimensión inesperada en {img_name}: {img.shape} (debe ser 512x512x6)")
        
        if mask.shape != (512, 512):
            print(f"⚠️ Dimensión inesperada en {img_name} (máscara): {mask.shape} (debe ser 512x512)")

        # 🔹 Revisar valores en la imagen
        min_vals = img.min(axis=(0, 1))
        max_vals = img.max(axis=(0, 1))
        image_min_max[split].append((min_vals, max_vals))

        # 🔹 Revisar valores en la máscara (deben ser 0,1,2,3)
        unique_vals, counts = np.unique(mask, return_counts=True)
        class_pixel_counts[split].update(dict(zip(unique_vals, counts)))

# 📌 Revisar si todas las imágenes tienen el tamaño esperado
print("\n✅ Dimensiones de imágenes encontradas:", image_shapes)
print("✅ Dimensiones de máscaras encontradas:", mask_shapes)

# 📌 Estadísticas de valores en imágenes
for split in splits:
    min_vals, max_vals = zip(*image_min_max[split])
    min_vals = np.min(min_vals, axis=0)
    max_vals = np.max(max_vals, axis=0)
    print(f"\n📊 Rango de valores en imágenes ({split}):")
    for i in range(6):
        print(f"  - Canal {i+1}: Min={min_vals[i]}, Max={max_vals[i]}")

# 📌 Estadísticas de píxeles por clase
for split in splits:
    print(f"\n📊 Distribución de clases en máscaras ({split}):")
    total_pixels = sum(class_pixel_counts[split].values())
    for cls, count in sorted(class_pixel_counts[split].items()):
        percentage = (count / total_pixels) * 100
        print(f"  - Clase {cls}: {count} píxeles ({percentage:.2f}%)")

print("\n✅ Revisión finalizada.")
