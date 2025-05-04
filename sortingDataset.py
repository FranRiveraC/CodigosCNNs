import os
import shutil
import numpy as np

# 🔹 Rutas de los datos originales
imagenes_dir = r"E:\Memoria\CNN\fusionadas_redimensionadas"  
mascaras_dir = r"E:\Memoria\CNN\masks_redimensionadas"

# 🔹 Rutas de destino para cada conjunto
output_base = r"E:\Memoria\CNN\Dataset_Split_npy"
train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")
test_dir = os.path.join(output_base, "test")

# 🔹 Crear carpetas si no existen
for subdir in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(subdir, "images"), exist_ok=True)
    os.makedirs(os.path.join(subdir, "masks"), exist_ok=True)

# 🔹 Obtener lista de archivos
imagenes_files = sorted([f for f in os.listdir(imagenes_dir) if f.endswith(".npy")])
mascaras_files = sorted([f for f in os.listdir(mascaras_dir) if f.endswith(".npy")])

# 🔹 Asegurar correspondencia entre imágenes y máscaras
imagenes_dict = {f.replace(".npy", ""): f for f in imagenes_files}
mascaras_dict = {f.replace("_ground_truth.npy", ""): f for f in mascaras_files}

# 🔹 Filtrar solo archivos que tienen su par correspondiente
paired_files = sorted(set(imagenes_dict.keys()) & set(mascaras_dict.keys()))
np.random.shuffle(paired_files)  # Mezclar aleatoriamente

# 🔹 División del dataset (50% train, 20% val, 30% test)
num_total = len(paired_files)
num_train = int(0.5 * num_total)
num_val = int(0.2 * num_total)

train_set = paired_files[:num_train]
val_set = paired_files[num_train:num_train + num_val]
test_set = paired_files[num_train + num_val:]

# 🔹 Función para mover archivos a la carpeta correspondiente
def move_files(file_list, dataset_type):
    img_dst = os.path.join(output_base, dataset_type, "images")
    mask_dst = os.path.join(output_base, dataset_type, "masks")
    
    for name in file_list:
        img_src = os.path.join(imagenes_dir, imagenes_dict[name])
        mask_src = os.path.join(mascaras_dir, mascaras_dict[name])
        
        shutil.move(img_src, os.path.join(img_dst, imagenes_dict[name]))
        shutil.move(mask_src, os.path.join(mask_dst, mascaras_dict[name]))

# 🔹 Mover archivos
move_files(train_set, "train")
move_files(val_set, "val")
move_files(test_set, "test")

print(f"✅ División completada: {num_train} train, {num_val} val, {num_total - num_train - num_val} test.")
