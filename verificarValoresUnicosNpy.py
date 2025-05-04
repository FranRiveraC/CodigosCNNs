import os
import numpy as np
import rasterio

# 🔹 Rutas de las carpetas
tiff_dir = r"E:\Memoria\CNN\pruebasMartes\OG_Fixed\TIF_fixed\Masks"  # Carpeta con los TIFF originales
npy_dir = r"E:\Memoria\CNN\masks_redimensionadas"  # Carpeta con los NPY redimensionados

# 🔹 Obtener listas ordenadas de archivos
tiff_files = sorted([f for f in os.listdir(tiff_dir) if f.endswith(".tif")])
npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")])

# 🔹 Verificar correspondencia
if len(tiff_files) != len(npy_files):
    print(f"⚠️ Diferente número de archivos: {len(tiff_files)} TIFF vs {len(npy_files)} NPY")

# 🔹 Comparar cada par TIFF - NPY
for tiff_name in tiff_files:
    tiff_path = os.path.join(tiff_dir, tiff_name)
    npy_name = tiff_name.replace(".tif", ".npy")
    npy_path = os.path.join(npy_dir, npy_name)

    if not os.path.exists(npy_path):
        print(f"❌ No se encontró NPY correspondiente para {tiff_name}")
        continue

    # Cargar TIFF
    with rasterio.open(tiff_path) as src:
        tiff_data = src.read(1)  # Leer única banda (grayscale)
        tiff_nodata = src.nodata

    # Cargar NPY
    npy_data = np.load(npy_path)

    # Comparar dimensiones
    if tiff_data.shape != npy_data.shape:
        print(f"⚠️ Dimensiones distintas en {tiff_name}: TIFF {tiff_data.shape}, NPY {npy_data.shape}")

    # Comparar valores únicos
    unique_tiff = np.unique(tiff_data)
    unique_npy = np.unique(npy_data)

    # Mostrar resultados
    print(f"\n📂 Comparando {tiff_name} vs {npy_name}")
    print(f"🔹 Valores únicos TIFF: {unique_tiff}")
    print(f"🔹 Valores únicos NPY: {unique_npy}")

    # Ver diferencias
    diff_tiff_to_npy = set(unique_tiff) - set(unique_npy)
    diff_npy_to_tiff = set(unique_npy) - set(unique_tiff)

    if diff_tiff_to_npy:
        print(f"⚠️ Valores en TIFF no encontrados en NPY: {diff_tiff_to_npy}")
    if diff_npy_to_tiff:
        print(f"⚠️ Valores en NPY no encontrados en TIFF: {diff_npy_to_tiff}")

print("\n✅ Comparación finalizada.")
