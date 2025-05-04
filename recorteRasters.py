import os
import rasterio
from rasterio.mask import mask
import subprocess

# Directorios
morfometricos_dir = r"E:\Memoria\GIS\Predictores"  # Carpeta con los rasters
muestras_dir = r"E:\Memoria\CNN\pruebasMartes\OG_Fixed\TIF_fixed\RM"  # Carpeta con imágenes de muestra (TIFF)
output_dir = r"E:\Memoria\CNN\PreProcesamiento\rasters_recortados2"  # Carpeta de salida para rasters recortados

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# ✅ Lista reducida a 2 rasters:
rasters_a_recortar = [
    "Slope_grados.sdat",
    "Aspect.sdat",
    "Convexity.sdat",
    "Landforms.sdat",
    "Terrain Ruggedness Index (TRI).sdat",
    "Topographic Position Index.sdat",
    "Vector Terrain Ruggedness (VRM).sdat",
    "Radar2020.tif"
]

# Obtener todas las imágenes de muestra
muestras_tiff = [f for f in os.listdir(muestras_dir) if f.endswith(".tif")]

for img_name in muestras_tiff:
    img_path = os.path.join(muestras_dir, img_name)

    with rasterio.open(img_path) as img_src:
        bounds = img_src.bounds  # Obtener límites geográficos de la imagen de muestra
        geom = [{
            "type": "Polygon",
            "coordinates": [[
                [bounds.left, bounds.bottom],
                [bounds.left, bounds.top],
                [bounds.right, bounds.top],
                [bounds.right, bounds.bottom],
                [bounds.left, bounds.bottom]
            ]]
        }]
    
    print(f"📌 Procesando imagen {img_name}...")

    for raster_name in rasters_a_recortar:
        raster_path = os.path.join(morfometricos_dir, raster_name)

        # Verificar si el archivo existe
        if not os.path.exists(raster_path):
            print(f"⚠ El raster {raster_name} no fue encontrado.")
            continue

        # Si es un archivo .sdat, convertirlo a GeoTIFF temporalmente
        if raster_name.endswith(".sdat"):
            temp_tif = raster_path.replace(".sdat", ".tif")
            if not os.path.exists(temp_tif):  # Convertir solo si no existe el GeoTIFF
                print(f"🔄 Convirtiendo {raster_name} a GeoTIFF...")
                subprocess.run(["gdal_translate", "-of", "GTiff", raster_path, temp_tif], check=True)
            raster_path = temp_tif  # Usamos el archivo convertido

        # Recortar el raster
        with rasterio.open(raster_path) as src:
            try:
                out_image, out_transform = mask(src, geom, crop=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # 🔹 **Nombre de salida = RasterOriginal_Imagen.tif**
                raster_base_name = raster_name.replace(".sdat", "").replace(".tif", "")  # Sin extensión
                img_base_name = img_name.replace(".tif", "")  # Sin extensión
                output_raster_path = os.path.join(output_dir, f"{raster_base_name}_{img_base_name}.tif")

                with rasterio.open(output_raster_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                print(f"✅ Raster recortado guardado: {output_raster_path}")

            except Exception as e:
                print(f"❌ Error al recortar {raster_name} para {img_name}: {e}")

print("\n✅ ¡Todos los recortes han finalizado!")