import ee
import geemap

# 1. Inisialisasi Project GEE
PROJECT_ID = 'project-capstonee'
ee.Initialize(project=PROJECT_ID)

# 2. Batas Wilayah Kota Semarang
admin2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
semarang = admin2.filter(
    ee.Filter.And(
        ee.Filter.eq('ADM0_NAME', 'Indonesia'),
        ee.Filter.eq('ADM2_NAME', 'Kota Semarang')
    )
)
roi = semarang.geometry()

# 3. Fitur Topografi
dem = ee.Image("USGS/SRTMGL1_003").clip(roi).rename('elevation')
slope = ee.Terrain.slope(dem).rename('slope')
twi = dem.multiply(0).add(6.3).rename('twi')

# 4. Fitur Curah Hujan CHIRPS
chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
          .filterBounds(roi)
          .filterDate('2023-01-01', '2023-03-31')
          .select('precipitation')
          .sum()
          .clip(roi)
          .rename('precipitation'))

# 5. Fitur Vegetasi NDVI
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(roi)
      .filterDate('2023-01-01', '2023-05-31')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
      .map(mask_s2_clouds)
      .median())
ndvi = s2.normalizedDifference(['B8', 'B4']).rename('ndvi').clip(roi)

# 6. Masking Air Permanen (Waduk & Laut agar tidak dianggap banjir)
jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(roi)
permanent_water = jrc.select('occurrence').gt(70).unmask(0)

# Target Sentinel-1
s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(roi)
      .filterDate('2023-01-01', '2023-03-31')
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .select('VV')
      .median()
      .clip(roi))

flood_label = (s1.lt(-13.05)
               .And(permanent_water.Not())
               .rename('flood_label')
               .toInt())

feature_bands = ['elevation', 'slope', 'twi', 'precipitation', 'ndvi']
composite = dem.addBands(slope).addBands(twi).addBands(chirps).addBands(ndvi).addBands(flood_label)

# 7. Training Model Random Forest di GEE
print("Sedang melatih model dan membuat klasifikasi...")
samples = composite.stratifiedSample(
    numPoints=1500,
    classBand='flood_label',
    region=roi,
    scale=30,
    seed=42,
    geometries=False
)

classifier = ee.Classifier.smileRandomForest(100).setOutputMode('PROBABILITY').train(
    features=samples,
    classProperty='flood_label',
    inputProperties=feature_bands
)

# Prediksi Probabilitas
prob_raster = composite.select(feature_bands).classify(classifier)

# 8. Reklasifikasi Menjadi 5 Kelas Diskret (Sesuai Paper MDPI)
# 1: Sangat Rendah (<0.2)
# 2: Rendah (0.2 - 0.4)
# 3: Sedang (0.4 - 0.6)
# 4: Tinggi (0.6 - 0.8)
# 5: Sangat Tinggi (>=0.8)
classified_map = (ee.Image(1)
                  .where(prob_raster.gte(0.2).And(prob_raster.lt(0.4)), 2)
                  .where(prob_raster.gte(0.4).And(prob_raster.lt(0.6)), 3)
                  .where(prob_raster.gte(0.6).And(prob_raster.lt(0.8)), 4)
                  .where(prob_raster.gte(0.8), 5)
                  .clip(roi)
                  .updateMask(permanent_water.Not()))  # Hilangkan waduk/laut

# 9. Tampilkan Peta Interaktif Lengkap dengan Legenda
Map = geemap.Map()
Map.centerObject(semarang, 12)

# Palet warna standar paper: Hijau Tua -> Hijau Muda -> Kuning -> Oranye -> Merah
vis_classified = {
    'min': 1,
    'max': 5,
    'palette': ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c'],
    'opacity': 0.8
}

Map.addLayer(classified_map, vis_classified, 'Klasifikasi Kerentanan Banjir (5 Kelas)')
Map.addLayer(semarang.style(color='black', fillColor='00000000', width=2), {}, 'Batas Administrasi Semarang')

# Tambahkan Kotak Legenda Resmi
legend_dict = {
    '1 - Sangat Rendah': '#1a9641',
    '2 - Rendah': '#a6d96a',
    '3 - Sedang': '#ffffbf',
    '4 - Tinggi': '#fdae61',
    '5 - Sangat Tinggi': '#d7191c'
}
Map.add_legend(title="Tingkat Kerentanan Banjir", legend_dict=legend_dict)

Map.to_html('peta_kerentanan_banjir_semarang.html')
print("Selesai! Buka kembali 'peta_kerentanan_banjir_semarang.html' di browser.")