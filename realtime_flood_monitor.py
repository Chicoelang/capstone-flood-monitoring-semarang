import os
import json
import ee
import geemap
from datetime import datetime, timedelta

# 1. Inisialisasi GEE (GitHub Actions / Lokal)
service_account = os.environ.get('EARTHENGINE_SERVICE_ACCOUNT')
private_key = os.environ.get('EARTHENGINE_PRIVATE_KEY')

if service_account and private_key:
    credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
    ee.Initialize(credentials)
    print("Terhubung ke GEE via Service Account di GitHub Actions!")
else:
    PROJECT_ID = 'project-capstonee'
    ee.Initialize(project=PROJECT_ID)
    print("Terhubung ke GEE secara lokal!")

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

# 6. Masking Air Permanen (Waduk & Laut)
jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(roi)
permanent_water = jrc.select('occurrence').gt(70).unmask(0)

# Target Latih Sentinel-1
s1_train = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(roi)
            .filterDate('2023-01-01', '2023-03-31')
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .select('VV')
            .median()
            .clip(roi))

flood_label = (s1_train.lt(-13.05)
               .And(permanent_water.Not())
               .rename('flood_label')
               .toInt())

feature_bands = ['elevation', 'slope', 'twi', 'precipitation', 'ndvi']
composite = dem.addBands(slope).addBands(twi).addBands(chirps).addBands(ndvi).addBands(flood_label)

# 7. Latih Model Random Forest
print("Sedang melatih model AI...")
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

# Probabilitas Risiko AI
prob_raster = composite.select(feature_bands).classify(classifier).updateMask(permanent_water.Not())

# 8. Ambil Citra Radar Sentinel-1 Terkini (Real-Time)
today = datetime.now()
start_date = (today - timedelta(days=14)).strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')

s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(roi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .select('VV'))

if s1_col.size().getInfo() > 0:
    s1_latest = s1_col.mosaic().clip(roi)
else:
    s1_latest = (ee.ImageCollection('COPERNICUS/S1_GRD')
                 .filterBounds(roi)
                 .filter(ee.Filter.eq('instrumentMode', 'IW'))
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                 .select('VV')
                 .limit(5, 'system:time_start', False)
                 .median()
                 .clip(roi))

# Deteksi Genangan Terkini (Titik Merah Sesaat)
current_flooding = (s1_latest.lt(-13.05)
                    .And(permanent_water.Not())
                    .selfMask()
                    .rename('live_flood'))

# 9. Tampilkan Peta Lengkap dengan Opsi Centang (Layer Control)
Map = geemap.Map()
Map.centerObject(semarang, 12)

# Layer 1: Kerentanan AI (Gradasi Lengkap Biru - Merah)
vis_ai = {
    'min': 0.0,
    'max': 1.0,
    'palette': ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'],
    'opacity': 0.75
}
Map.addLayer(prob_raster, vis_ai, 'Kerentanan Banjir AI (Prediksi)')

# Layer 2: Radar Sentinel-1 Terbaru (Default dimatikan/False agar tidak menutupi warna AI)
Map.addLayer(s1_latest, {'min': -25, 'max': 0, 'palette': ['black', 'white']}, 'Sentinel-1 Terbaru (dB)', False)

# Layer 3: Deteksi Genangan Air Terkini (Titik Merah)
Map.addLayer(current_flooding, {'palette': ['red']}, 'Deteksi Genangan Air Terkini', True)

# Layer 4: Garis Batas Wilayah
Map.addLayer(semarang.style(color='black', fillColor='00000000', width=2), {}, 'Batas Administrasi Semarang')

# Pasang Kotak Opsi Centang (Turn On / Turn Off) di Pojok Kanan Atas
Map.add_layer_control()

# Pasang Legenda Warna Risiko
Map.add_colorbar(vis_ai, label='Probabilitas Kerentanan (0 = Rendah/Biru, 1 = Tinggi/Merah)', orientation='horizontal')

# Ekspor ke file HTML
Map.to_html('peta_realtime_semarang.html')
Map.to_html('index.html')
print("Selesai! Peta dengan layer control berhasil dibuat.")