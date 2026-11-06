import ee
import geemap

# 1. Inisialisasi Project GEE
PROJECT_ID = 'project-capstonee'
ee.Initialize(project=PROJECT_ID)

# 2. Definisikan Wilayah Kajian (Kota Semarang)
# Menggunakan FAO GAUL Level 2 (Kabupaten/Kota di Indonesia)
admin2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
semarang = admin2.filter(
    ee.Filter.And(
        ee.Filter.eq('ADM0_NAME', 'Indonesia'),
        ee.Filter.eq('ADM2_NAME', 'Kota Semarang')
    )
)
roi = semarang.geometry()

# 3. Filter Citra Sentinel-1 SAR (Periode Musim Hujan / Kejadian Banjir)
# Parameter sesuai paper: mode IW, polarisasi VV
s1_col = (
    ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate('2023-01-01', '2023-03-31')  # Contoh periode puncak musim hujan
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .select('VV')
)

# Ambil median dari citra radar dan potong (clip) sesuai Semarang
s1_median = s1_col.median().clip(roi)

# Sesuai paper: air/genangan memiliki nilai backscatter rendah (VV < -13.05 dB)
water_mask = s1_median.lt(-13.05).selfMask().rename('Genangan_Air')

# 4. Filter Curah Hujan CHIRPS
chirps_col = (
    ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterBounds(roi)
    .filterDate('2023-01-01', '2023-03-31')
    .select('precipitation')
)
chirps_total = chirps_col.sum().clip(roi)

# 5. Visualisasi dengan Peta Interaktif
Map = geemap.Map()
Map.centerObject(semarang, 11)

# Tambahkan layer batas wilayah
Map.addLayer(semarang, {'color': 'black'}, 'Batas Kota Semarang', False)

# Layer Sentinel-1 (Backscatter Radar)
Map.addLayer(
    s1_median,
    {'min': -25, 'max': 0, 'palette': ['black', 'white']},
    'Sentinel-1 VV (dB)'
)

# Layer Deteksi Genangan (Biru)
Map.addLayer(
    water_mask,
    {'palette': ['blue']},
    'Potensi Genangan (VV < -13.05 dB)'
)

# Layer CHIRPS (Curah Hujan)
Map.addLayer(
    chirps_total,
    {'min': 500, 'max': 1500, 'palette': ['blue', 'cyan', 'yellow', 'red']},
    'Akumulasi Hujan CHIRPS (mm)',
    False
)

# Simpan sebagai file HTML untuk dibuka di browser atau tampil langsung jika pakai .ipynb
Map.to_html('peta_semarang.html')
print("Peta interaktif berhasil dibuat dan disimpan ke 'peta_semarang.html'!")