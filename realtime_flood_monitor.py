import os
import json
import ee
import geemap
from datetime import datetime, timedelta

# 1. Inisialisasi Project GEE (Otomatis Service Account atau Lokal)
service_account = os.environ.get('EARTHENGINE_SERVICE_ACCOUNT')
private_key = os.environ.get('EARTHENGINE_PRIVATE_KEY')

if service_account and private_key:
    # Mode GitHub Actions (Awan)
    key_dict = json.loads(private_key)
    credentials = ee.ServiceAccountCredentials(service_account, key_data=key_dict)
    ee.Initialize(credentials)
    print("Berhasil terhubung ke GEE menggunakan Service Account di GitHub Actions!")
else:
    # Mode Lokal di Laptop
    PROJECT_ID = 'project-capstonee'
    ee.Initialize(project=PROJECT_ID)
    print("Berhasil terhubung ke GEE menggunakan sesi lokal!")

# 2. Definisikan Wilayah Kajian (Kota Semarang)
admin2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
semarang = admin2.filter(
    ee.Filter.And(
        ee.Filter.eq('ADM0_NAME', 'Indonesia'),
        ee.Filter.eq('ADM2_NAME', 'Kota Semarang')
    )
)
roi = semarang.geometry()

# 3. Filter Rentang Waktu Dinamis (14 Hari Terakhir dari Hari Ini)
today = datetime.now()
start_date = (today - timedelta(days=14)).strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')

print(f"Memindai data satelit Sentinel-1 terbaru: {start_date} s/d {end_date}...")

# 4. Ambil Citra Sentinel-1 Terbaru
s1_col = (
    ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .select('VV')
)

count = s1_col.size().getInfo()
print(f"Ditemukan {count} citra Sentinel-1 terbaru.")

# Ambil data Sentinel-1 (jika 14 hari terakhir kosong karena latency orbit, ambil citra valid terakhir)
if count > 0:
    s1_latest = s1_col.mosaic().clip(roi)
else:
    print("Menggunakan citra satelit valid terdekat...")
    s1_latest = (ee.ImageCollection('COPERNICUS/S1_GRD')
                 .filterBounds(roi)
                 .filter(ee.Filter.eq('instrumentMode', 'IW'))
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                 .select('VV')
                 .limit(5, 'system:time_start', False)
                 .median()
                 .clip(roi))

# 5. Masking Air Permanen (Waduk & Laut)
jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(roi)
permanent_water = jrc.select('occurrence').gt(70).unmask(0)

# Deteksi Genangan Air Terkini (VV < -13.05 dB)
current_flooding = (s1_latest.lt(-13.05)
                    .And(permanent_water.Not())
                    .selfMask()
                    .rename('live_flood'))

# 6. Perbarui Visualisasi Peta
Map = geemap.Map()
Map.centerObject(semarang, 12)

# Layer Citra Radar
Map.addLayer(s1_latest, {'min': -25, 'max': 0, 'palette': ['black', 'white']}, 'Sentinel-1 Terbaru (dB)', False)

# Layer Deteksi Genangan Real-Time
Map.addLayer(current_flooding, {'palette': ['red']}, 'Deteksi Genangan Air Terkini')
Map.addLayer(semarang.style(color='black', fillColor='00000000', width=2), {}, 'Batas Administrasi Semarang')

output_html = "peta_realtime_semarang.html"
Map.to_html(output_html)
print(f"Peta pemantauan terbaru berhasil diperbarui ke '{output_html}' pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")