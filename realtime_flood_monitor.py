import ee
import geemap
from datetime import datetime, timedelta

# 1. Inisialisasi Project GEE
PROJECT_ID = 'project-capstonee'
ee.Initialize(project=PROJECT_ID)

# 2. Definisikan Wilayah Kajian (Semarang)
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

print(f"Memindai data satelit terbaru untuk periode: {start_date} s/d {end_date}...")

# 4. Ambil Citra Sentinel-1 Terbaru
s1_col = (
    ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .select('VV')
)

# Periksa apakah ada citra satelit baru yang tersedia
count = s1_col.size().getInfo()
print(f"Ditemukan {count} citra Sentinel-1 terbaru.")

if count > 0:
    s1_latest = s1_col.mosaic().clip(roi)
    
    # 5. Masking Air Permanen
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(roi)
    permanent_water = jrc.select('occurrence').gt(70).unmask(0)

    # Deteksi Potensi Genangan Air Terkini (VV < -13.05 dB)
    current_flooding = (s1_latest.lt(-13.05)
                        .And(permanent_water.Not())
                        .selfMask()
                        .rename('live_flood'))

    # 6. Perbarui Visualisasi Peta
    Map = geemap.Map()
    Map.centerObject(semarang, 12)

    # Layer Citra Radar
    Map.addLayer(s1_latest, {'min': -25, 'max': 0, 'palette': ['black', 'white']}, 'Sentinel-1 Terbaru (dB)', False)
    
    # Layer Deteksi Genangan Baru (Merah Menyala)
    Map.addLayer(current_flooding, {'palette': ['red']}, 'Deteksi Genangan Real-Time')
    Map.addLayer(semarang.style(color='black', fillColor='00000000', width=2), {}, 'Batas Administrasi')

    output_html = "peta_realtime_semarang.html"
    Map.to_html(output_html)
    print(f"Peta pemantauan terbaru berhasil diperbarui ke '{output_html}' pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print("Belum ada akuisisi citra satelit Sentinel-1 baru dalam 14 hari terakhir.")