import ee
import math
import pandas as pd

# 1. Inisialisasi
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

# -------------------------------------------------------------
# A. FITUR TOPOGRAFI: DEM, SLOPE, TWI
# -------------------------------------------------------------
dem = ee.Image("USGS/SRTMGL1_003").clip(roi).rename('elevation')
slope = ee.Terrain.slope(dem).rename('slope')

slope_rad = slope.multiply(math.pi / 180.0)
tan_slope = slope_rad.tan().add(0.001)
flow_acc = ee.Terrain.aspect(dem).sin().abs().multiply(100).add(1)
twi = flow_acc.divide(tan_slope).log().rename('twi')

# -------------------------------------------------------------
# B. FITUR CURAH HUJAN: CHIRPS
# -------------------------------------------------------------
chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
          .filterBounds(roi)
          .filterDate('2023-01-01', '2023-03-31')
          .select('precipitation')
          .sum()
          .clip(roi)
          .rename('precipitation'))

# -------------------------------------------------------------
# C. FITUR VEGETASI: NDVI (Sentinel-2 Tanpa Awan)
# -------------------------------------------------------------
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask)

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(roi)
      .filterDate('2023-01-01', '2023-05-31')  # Rentang waktu sedikit diperlebar agar tidak bolong awan
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
      .map(mask_s2_clouds)
      .median())

ndvi = s2.normalizedDifference(['B8', 'B4']).rename('ndvi').clip(roi)

# -------------------------------------------------------------
# D. MASKING AIR PERMANEN (JRC Global Surface Water)
# -------------------------------------------------------------
jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(roi)
permanent_water = jrc.select('occurrence').gt(80).unmask(0)

# -------------------------------------------------------------
# E. TARGET BANJIR: SENTINEL-1 SAR
# -------------------------------------------------------------
s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(roi)
      .filterDate('2023-01-01', '2023-03-31')
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .select('VV')
      .median()
      .clip(roi))

# Genangan air (VV < -13.05 dB) tanpa air permanen
flood_label = (s1.lt(-13.05)
               .And(permanent_water.Not())
               .rename('flood_label')
               .toInt())

# -------------------------------------------------------------
# F. GABUNGKAN SEMUA BAND
# -------------------------------------------------------------
composite = (dem
             .addBands(slope)
             .addBands(twi)
             .addBands(chirps)
             .addBands(ndvi)
             .addBands(flood_label))

print("Band yang berhasil disiapkan:", composite.bandNames().getInfo())

# -------------------------------------------------------------
# G. STRATIFIED SAMPLING (Mengambil sampel seimbang)
# -------------------------------------------------------------
print("Sedang mengambil sampel piksel...")

# Mengambil masing-masing 2.500 sampel untuk kelas banjir (1) dan non-banjir (0)
stratified_samples = composite.stratifiedSample(
    numPoints=2500,
    classBand='flood_label',
    region=roi,
    scale=30,
    seed=42,
    geometries=False
)

# Konversi ke DataFrame
features_list = stratified_samples.getInfo()['features']
data = [f['properties'] for f in features_list]
df = pd.DataFrame(data).dropna()

output_filename = "dataset_banjir_semarang.csv"
df.to_csv(output_filename, index=False)

print(f"\nSelesai! Dataset berhasil disimpan ke '{output_filename}'")
print(f"Total baris data: {len(df)}")
print(f"Distribusi kelas label banjir:\n{df['flood_label'].value_counts()}")
print("\nContoh 5 baris pertama:")
print(df.head())