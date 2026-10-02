// Sentinel-2 L2A indices for durian orchard polygon, 2020-2026 (run in code.earthengine.google.com)
var poly = ee.Geometry.Polygon([[[102.2123655,12.6494732],[102.2128604,12.6492678],[102.2132364,12.6497048],[102.213768,12.6506003],[102.2143889,12.6519322],[102.2127727,12.652682],[102.212306,12.6523418],[102.2114791,12.6507442],[102.2116678,12.6503253],[102.2118234,12.6494747],[102.2123655,12.6494732]]]);
var csp = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(poly)
  .filterDate('2020-01-01','2026-09-30').linkCollection(csp, ['cs_cdf']);
var feats = s2.map(function(img){
  var scl = img.select('SCL');
  var clear = img.select('cs_cdf').gte(0.6).and(scl.neq(3)).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  var r = img.select(['B2','B4','B5','B8','B11']).divide(10000);
  var ndvi = r.normalizedDifference(['B8','B4']).rename('NDVI');
  var evi = r.expression('2.5*(N-R)/(N+6*R-7.5*B+1)',{N:r.select('B8'),R:r.select('B4'),B:r.select('B2')}).rename('EVI');
  var ndre = r.normalizedDifference(['B8','B5']).rename('NDRE');
  var ndmi = r.normalizedDifference(['B8','B11']).rename('NDMI');
  var idx = ee.Image.cat([ndvi,evi,ndre,ndmi]).updateMask(clear);
  var m = idx.reduceRegion({reducer: ee.Reducer.mean(), geometry: poly, scale: 10, maxPixels: 1e8});
  var cnt = ee.Image.constant(1).rename('c').addBands(clear.rename('k')).reduceRegion({reducer: ee.Reducer.sum(), geometry: poly, scale: 10});
  return ee.Feature(null, m).set({date: img.date().format('YYYY-MM-dd HH:mm'), tile: img.get('MGRS_TILE'),
    scene_cloud_pct: img.get('CLOUDY_PIXEL_PERCENTAGE'), n_pix: cnt.get('c'), n_clear: cnt.get('k')});
});
Export.table.toDrive({collection: ee.FeatureCollection(feats).sort('date'), description: 'S2_orchard_indices_2020_2026', fileFormat: 'CSV'});
