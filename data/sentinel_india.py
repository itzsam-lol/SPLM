"""
Module: data/sentinel_india.py
Purpose: Download quarterly Sentinel-2 composites via Google Earth Engine.
Data Sources: COPERNICUS/S2_SR_HARMONIZED (GEE API)
Synthetic: False
Point-in-Time: No strict PIT lag at extraction.
"""

import os
import time
import json
import argparse
import ee

import geemap

def init_ee(project_id='smpl-489416'):
    try:
        ee.Initialize(project=project_id)
    except Exception:
        print(f"Initializing Earth Engine with project: {project_id}")
        try:
            ee.Initialize(project=project_id)
        except Exception as e:
            print(f"Error initializing Earth Engine: {e}")
            print("Running `earthengine authenticate` may be required.")
            ee.Authenticate()
            ee.Initialize(project=project_id)

def get_quarterly_composite(polygon_geojson, year, quarter, symbol, dry_run=False, project_id='smpl-489416'):
    monsoon_flag = (quarter == 3)
    
    if quarter == 1:
        start_date, end_date = f"{year}-01-01", f"{year}-03-31"
    elif quarter == 2:
        start_date, end_date = f"{year}-04-01", f"{year}-06-14"
    elif quarter == 3:
        start_date, end_date = f"{year}-07-01", f"{year}-09-30"
    elif quarter == 4:
        start_date, end_date = f"{year}-10-01", f"{year}-12-31"
    else:
        raise ValueError("Quarter must be 1..4")

    out_dir = os.path.join("data", "raw", "imagery", symbol)
    os.makedirs(out_dir, exist_ok=True)
    
    features = []
    if os.path.exists(polygon_geojson):
        with open(polygon_geojson, 'r') as f:
            geo_data = json.load(f)
        features = [feat for feat in geo_data.get('features', []) if feat.get('properties', {}).get('ticker') == symbol]
    elif dry_run:
        features = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0,0], [1,0], [1,1], [0,1], [0,0]]]}}]

    if not features:
        return {'success': False, 'error': f'No polygons for {symbol}'}

    if dry_run:
        print(f"Would download Q{quarter} {year} composites for {symbol} ({len(features)} chips)")
        return {'success': True, 'dry_run': True}

    init_ee(project_id=project_id)
    
    success_count = 0
    paths = []
    
    for i, feat in enumerate(features):
        chip_path = os.path.join(out_dir, f"Q{quarter}_{year}_loc{i}.tif")
        if os.path.exists(chip_path):
             success_count += 1
             paths.append(chip_path)
             continue
             
        geom = ee.Geometry(feat['geometry'])
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(geom)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25)))
        
        count = collection.size().getInfo()
        if count == 0:
            print(f"  - Loc {i}: No clear images found.")
            continue

        composite = collection.median().select(['B4', 'B3', 'B2', 'B8', 'B11'])
        
        try:
            geemap.ee_export_image(composite, filename=chip_path, scale=10, region=geom, file_per_band=False)
            if os.path.exists(chip_path):
                success_count += 1
                paths.append(chip_path)
        except Exception as e:
            print(f"  - Loc {i} failed: {e}")

    if success_count > 0:
        return {'success': True, 'paths': paths, 'count': success_count, 'monsoon_flag': monsoon_flag}
    else:
        return {'success': False, 'error': 'All location downloads failed or no images available.'}

def batch_download(universe_dict, years=range(2019, 2025), project_id='smpl-489416'):
    log_dir = os.path.join("data", "manifests")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "download_log.jsonl")
    geojson_path = os.path.join("data", "location_registry_india.geojson")
    
    for symbol in universe_dict.keys():
        for year in years:
            for quarter in range(1, 5):
                res = get_quarterly_composite(geojson_path, year, quarter, symbol, project_id=project_id)
                with open(log_file, 'a') as f:
                    f.write(json.dumps({'symbol': symbol, 'year': year, 'quarter': quarter, **res}) + "\n")
                if res.get('success') and not res.get('dry_run'):
                    count = res.get('cloud_images_used', 0)
                    print(f"{symbol} Q{quarter} {year} -> \u2713 ({count} images composited)")
                time.sleep(2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--quarter', type=int, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--project', type=str, default='smpl-489416')
    args = parser.parse_args()
    
    geojson_path = os.path.join("data", "location_registry_india.geojson")
    print(f"Requesting Q{args.quarter} {args.year} composite for {args.ticker}...")
    res = get_quarterly_composite(geojson_path, args.year, args.quarter, args.ticker, args.dry_run, project_id=args.project)
    if res.get('success'):
        print(f"Success! {res.get('count')} location chips saved to {args.ticker}/ folder.")
    else:
        print(f"Failed: {res.get('error')}")
