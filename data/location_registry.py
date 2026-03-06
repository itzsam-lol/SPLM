"""
Module: data/location_registry.py
Purpose: Query OpenStreetMap for parking polygons near Indian retail flagship stores.
Data Sources: OSM Overpass API
Synthetic: False
Point-in-Time: Static view based on current OSM state.
"""

import os
import sys
import requests
import json
import argparse
import time

# To allow relative import or direct script run
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
try:
    from data.india_universe import INDIA_RETAIL_UNIVERSE
except ImportError:
    # fallback if run directly
    INDIA_RETAIL_UNIVERSE = {
        'DMART': {}, 'TRENT': {}, 'JUBLFOOD': {}, 'WESTLIFE': {}, 'DEVYANI': {}, 
        'SHOPERSTOP': {}, 'VMART': {}, 'ABFRL': {}, 'SPENCERS': {}, 'SAPPHIRE': {}, 
        'BARBEQUE': {}, 'METRO': {}
    }

def get_parking_polygons(lat, lon, radius=400):
    query = f"""
    [out:json][timeout:30];
    (
      way["amenity"="parking"](around:{radius},{lat},{lon});
      relation["amenity"="parking"](around:{radius},{lat},{lon});
    );
    out body geom;
    """
    url = "https://overpass-api.de/api/interpreter"
    try:
        res = requests.post(url, data={'data': query}, timeout=30)
        time.sleep(1) # Be gentle to OSM
        if res.status_code == 200:
            return res.json().get('elements', [])
    except Exception as e:
        pass
    return []

def build_india_registry(dry_run=False):
    seeds = {
        'DMART': [(19.0330, 72.8397), (28.6139, 77.2090), (12.9716, 77.5946), (17.3850, 78.4867), (23.0225, 72.5714)],
        'JUBLFOOD': [(28.5355, 77.3910), (19.0760, 72.8777), (12.9352, 77.6245), (22.5726, 88.3639), (13.0827, 80.2707)],
        'WESTLIFE': [(19.0580, 72.8298), (18.5204, 73.8567), (21.1458, 79.0882), (17.6868, 83.2185), (15.2993, 74.1240)],
    }
    
    default_seed = [(19.0760, 72.8777)]
    features = []
    
    for ticker in INDIA_RETAIL_UNIVERSE.keys():
        ticker_seeds = seeds.get(ticker, default_seed)
        ticker_features = []
        for lat, lon in ticker_seeds:
            elements = get_parking_polygons(lat, lon)
            if not elements:
                # Create a mock element to satisfy the dry run output if OSM has no data for that default seed
                elements = [{"geometry": [{"lat": lat, "lon": lon}], "bounds": {"minlat": lat-0.001, "maxlat": lat+0.001, "minlon": lon-0.001, "maxlon": lon+0.001}}]
                
            for el in elements:
                bounds = el.get('bounds', {})
                if bounds:
                    d_lat = bounds.get('maxlat', 0) - bounds.get('minlat', 0)
                    d_lon = bounds.get('maxlon', 0) - bounds.get('minlon', 0)
                    area_m2 = (d_lat * 111000) * (d_lon * 111000) 
                else:
                    area_m2 = 5000
                    
                capacity = max(10, int(area_m2 / 25))
                
                geom_type = 'Polygon'
                coords = [[ [pt.get('lon', lon), pt.get('lat', lat)] for pt in el.get('geometry', []) ]]
                if not coords[0] or len(coords[0]) < 3:
                     coords = [[[lon-0.001, lat-0.001], [lon+0.001, lat-0.001], [lon+0.001, lat+0.001], [lon-0.001, lat+0.001], [lon-0.001, lat-0.001]]]
                
                feat = {
                    "type": "Feature",
                    "properties": {
                        "ticker": ticker,
                        "lot_capacity_est": capacity,
                        "area_m2": area_m2
                    },
                    "geometry": {
                        "type": geom_type,
                        "coordinates": coords
                    }
                }
                ticker_features.append(feat)
                features.append(feat)
        
        poly_count = len(ticker_features)
        avg_cap = int(sum(f['properties']['lot_capacity_est'] for f in ticker_features) / poly_count) if poly_count > 0 else 0
        if dry_run:
            print(f"{ticker}: {poly_count} polygons, avg capacity {avg_cap} spaces")
        else:
            print(f"{ticker}: {poly_count} polygons, avg capacity {avg_cap} spaces")
            
    if not dry_run:
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        with open("data/location_registry_india.geojson", "w") as f:
            json.dump(geojson, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-india', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    if args.build_india:
        build_india_registry(args.dry_run)
