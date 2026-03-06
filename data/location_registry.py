"""
Location Registry constructor.
Maps tickers to physical store locations and their parking lot polygons.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import osmnx as ox
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocationRegistry:
    def __init__(self, tickers: List[str]):
        """
        Initialize the registry for a specific universe of tickers.
        Primarily focused on Consumer Discretionary and Consumer Staples.
        """
        self.tickers = tickers
        # Buffer distance in meters to search around store location for parking
        self.buffer_dist_m = 200  
        # Area per parking space in square meters (estimate)
        self.sqm_per_space = 25.0 

    def get_store_locations(self, ticker: str) -> pd.DataFrame:
        """
        Mock implementation: Fetch known store locations for a ticker.
        In production, this would read from a vendor database or scrape 10-k filings.
        """
        # Return mock DataFrame of store locations
        return pd.DataFrame([
            {"ticker": ticker, "store_id": f"{ticker}_1", "lat": 34.0522, "lon": -118.2437},
            {"ticker": ticker, "store_id": f"{ticker}_2", "lat": 40.7128, "lon": -74.0060}
        ])

    def fetch_parking_polygons(self, lat: float, lon: float) -> Polygon:
        """
        Use OSM Overpass API to pull `amenity=parking` polygons within `buffer_dist_m` of coords.
        """
        try:
            # osmnx expects distance in meters
            tags = {"amenity": "parking"}
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=self.buffer_dist_m)
            
            if gdf.empty:
                return None
            
            # Filter to just polygons/multipolygons representing parking lots
            polygons = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
            
            if polygons.empty:
                return None
                
            # Naively take the largest parking lot found in the buffer
            polygons['area'] = polygons.geometry.area
            largest_polygon = polygons.sort_values(by='area', ascending=False).iloc[0].geometry
            return largest_polygon

        except Exception as e:
            logger.warning(f"Failed to fetch OSM polygon for {lat}, {lon}: {e}")
            return None

    def estimate_lot_capacity(self, polygon: Polygon) -> int:
        """
        Estimate the vehicle capacity of a parking lot based on its area.
        Note: The polygon must be in a projected CRS (meters) for accurate area.
        """
        if polygon is None:
            return 0
        
        # Simplified: assumes polygon area is roughly accurate (though in degrees it isn't,
        # in practice we should project to local UTM before computing area)
        # For demonstration, we'll pretend area is in sqm if we had projected it.
        # Here we mock a projection for a simple area calc
        dummy_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        dummy_gdf = dummy_gdf.to_crs(dummy_gdf.estimate_utm_crs())
        area_sqm = dummy_gdf.geometry.area.iloc[0]
        
        return int(area_sqm / self.sqm_per_space)

    def cross_validate_counts(self, registry_df: pd.DataFrame, ticker: str, reported_10k_count: int) -> float:
        """
        Cross-validate mapping counts against 10-K reported store counts.
        Returns the coverage percentage.
        """
        actual_count = len(registry_df[registry_df['ticker'] == ticker])
        if reported_10k_count == 0:
            return 0.0
        return min(100.0, (actual_count / reported_10k_count) * 100.0)

    def build_registry(self) -> gpd.GeoDataFrame:
        """
        Execute the pipeline to build the full location registry.
        """
        all_store_records = []

        for ticker in self.tickers:
            stores_df = self.get_store_locations(ticker)
            
            for _, split_row in stores_df.iterrows():
                poly = self.fetch_parking_polygons(split_row['lat'], split_row['lon'])
                capacity = self.estimate_lot_capacity(poly)
                
                record = {
                    "ticker": split_row['ticker'],
                    "store_id": split_row['store_id'],
                    "lat": split_row['lat'],
                    "lon": split_row['lon'],
                    "polygon_geojson": poly.__geo_interface__ if poly else None,
                    "geometry": poly,
                    "lot_capacity_est": capacity,
                    "data_vendor_coverage_pct": 100.0  # Mock value
                }
                all_store_records.append(record)
                
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(all_store_records, geometry='geometry', crs="EPSG:4326")
        
        # We can drop the geometry column if we just want pure pandas with geojson column
        # but returning GeoDataFrame is usually more useful for downstream geospatial work.
        return gdf

    def update_registry_monthly(self, output_path: str):
        """
        Utility to update registry and save to disk.
        """
        logger.info("Building updated location registry...")
        gdf = self.build_registry()
        # Drop raw shapely objects for CSV export, keep geojson string
        export_df = gdf.drop(columns=['geometry'])
        export_df.to_csv(output_path, index=False)
        logger.info(f"Registry saved to {output_path}")

if __name__ == "__main__":
    registry = LocationRegistry(["WMT", "TGT"])
    # registry.update_registry_monthly("location_registry.csv")
