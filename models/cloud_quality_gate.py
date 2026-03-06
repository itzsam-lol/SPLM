"""
Cloud and Shadow Quality Gate.
Detects occlusion in satellite imagery to prevent spurious zero-counts.
"""
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
import logging

logger = logging.getLogger(__name__)

class CloudQualityGate:
    def __init__(self, max_cloud_fraction: float = 0.20, max_shadow_fraction: float = 0.35):
        """
        Initialize the Quality Gate.
        Images exceeding these fractions of occlusion will be rejected.
        """
        self.max_cloud_fraction = max_cloud_fraction
        self.max_shadow_fraction = max_shadow_fraction

    def analyze_scene_quality(self, tiff_path: str, polygon: Polygon) -> dict:
        """
        Analyze the cropped scene for cloud and shadow coverage.
        Returns occlusion metrics and a boolean pass/fail flag.
        """
        try:
            with rasterio.open(tiff_path) as src:
                # Mask out pixels outside the polygon
                out_image, out_transform = mask(src, [polygon.__geo_interface__], crop=True)
                
                # For a typical 4-band image (B, G, R, NIR)
                b, g, r = out_image[0], out_image[1], out_image[2]
                
                # ---------------------------------------------------------
                # Heuristic Cloud Detection (Very simplified for example)
                # Clouds are bright across all visible bands.
                # In production, use standard Cloud Masks provided by the vendor (e.g. UDM2 from Planet)
                # or a dedicated ML cloud segmentation model.
                # ---------------------------------------------------------
                
                # Assume values > 80% of max represent cloud/snow brightness
                # (Highly dependent on radiometric correction of the source imagery)
                brightness = (b.astype(float) + g.astype(float) + r.astype(float)) / 3.0
                cloud_mask = brightness > (np.max(brightness) * 0.8)
                
                # ---------------------------------------------------------
                # Heuristic Shadow Detection
                # Shadows are very dark. Often evaluated using (B-R)/(B+R) or similar indices.
                # ---------------------------------------------------------
                shadow_mask = brightness < (np.max(brightness) * 0.2)
                
                # Calculate fractions of the VALID (non-zero) polygon area
                valid_pixels_mask = b > 0  # Where data actually exists
                total_valid_pixels = np.sum(valid_pixels_mask)
                
                if total_valid_pixels == 0:
                     return {"passed": False, "cloud_fraction": 1.0, "shadow_fraction": 1.0, "error": "empty_polygon"}

                cloud_fraction = np.sum(cloud_mask & valid_pixels_mask) / total_valid_pixels
                shadow_fraction = np.sum(shadow_mask & valid_pixels_mask) / total_valid_pixels
                
                # Gate Logic
                passed = True
                if cloud_fraction > self.max_cloud_fraction or shadow_fraction > self.max_shadow_fraction:
                    passed = False
                    
                return {
                    "passed": passed,
                    "cloud_fraction": float(cloud_fraction),
                    "shadow_fraction": float(shadow_fraction)
                }

        except Exception as e:
            logger.error(f"Failed to analyze quality for {tiff_path}: {e}")
            # Err on the side of caution: if we can't analyze it, reject it.
            return {"passed": False, "cloud_fraction": 1.0, "shadow_fraction": 1.0, "error": str(e)}

    def is_valid(self, quality_metrics: dict) -> bool:
        """ Helper to just return the boolean """
        return quality_metrics.get("passed", False)
