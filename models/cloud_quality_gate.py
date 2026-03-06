"""
Cloud and Shadow Quality Gate.
Detects occlusion in satellite imagery to prevent spurious zero-counts.
Flags monsoon quarters based on date.
Data Sources: Tiff imagery
Synthetic: False
Point-in-Time: NA
"""
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
import logging
import re

logger = logging.getLogger(__name__)

MONSOON_MONTHS = [6, 7, 8, 9]  # June–September

def is_monsoon_quarter(year, quarter):
    """
    Q3 (Jul-Sep) -> return True
    All others   -> return False
    """
    return quarter == 3

class CloudQualityGate:
    def __init__(self, max_cloud_fraction: float = 0.20, max_shadow_fraction: float = 0.35):
        """
        Images exceeding these fractions of occlusion will be rejected.
        """
        self.max_cloud_fraction = max_cloud_fraction
        self.max_shadow_fraction = max_shadow_fraction

    def analyze_scene_quality(self, tiff_path: str, polygon: Polygon, quarter: int = None) -> dict:
        """
        Analyze the cropped scene for cloud and shadow coverage.
        Returns occlusion metrics and a boolean pass/fail flag.
        Images from monsoon quarters are NOT rejected - they are flagged.
        The PAI constructor will decide whether to use or exclude them.
        """
        if quarter is None:
            m = re.search(r'Q(\d)_', tiff_path)
            if m:
                quarter = int(m.group(1))
            else:
                quarter = 1
                
        monsoon_flag = is_monsoon_quarter(2022, quarter)
        
        try:
            with rasterio.open(tiff_path) as src:
                out_image, out_transform = mask(src, [polygon.__geo_interface__], crop=True)
                b, g, r = out_image[0], out_image[1], out_image[2]
                
                brightness = (b.astype(float) + g.astype(float) + r.astype(float)) / 3.0
                cloud_mask = brightness > (np.max(brightness) * 0.8)
                shadow_mask = brightness < (np.max(brightness) * 0.2)
                
                valid_pixels_mask = b > 0  # Where data actually exists
                total_valid_pixels = np.sum(valid_pixels_mask)
                
                if total_valid_pixels == 0:
                     return {"passed": False, "cloud_fraction": 1.0, "shadow_fraction": 1.0, "monsoon_flag": monsoon_flag, "error": "empty_polygon"}

                cloud_fraction = np.sum(cloud_mask & valid_pixels_mask) / total_valid_pixels
                shadow_fraction = np.sum(shadow_mask & valid_pixels_mask) / total_valid_pixels
                
                passed = True
                if cloud_fraction > self.max_cloud_fraction or shadow_fraction > self.max_shadow_fraction:
                    passed = False
                    
                return {
                    "passed": passed,
                    "cloud_fraction": float(cloud_fraction),
                    "shadow_fraction": float(shadow_fraction),
                    "monsoon_flag": monsoon_flag
                }

        except Exception as e:
            logger.error(f"Failed to analyze quality for {tiff_path}: {e}")
            return {"passed": False, "cloud_fraction": 1.0, "shadow_fraction": 1.0, "monsoon_flag": monsoon_flag, "error": str(e)}

    def is_valid(self, quality_metrics: dict) -> bool:
        return quality_metrics.get("passed", False)
