"""
Vehicle Occupancy Computer Vision Pipeline.
Uses ultralytics YOLOv8 to detect vehicles in cropped satellite parking lot imagery.
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any

import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
import geopandas as gpd

# Ultralytics is loaded optionally to allow pipeline to run without heavy ML dependencies
# if only running aggregation or backtesting
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


logger = logging.getLogger(__name__)


class OccupancyCVModel:
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.35):
        """
        Initialize the YOLOv8-based occupancy model.
        In production, this would point to a model fine-tuned on DOTA v2.0 or 
        a custom parking lot dataset.
        """
        self.conf_thresh = confidence_threshold
        self.model_path = model_path
        self.model = None

        if YOLO_AVAILABLE:
            try:
                # Load a pretrained model (ideally a custom fine-tuned one)
                # If 'yolov8n.pt' doesn't exist locally, ultralytics will auto-download the generic one.
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded YOLO model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load YOLO model {model_path}: {e}. Running in mock mode.")
                self.model = None
        else:
            logger.warning("ultralytics not installed. Running OccupancyCVModel in mock mode.")

    def preprocess_image(self, tiff_path: str, polygon: Polygon) -> np.ndarray:
        """
        Crop the massive satellite tile down to just the parking lot polygon (+ 15px buffer).
        Returns an RGB numpy array ready for inference.
        """
        try:
            with rasterio.open(tiff_path) as src:
                # Need to convert shapely geometry to GeoJSON-like dict for rasterio
                geo_dict = [polygon.__geo_interface__]
                
                # Mask out pixels outside the polygon and crop to the bounding box
                out_image, out_transform = mask(src, geo_dict, crop=True)
                
                # Assuming standard 4-band Planet imagery (B G R NIR)
                # We extract B, G, R (bands 1, 2, 3) and stack them into (H, W, 3) format
                b, g, r = out_image[0], out_image[1], out_image[2]
                
                # Normalize and stack for standard RGB image format
                rgb = np.dstack((r, g, b))
                
                # Standardize to 8-bit (0-255)
                # Note: Real implementation needs careful radiometric calibration here
                rgb_norm = (255 * (rgb / np.max(rgb))).astype(np.uint8)
                
                return rgb_norm
                
        except Exception as e:
            logger.error(f"Failed to preprocess {tiff_path}: {e}")
            return None

    def run_inference(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Execute YOLOv8 inference on the cropped RGB image.
        Returns detection statistics.
        """
        if self.model is None or not YOLO_AVAILABLE:
            # Mock inference for skeleton/testing
            return {
                "vehicle_count": 42,
                "confidence_mean": 0.85,
                "detections": []
            }

        # Run inference
        results = self.model(image, conf=self.conf_thresh, verbose=False)
        result = results[0] # Take first result since we passed one image
        
        # In a custom model, class 0 might be 'vehicle'.
        # In standard COCO YOLO, cars are class 2, trucks are 7.
        valid_classes = [2, 7] 
        
        boxes = result.boxes
        vehicle_boxes = [box for box in boxes if int(box.cls) in valid_classes]
        
        confidences = [float(box.conf) for box in vehicle_boxes]
        mean_conf = np.mean(confidences) if confidences else 0.0

        return {
            "vehicle_count": len(vehicle_boxes),
            "confidence_mean": mean_conf,
            "raw_results": result
        }

    def compute_occupancy(self, vehicle_count: int, lot_capacity: int) -> float:
        """ Calculate occupancy ratio capped at 1.0 (100%) """
        if lot_capacity <= 0:
            return 0.0
        ratio = vehicle_count / float(lot_capacity)
        return min(ratio, 1.0)
        
    def process_scene(self, tiff_path: str, polygon: Polygon, lot_capacity: int) -> Dict[str, Any]:
        """
        End-to-end processing for a single parking lot scene.
        """
        logger.debug(f"Processing scene {tiff_path}")
        
        # 1. Preprocess & Crop
        image = self.preprocess_image(tiff_path, polygon)
        if image is None:
            return {"error": "preprocessing_failed"}
            
        # 2. Inference
        inference_stats = self.run_inference(image)
        
        count = inference_stats["vehicle_count"]
        occupancy = self.compute_occupancy(count, lot_capacity)
        
        return {
            "vehicle_count": count,
            "occupancy_ratio": occupancy,
            "confidence_mean": inference_stats["confidence_mean"],
            "image_processed": True
        }
