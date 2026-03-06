import pytest
import numpy as np
import pandas as pd
from models.cloud_quality_gate import is_monsoon_quarter, CloudQualityGate
from signals.pai_constructor import PAIConstructor

def test_q3_is_flagged():
    gate = CloudQualityGate()
    metrics = gate.analyze_scene_quality("fake_Q3_2022.tif", "dummy_polygon_mockout", quarter=3)
    assert metrics['monsoon_flag'] == True

def test_non_q3_not_flagged():
    gate = CloudQualityGate()
    for q in [1, 2, 4]:
        metrics = gate.analyze_scene_quality(f"fake_Q{q}_2022.tif", "dummy_polygon_mockout", quarter=q)
        assert metrics['monsoon_flag'] == False

def test_q3_excluded_from_pai():
    pai = PAIConstructor()
    # Mock data directly to build_pai
    agg_occ = pd.DataFrame([
        {'ticker': 'DMART', 'year': 2022, 'quarter': 1, 'raw_occupancy_ratio': 0.8},
        {'ticker': 'DMART', 'year': 2022, 'quarter': 2, 'raw_occupancy_ratio': 0.85},
        {'ticker': 'DMART', 'year': 2022, 'quarter': 3, 'raw_occupancy_ratio': 0.9}, # Monsoon
        {'ticker': 'DMART', 'year': 2022, 'quarter': 4, 'raw_occupancy_ratio': 0.95},
    ])
    df = pai.build_pai(agg_occ)
    
    # Check Q3 is NaN
    q3_row = df[(df['year'] == 2022) & (df['quarter'] == 3)].iloc[0]
    assert pd.isna(q3_row['pai_zscore'])
    assert q3_row['monsoon_quarter'] == True
