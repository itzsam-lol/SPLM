"""
Quarterly execution runner for the Indian retail SPLM pipeline.
Replaces the old daily US runner.
"""
import os
import argparse
import logging
from datetime import datetime

# Improvised imports to demonstrate orchestration
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from data.india_universe import INDIA_RETAIL_UNIVERSE
from data.nse_earnings_loader import get_nse_quarterly_results, compute_revenue_surprise
from data.sentinel_india import get_quarterly_composite, init_ee
from models.occupancy_cv import OccupancyCVModel
from signals.pai_constructor import PAIConstructor
from signals.ami_constructor import AMIConstructor
from signals.divergence_signal import DivergenceSignal

logger = logging.getLogger(__name__)

def run_pipeline(mode='real', dry_run=False, start_year=2023, end_year=2023, ticker_filter=None):
    log_messages = []
    def log_step(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        log_messages.append(f"[{datetime.now().isoformat()}] {msg}")
    
    log_step(f"Starting Quarterly Pipeline (Mode: {mode}, Range: {start_year}-{end_year})")
    
    if mode == 'synthetic':
        log_step("Synthetic mode selected. (Functionality available in notebooks/01)")
        return

    # Step 0: Init Earth Engine
    log_step("Step 0: Initializing Earth Engine...")
    try:
        init_ee()
    except Exception as e:
        log_step(f"EE Init Failed: {e}")

    registry_path = os.path.join("data", "location_registry_india.geojson")
    if not os.path.exists(registry_path):
        log_step("Warning: Location registry missing. Run 'python data/location_registry.py --build-india' first.")
        return

    cv_model = OccupancyCVModel()
    all_occupancy_rows = []

    # Iterate through universe and time
    tickers_to_process = INDIA_RETAIL_UNIVERSE.keys()
    if ticker_filter:
        tickers_to_process = [ticker_filter]

    for ticker in tickers_to_process:
        log_step(f"Processing Universe Member: {ticker}")
        
        # Step 1: Earnings
        log_step(f"  [{ticker}] Fetching NSE Earnings...")
        earnings_df = get_nse_quarterly_results(ticker)
        
        for year in range(start_year, end_year + 1):
            for q in range(1, 5):
                log_step(f"  [{ticker}] Q{q} {year} processing...")
                
                # Step 2: Imagery
                img_res = get_quarterly_composite(registry_path, year, q, ticker, dry_run=dry_run)
                if not img_res.get('success'):
                    log_step(f"    - Imagery Step Failed: {img_res.get('error')}")
                    continue
                
                # Step 3: CV Extraction
                success_chips = 0
                total_occupancy = 0
                import json # Moved import here as it's only used in this block
                for i, chip_path in enumerate(img_res.get('paths', [])):
                    # We need the polygon from the geojson
                    with open(registry_path, 'r') as f:
                        geo_data = json.load(f)
                    feats = [f for f in geo_data['features'] if f['properties']['ticker'] == ticker]
                    if i < len(feats):
                        from shapely.geometry import Polygon
                        poly = Polygon(feats[i]['geometry']['coordinates'][0])
                        res = cv_model.process_scene(chip_path, poly, feats[i]['properties'].get('lot_capacity_est', 50))
                        total_occupancy += res.get('occupancy_ratio', 0)
                        success_chips += 1

                if success_chips > 0:
                    avg_occ = total_occupancy / success_chips
                    all_occupancy_rows.append({
                        'ticker': ticker, 'year': year, 'quarter': q, 'raw_occupancy_ratio': avg_occ
                    })
                    log_step(f"    - CV Extraction Success. Avg Occupancy: {avg_occ:.4f}")

    if not all_occupancy_rows:
        log_step("Pipeline finished: No data processed.")
        return

    # Step 4: Aggregation and Signal Construction
    log_step("Step 4: Consolidating Occupancy into PAI...")
    occ_df = pd.DataFrame(all_occupancy_rows)
    pai_gen = PAIConstructor()
    pai_df = pai_gen.build_pai(occ_df)

    # Step 5: AMI Targets
    log_step("Step 5: Consolidating Earnings into AMI Targets...")
    all_earnings = []
    processed_tickers = occ_df['ticker'].unique()
    for ticker in processed_tickers:
        e_path = os.path.join("data", "raw", f"nse_earnings_{ticker}.parquet")
        if os.path.exists(e_path):
            all_earnings.append(pd.read_parquet(e_path))
    
    if all_earnings:
        earnings_master = pd.concat(all_earnings)
        earnings_master = compute_revenue_surprise(earnings_master)
        ami_gen = AMIConstructor(data_source='revenue_surprise')
        ami_df = ami_gen.generate_signal(earnings_master)
    else:
        ami_df = pd.DataFrame(columns=['ticker', 'year', 'quarter', 'revenue_surprise_zscore', 'surprise_direction'])

    # Step 6: Final Merge & Divergence
    log_step("Step 6: Merging PAI and AMI for Divergence Analysis...")
    div_gen = DivergenceSignal()
    master_df = div_gen.generate_signal(pai_df, ami_df)

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    out_path = os.path.join("data", "processed", "final_signals.csv")
    master_df.to_csv(out_path, index=False)
    
    log_step(f"Pipeline Complete. Final signals saved to {out_path}")
    print("\n--- Summary Performance Table ---")
    if not master_df.empty:
        # Check for columns to avoid error
        cols = ['ticker', 'year', 'quarter', 'pai_zscore', 'revenue_surprise_zscore', 'delta_signal_smooth']
        print(master_df[[c for c in cols if c in master_df.columns]].tail(10))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='real')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--ticker', type=str, help='Ticker to process specifically')
    parser.add_argument('--start-year', type=int, default=2023)
    parser.add_argument('--end-year', type=int, default=2023)
    args = parser.parse_args()
    
    run_pipeline(mode=args.mode, dry_run=args.dry_run, start_year=args.start_year, end_year=args.end_year, ticker_filter=args.ticker)
