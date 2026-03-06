# Commit sequence
git add data/processed/final_signals.csv
git commit -m "Update processed final signals with 5-ticker subset"
git push

git add data/raw/imagery/DMART/Q2_2021*
git commit -m "Add DMART satellite imagery for early 2021"
git push

git add data/raw/imagery/DMART/* data/raw/imagery/JUBLFOOD/*
git commit -m "Add remaining DMART and JUBLFOOD satellite images"
git push

git add data/raw/imagery/METRO/* data/raw/imagery/TRENT/* data/raw/imagery/WESTLIFE/*
git commit -m "Add raw images for METRO, TRENT, WESTLIFE targets"
git push

git add fix_signals.py
git commit -m "chore: add temporary script to fix signal column names"
git push

git add generate_results.py
git commit -m "refactor: add script to generate paper results, tables and figures"
git push

git add results/02_tables/table1*.csv results/02_tables/table2*.csv
git commit -m "docs: add data coverage and case study tables"
git push

git add results/02_tables/ablation* results/02_tables/table4*.csv
git commit -m "docs: include ablation study results"
git push

git add results/02_tables/table3* results/02_tables/table5* results/03_figures/*
git commit -m "docs: add IC analysis, backtest status and all figures"
git push

git add results/00_validation/* results/paper_results_summary.json
git commit -m "docs: record validation logs and final json summary"
git push

git add .
git commit -m "chore: final minor cleanups and untracked files"
git push
