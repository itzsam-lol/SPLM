"""
Module: data/india_universe.py
Purpose: Define the pilot universe of Indian retail stocks.
Data Sources: None
Synthetic: False
Point-in-Time: N/A
"""

import argparse
import json

INDIA_RETAIL_UNIVERSE = {
    'DMART':       {'name': 'Avenue Supermarts',          'sector': 'hypermarket',   'nse': True},
    'TRENT':       {'name': 'Tata Trent (Westside/Zudio)','sector': 'fashion',       'nse': True},
    'JUBLFOOD':    {'name': 'Jubilant FoodWorks (Domino\'s)','sector': 'qsr',         'nse': True},
    'WESTLIFE':    {'name': 'Westlife Foodworld (McD)',    'sector': 'qsr',           'nse': True},
    'DEVYANI':     {'name': 'Devyani Intl (KFC/PH)',      'sector': 'qsr',           'nse': True},
    'SHOPERSTOP':  {'name': 'Shoppers Stop',               'sector': 'department',    'nse': True},
    'VMART':       {'name': 'V-Mart Retail',               'sector': 'value_retail',  'nse': True},
    'ABFRL':       {'name': 'Aditya Birla Fashion (Pantaloons)','sector': 'fashion',  'nse': True},
    'SPENCERS':    {'name': "Spencer's Retail",            'sector': 'hypermarket',   'nse': True},
    'SAPPHIRE':    {'name': 'Sapphire Foods (KFC N/E)',    'sector': 'qsr',           'nse': True},
    'BARBEQUE':    {'name': 'Barbeque Nation',             'sector': 'casual_dining', 'nse': True},
    'METRO':       {'name': 'Metro Brands',                'sector': 'footwear',      'nse': True},
}

def get_yfinance_ticker(symbol):
    """Returns the Yahoo Finance compatible ticker for NSE."""
    return symbol + '.NS'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Indian Retail Universe")
    parser.add_argument('--list', action='store_true', help="List all tickers with metadata")
    args = parser.parse_args()

    if args.list:
        for symbol, metadata in INDIA_RETAIL_UNIVERSE.items():
            print(f"{symbol}: {json.dumps(metadata)}")
