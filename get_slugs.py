import requests
import json

def get_slug(query):
    url = f"https://www.screener.in/api/company/search/?q={query}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code == 200 and r.json():
        return r.json()[0]['url'].split('/')[2]
    return query

tickers = ['DMART', 'METROBRAND', 'BARBEQUE', 'JUBLFOOD', 'WESTLIFE', 'TRENT', 'DEVYANI', 'SHOPERSTOP', 'VMART', 'ABFRL', 'SPENCERS', 'SAPPHIRE']
for t in tickers:
    print(f"{t}: {get_slug(t)}")
