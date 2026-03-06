import requests
from bs4 import BeautifulSoup

def search_screener(query):
    url = f"https://www.screener.in/api/company/search/?q={query}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code == 200:
        print(f"Results for {query}:", r.json())
    else:
        print(f"Search failed for {query}")

search_screener('DMART')
search_screener('Avenue Supermarts')
search_screener('BARBEQUE')
search_screener('Jubilant FoodWorks')
search_screener('Westlife')
search_screener('Trent')
search_screener('Devyani')
