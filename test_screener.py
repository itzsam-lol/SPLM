import requests
from bs4 import BeautifulSoup

def test_scr():
    url = f"https://www.screener.in/company/AVENUE/consolidated/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Safari/537.36"}
    
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        url = f"https://www.screener.in/company/AVENUE/"
        r = requests.get(url, headers=headers)
        print(f"Fallback fetch status: {r.status_code}")
    
    soup = BeautifulSoup(r.text, 'html.parser')
    quarters_section = soup.find('section', id='quarters')
    if not quarters_section:
        print("No quarters section")
        return
        
    table = quarters_section.find('table', class_='data-table')
    if not table:
        print("No table")
        return
        
    headers_list = [t.text.strip() for t in table.find('thead').find_all('th')]
    print("Headers:", headers_list)

test_scr()
