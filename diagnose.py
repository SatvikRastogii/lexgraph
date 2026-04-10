import requests
from bs4 import BeautifulSoup

url = "https://indiankanoon.org/search/"
params = {"formInput": "Article 21 Supreme Court", "pagenum": 0}
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

response = requests.get(url, params=params, headers=headers, timeout=15)
soup = BeautifulSoup(response.text, "html.parser")

# Save the page so we can inspect it
with open("kanoon_test.html", "w", encoding="utf-8") as f:
    f.write(response.text)

print(f"Status code: {response.status_code}")
print(f"Page length: {len(response.text)} characters")

# Try finding any divs with links
all_links = soup.find_all("a", href=True)
doc_links = [a["href"] for a in all_links if "/doc/" in a["href"]]
print(f"Document links found: {len(doc_links)}")
for link in doc_links[:5]:
    print(f"  {link}")
