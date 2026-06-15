import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os

JP_TICKERS = [f"{t}.T" for t in range(1617, 1634)]

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
}

opens = []
print("Fetching prices from Google Finance...")
for tk in JP_TICKERS:
    code = tk.replace(".T", "")
    url = f"https://www.google.com/finance/quote/{code}:TYO"
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        price_div = soup.find("div", class_="N6SYTe")
        if not price_div:
            price_div = soup.find("span", class_="N6SYTe")
        if not price_div:
            price_div = soup.find("div", class_="YMlKec fxKbKc")
        if not price_div:
            for p in soup.find_all(attrs={"jsname": "Pdsbrc"}):
                if "¥" in p.text:
                    price_div = p
                    break
        if not price_div:
            p_elements = soup.find_all(attrs={"jsname": "Pdsbrc"})
            if p_elements:
                price_div = p_elements[-1]

        if price_div:
            price_text = price_div.text.replace("¥", "").replace(",", "").strip()
            price = float(price_text)
        else:
            price = 0.0
        opens.append({"ticker": tk, "open_price": price})
    except Exception as e:
        print(f"Failed to fetch {tk}: {e}")
        opens.append({"ticker": tk, "open_price": 0.0})
    time.sleep(0.5)

df = pd.DataFrame(opens)
df.to_csv("jp_opens_google.csv", index=False)
print("CSV generated: jp_opens_google.csv")
print(df)
