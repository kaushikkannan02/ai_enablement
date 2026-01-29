import json
import urllib.request
import gzip
import io
from bs4 import BeautifulSoup

MAX_BYTES = 1_000_000
TIMEOUT = 8

def fetch_url(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "BedrockWebCrawler/1.0",
            "Accept-Encoding": "gzip"
        }
    )

    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        raw = r.read(MAX_BYTES)
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        return raw.decode("utf-8", errors="ignore")

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())[:5000]

def lambda_handler(event, context):
    url = event.get("url")
    if not url:
        return {"error": "Missing url"}

    html = fetch_url(url)
    text = clean_html(html)

    return {
        "source_url": url,
        "text": text
    }
