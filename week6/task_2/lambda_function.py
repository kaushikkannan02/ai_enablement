import json
import urllib.request
import urllib.error
import gzip
import io
import html
from html.parser import HTMLParser

MAX_BYTES = 500_000  # 500 KB limit


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.chunks = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.chunks.append(text)

    def get_text(self):
        return " ".join(self.chunks)


def fetch_url(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "BedrockWebCrawler/1.0",
            "Accept-Encoding": "gzip"
        }
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read(MAX_BYTES + 1)

        if len(raw) > MAX_BYTES:
            raise ValueError("Page too large")

        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()

        return raw.decode("utf-8", errors="ignore")


def clean_html(html_text: str) -> str:
    parser = TextExtractor()
    parser.feed(html_text)
    return html.unescape(parser.get_text())


def lambda_handler(event, context):
    """
    Expected event:
    { "url": "https://example.com" }
    """
    url = event.get("url")

    if not url:
        return {
            "error": "Missing 'url' parameter"
        }

    try:
        html_text = fetch_url(url)
        text = clean_html(html_text)

        return {
            "url": url,
            "content": text[:4000]  # safety trim
        }

    except Exception as e:
        return {
            "url": url,
            "error": str(e)
        }
