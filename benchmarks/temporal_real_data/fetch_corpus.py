"""
Build a versioned corpus from Wikipedia revision history.

For each article, take the lead section as it stood on several dates
(2018, 2020, 2022, 2024 and today). Every snapshot becomes a document
whose created_at is the revision timestamp; the newest snapshot is the
ground-truth "current" version. Snapshots whose lead text did not change
are dropped, so every older document really is stale.

    python fetch_corpus.py              # writes data/corpus.jsonl
    python fetch_corpus.py --limit 10   # quick run

Text is CC BY-SA 4.0 (Wikipedia contributors).
"""

import argparse
import calendar
import html
import json
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
UA = "TideVecBenchmark/0.3 (https://github.com/ashishodu2023/TideVec)"
DATES = ["2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"]

ARTICLES = [
    # Organisations and products that changed a lot
    "OpenAI", "Twitter", "Meta Platforms", "Tesla, Inc.", "Nvidia", "Netflix", "TikTok",
    "SpaceX", "Microsoft", "Apple Inc.", "Amazon (company)", "Alphabet Inc.", "Intel",
    "Samsung Electronics", "Uber", "Spotify", "Zoom Video Communications", "Reddit",
    # Software
    "Python (programming language)", "Rust (programming language)", "Linux kernel",
    "Kubernetes", "Android (operating system)", "MacOS", "Microsoft Windows", "Firefox",
    "Google Chrome", "Java (programming language)", "TypeScript", "PostgreSQL",
    # Science, space, health
    "Artemis program", "James Webb Space Telescope", "International Space Station",
    "Mars 2020", "Large Hadron Collider", "CRISPR gene editing", "Moderna",
    # Economy, society, places
    "Bitcoin", "Ethereum", "Euro", "World population", "Brexit", "European Union",
    "United Nations", "Burj Khalifa", "Tokyo", "Singapore", "Nairobi", "Iceland",
    # Sport and culture
    "FIFA World Cup", "Summer Olympic Games", "Formula One", "Wimbledon Championships",
    "Marvel Cinematic Universe", "Taylor Swift",
]


def api(params):
    params = dict(params, format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def revision_at(title, date):
    """Latest revision at or before `date` (None for today's revision)."""
    params = {"action": "query", "prop": "revisions", "titles": title,
              "rvprop": "ids|timestamp", "rvlimit": 1, "redirects": 1}
    if date:
        params["rvstart"] = f"{date}T00:00:00Z"
    page = api(params)["query"]["pages"][0]
    revs = page.get("revisions")
    return revs[0] if revs else None


class _LeadText(HTMLParser):
    """Collect paragraph text, skipping tables, references and styles."""

    SKIP = {"table", "style", "sup", "script"}

    def __init__(self):
        super().__init__()
        self.depth_skip = 0
        self.in_p = 0
        self.parts, self.cur = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth_skip += 1
        elif tag == "p":
            self.in_p += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth_skip:
            self.depth_skip -= 1
        elif tag == "p" and self.in_p:
            self.in_p -= 1
            text = " ".join("".join(self.cur).split())
            if text:
                self.parts.append(text)
            self.cur = []

    def handle_data(self, data):
        if self.in_p and not self.depth_skip:
            self.cur.append(data)


def lead_text(revid, max_chars=1500):
    data = api({"action": "parse", "oldid": revid, "prop": "text", "section": 0})
    parser = _LeadText()
    parser.feed(data["parse"]["text"])
    text = html.unescape(" ".join(parser.parts))
    text = re.sub(r"\[\d+\]|\(listen\)", "", text)
    return text[:max_chars].rsplit(" ", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "corpus.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    titles = ARTICLES[: args.limit] if args.limit else ARTICLES
    n_docs = 0
    with out.open("w") as f:
        for title in titles:
            seen_rev, seen_text, versions = set(), set(), []
            for date in DATES + [None]:
                rev = revision_at(title, date)
                if not rev or rev["revid"] in seen_rev:
                    continue
                seen_rev.add(rev["revid"])
                text = lead_text(rev["revid"])
                if len(text) < 200:
                    continue
                versions.append({"revid": rev["revid"], "timestamp": rev["timestamp"], "text": text})
                time.sleep(0.1)
            # Keep one document per distinct lead text; the newest is current.
            kept = []
            for v in reversed(versions):
                if v["text"] not in seen_text:
                    seen_text.add(v["text"])
                    kept.append(v)
            kept.reverse()
            if len(kept) < 2:
                print(f"skip {title}: fewer than 2 distinct versions")
                continue
            for i, v in enumerate(kept):
                ts = calendar.timegm(time.strptime(v["timestamp"], "%Y-%m-%dT%H:%M:%SZ")) * 1000
                f.write(json.dumps({
                    "article": title, "version": i, "is_current": i == len(kept) - 1,
                    "revid": v["revid"], "created_at": ts, "text": v["text"],
                }) + "\n")
                n_docs += 1
            print(f"{title}: {len(kept)} versions")
    print(f"wrote {n_docs} documents to {out}")


if __name__ == "__main__":
    main()
