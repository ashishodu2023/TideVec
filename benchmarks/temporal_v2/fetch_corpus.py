"""
Build the temporal benchmark v2 corpora from real revision histories.

Three corpora, each a set of documents with 2+ dated versions; the newest
version of each document is the ground-truth "current" one:

  wiki     lead sections of Wikipedia articles (Vital articles level 3,
           Vital articles level 4 / Technology, and the 55 v1 articles)
           as of 2018-01-01, 2020-01-01, 2022-01-01, 2024-01-01 and today
  policy   lead sections of Wikipedia's own policies and guidelines
           (Verifiability, Biographies of living persons, ...) on the same
           dates: a real policy/handbook corpus with real edit history
  docs     PyPI project descriptions (READMEs) of ~150 popular packages,
           one per release current on each date: product docs whose
           install instructions, supported versions and APIs change

    python fetch_corpus.py                        # all corpora, ~40 min
    python fetch_corpus.py --corpus docs          # one corpus
    python fetch_corpus.py --corpus wiki --limit 50

Resumable: titles already present in data/<corpus>.jsonl are skipped.
Wikipedia text is CC BY-SA 4.0; README text belongs to each project.
"""

import argparse
import calendar
import html
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "temporal_real_data"))
import fetch_corpus as v1  # noqa: E402

DATES = ["2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"]
UA = v1.UA

POLICY_EXCLUDE = re.compile(
    r"Contents|Featured|Good (articles|topics)|Dashboard|Glossary|IRC|Discord|Meetup|Matrix|"
    r"FAQ|Days of the year|Community portal|Department directory|Directories|Essay directory|"
    r"Editor's index|In the news|List of (policies|guidelines)|Maintenance|Naming conventions \(")

DOCS_PACKAGES = """
requests numpy pandas scipy matplotlib scikit-learn django flask fastapi pydantic sqlalchemy
celery redis pytest tox black ruff mypy pylint isort flake8 poetry pip setuptools wheel virtualenv
boto3 botocore awscli google-cloud-storage azure-storage-blob httpx aiohttp urllib3 certifi
click typer rich tqdm jinja2 pyyaml toml tomli attrs cryptography paramiko pillow opencv-python
torch tensorflow keras jax transformers datasets tokenizers sentencepiece accelerate diffusers
langchain llama-index openai anthropic tiktoken spacy nltk gensim xgboost lightgbm catboost
statsmodels seaborn plotly bokeh dash streamlit gradio jupyter notebook ipython jupyterlab
sphinx mkdocs mkdocs-material uvicorn gunicorn starlette sanic tornado twisted websockets grpcio
protobuf psycopg2 pymongo elasticsearch kafka-python confluent-kafka pyspark dask ray polars
pyarrow duckdb networkx sympy numba cython pybind11 cffi lxml beautifulsoup4 scrapy selenium
playwright pytest-cov coverage hypothesis faker freezegun responses mock docker kubernetes ansible
fabric invoke sentry-sdk loguru structlog python-dotenv marshmallow cerberus jsonschema orjson
ujson msgpack arrow pendulum python-dateutil pytz babel alembic peewee tortoise-orm motor
""".split()


def ts_ms(iso):
    iso = iso.replace("Z", "").split(".")[0]
    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%S")) * 1000


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def api(params):
    """MediaWiki API call that backs off on HTTP 429 (honouring Retry-After)."""
    params = dict(params, format="json", formatversion="2")
    url = v1.API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if attempt == 7:
                raise
            wait = float(e.headers.get("Retry-After") or 0) if e.code == 429 else 0
            time.sleep(max(wait, 2 ** min(attempt, 5)))
        except Exception:
            if attempt == 7:
                raise
            time.sleep(2 ** min(attempt, 5))


v1.api = api  # revision_at / lead_text in the v1 module use this


def page_links(title, namespace):
    links, cont = [], {}
    while True:
        d = v1.api(dict({"action": "query", "prop": "links", "titles": title,
                         "plnamespace": namespace, "pllimit": "max", "redirects": 1}, **cont))
        links += [link["title"] for link in d["query"]["pages"][0].get("links", [])]
        if "continue" not in d:
            return links
        cont = d["continue"]


# ---- Wikipedia (articles and policies) ------------------------------

def wiki_versions(title):
    """Distinct lead sections of `title` on each date, oldest first."""
    seen_rev, versions = set(), []
    for date in DATES + [None]:
        rev = v1.revision_at(title, date)
        if not rev or rev["revid"] in seen_rev:
            continue
        seen_rev.add(rev["revid"])
        try:
            text = v1.lead_text(rev["revid"])
        except KeyError:  # revision hidden or unparsable
            continue
        if len(text) >= 200:
            versions.append({"revid": rev["revid"], "created_at": ts_ms(rev["timestamp"]),
                             "text": text})
    return dedupe(versions)


def dedupe(versions):
    # One document per distinct text; keep the newest revision carrying it.
    seen, kept = set(), []
    for v in reversed(versions):
        if v["text"] not in seen:
            seen.add(v["text"])
            kept.append(v)
    return kept[::-1]


def resolve_redirects(titles):
    """Map each title to its redirect target so aliases don't duplicate a page."""
    out = []
    for i in range(0, len(titles), 50):
        batch = titles[i: i + 50]
        d = v1.api({"action": "query", "titles": "|".join(batch), "redirects": 1})["query"]
        alias = {n["from"]: n["to"] for n in d.get("normalized", [])}
        redirect = {r["from"]: r["to"] for r in d.get("redirects", [])}
        for t in batch:
            t = alias.get(t, t)
            out.append(redirect.get(t, t))
    return out


def wiki_titles(corpus):
    if corpus == "wiki":
        titles = list(v1.ARTICLES)
        titles += page_links("Wikipedia:Vital articles/Level 3", 0)
        titles += page_links("Wikipedia:Vital articles/Level 4/Technology", 0)
    else:
        titles = [t for t in page_links("Wikipedia:List of policies and guidelines", 4)
                  if not POLICY_EXCLUDE.search(t)]
    titles = resolve_redirects(list(dict.fromkeys(titles)))
    return [t for t in dict.fromkeys(titles) if not (corpus == "policy" and POLICY_EXCLUDE.search(t))]


# ---- PyPI READMEs ------------------------------------------------------

_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML = re.compile(r"<[^>]+>")
_RST_DIRECTIVE = re.compile(r"^\.\. .*$(\n {2,}.*$)*", re.M)
_RST_SUBST = re.compile(r"\|[\w\- ]+\|_?")
_UNDERLINE = re.compile(r"^[=\-~^*#`+]{3,}\s*$", re.M)


def clean_readme(text, max_chars=1500):
    text = html.unescape(text or "")
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("|"))
    text = _MD_IMG.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _HTML.sub(" ", text)
    text = _RST_DIRECTIVE.sub("", text)
    text = _RST_SUBST.sub("", text)
    text = _UNDERLINE.sub("", text)
    text = re.sub(r"`{1,3}|\*\*|__|^#+\s*", "", text, flags=re.M)
    text = " ".join(text.split())
    return text[:max_chars].rsplit(" ", 1)[0]


_PRE = re.compile(r"(a|b|rc|dev|alpha|beta|pre|post\d*\.dev)\d*", re.I)


def docs_versions(pkg):
    info = get_json(f"https://pypi.org/pypi/{pkg}/json")
    releases = []
    for ver, files in info["releases"].items():
        if files and not _PRE.search(ver.split("+")[0].split(".")[-1]) and not _PRE.search(ver):
            releases.append((min(ts_ms(f["upload_time_iso_8601"]) for f in files), ver))
    releases.sort()
    picks = []
    for date in DATES:
        cutoff = calendar.timegm(time.strptime(date, "%Y-%m-%d")) * 1000
        before = [r for r in releases if r[0] <= cutoff]
        if before:
            picks.append(before[-1])
    if releases:
        picks.append(releases[-1])
    versions, seen = [], set()
    for t, ver in picks:
        if ver in seen:
            continue
        seen.add(ver)
        desc = get_json(f"https://pypi.org/pypi/{pkg}/{ver}/json")["info"].get("description")
        text = clean_readme(desc)
        if len(text) >= 200:
            versions.append({"revid": ver, "created_at": t, "text": text})
    return dedupe(versions)


# ---- main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["wiki", "policy", "docs", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    for corpus in (["wiki", "policy", "docs"] if args.corpus == "all" else [args.corpus]):
        out = HERE / "data" / f"{corpus}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            done = {json.loads(line)["title"] for line in out.open()}
        skipped_path = out.with_suffix(".skipped")
        if skipped_path.exists():
            done |= set(skipped_path.read_text().splitlines())
        titles = DOCS_PACKAGES if corpus == "docs" else wiki_titles(corpus)
        titles = [t for t in (titles[: args.limit] if args.limit else titles) if t not in done]
        fetch = docs_versions if corpus == "docs" else wiki_versions
        print(f"{corpus}: {len(titles)} titles to fetch ({len(done)} already done)")
        lock = threading.Lock()
        counter = {"n": 0}

        def work(title):
            try:
                kept = fetch(title)
            except Exception as e:  # noqa: BLE001
                print(f"  error {title}: {e}")
                return
            with lock:
                counter["n"] += 1
                if len(kept) < 2:
                    with skipped_path.open("a") as s:
                        s.write(title + "\n")
                    return
                with out.open("a") as f:
                    for i, v in enumerate(kept):
                        f.write(json.dumps({
                            "corpus": corpus, "title": title, "version": i,
                            "is_current": i == len(kept) - 1, "revid": v["revid"],
                            "created_at": v["created_at"], "text": v["text"]}) + "\n")
                if counter["n"] % 50 == 0:
                    print(f"  {corpus}: {counter['n']}/{len(titles)}")

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(work, titles))
        n = sum(1 for _ in out.open()) if out.exists() else 0
        print(f"{corpus}: {n} documents in {out}")


if __name__ == "__main__":
    main()
