"""
self_train.py — Autonomous Self-Training & Web-Evolution System for NanoLlamaPro
================================================================================
Upgrades Included:
  1. Auto-Scaling PyTorch (CUDA/MPS/CPU) & Vectorized NumPy Backend
  2. Full-Paragraph Wikipedia Crawling & Multi-Sentence Sequence Packing
  3. Larger BPE Vocab (2048-4096), Batches, & Longer Training Schedules
  4. Instruction/Chat SFT with Loss Masking on <USER> Tokens
  5. Fact-Grounded Self-Play & GRPO Verification (prevents model collapse)
  6. Live Progress Bars + ETA across every single stage
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import html as html_lib
from html.parser import HTMLParser
import json
import os
import random
import re
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import socket
import ssl
import subprocess
import threading
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("  NanoLlamaPro — Autonomous Self-Training AI", flush=True)
    print("=" * 60, flush=True)
    print("Startup      [###-----------------]  15.0% | Checking numpy...", end="\r", flush=True)

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    print("Missing required package: numpy")
    try:
        ans = input("Install numpy now with pip? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"
    if ans in ("", "y", "yes"):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        except Exception as exc:
            print(f"Failed to install numpy: {exc}")
            try:
                input("\nPress Enter to exit...")
            except (EOFError, KeyboardInterrupt):
                pass
            sys.exit(1)
    else:
        sys.exit(1)

if __name__ == "__main__":
    print("Startup      [########------------]  40.0% | Loading PyTorch / AI backend...", end="\r", flush=True)

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    print("PyTorch (torch) is not installed (currently using numpy-cpu).")
    try:
        ans = input("Install PyTorch now for faster batched/GPU training? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"
    if ans in ("", "y", "yes"):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        except Exception as exc:
            print(f"Could not install torch ({exc}). Continuing with numpy-cpu fallback...")

try:
    from nanollama_pro import HAS_TORCH, NanoLlamaUltra, get_best_device  # noqa: E402
except ModuleNotFoundError:
    from nanollama_ultra import HAS_TORCH, NanoLlamaUltra, get_best_device  # noqa: E402

if __name__ == "__main__":
    print("Startup      [####################] 100.0% | Engine ready!                  ", flush=True)

USER_AGENT = "NanoLlamaPro/4.0 (https://github.com/nanollama; educational-llm-trainer) Python-urllib/3"

DEFAULT_TOPICS = [
    "Artificial_intelligence", "Science", "History", "Biology", "Physics",
    "Chemistry", "Mathematics", "Literature", "Philosophy", "Psychology",
    "Economics", "Geography", "Earth", "Ocean", "Climate",
    "Human_body", "Medicine", "Genetics", "Evolution", "Neuroscience",
    "Astronomy", "Solar_System", "Mars", "Galaxy", "Quantum_mechanics",
    "Computer_science", "Internet", "Engineering", "Architecture", "Music",
    "Painting", "Cinema", "Democracy", "Law", "Agriculture",
    "Energy", "Electricity", "Language", "Linguistics", "World_history",
    "Ancient_Greece", "Renaissance", "Industrial_Revolution", "Spaceflight",
    "Ecology", "Botany", "Zoology", "Geology", "Volcano", "Earthquake",
    "Plate_tectonics", "Atmosphere", "Photosynthesis", "Cell_(biology)",
    "DNA", "Protein", "Neuron", "Brain", "Heart", "Immune_system",
    "Gravity", "Relativity", "Thermodynamics", "Electromagnetism", "Optics",
    "Atom", "Molecule", "Periodic_table", "Carbon", "Water",
    "Algorithm", "Cryptography", "Robotics", "Machine_learning", "Operating_system",
    "Roman_Empire", "Ancient_Egypt", "Middle_Ages", "World_War_II", "Cold_War",
    "United_States", "Europe", "Asia", "Africa", "Amazon_rainforest",
    "Antarctica", "Moon", "Sun", "Black_hole", "Supernova",
    "Star", "Exoplanet", "Telescope", "Rocket", "Satellite",
    "Symphony", "Jazz", "Poetry", "Novel", "Mythology",
    "Sociology", "Anthropology", "Archaeology", "Ethics", "Logic",
    "Statistics", "Calculus", "Geometry", "Algebra", "Number_theory",
    "Nutrition", "Sleep", "Memory", "Consciousness", "Perception",
]

DEFAULT_SUBREDDITS = [
    "explainlikeimfive", "todayilearned", "science", "space", "AskScience",
    "history", "technology", "biology", "books", "Futurology",
    "LifeProTips", "Showerthoughts", "psychology", "philosophy", "geology",
    "AskHistorians", "astronomy", "chemistry", "physics", "economics",
    "nature", "linguistics", "learnprogramming", "Cooking", "music",
    "movies", "DIY", "gardening", "mildlyinteresting", "changemyview",
]

DEFAULT_RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "https://www.sciencedaily.com/rss/all.xml",
    "https://www.sciencedaily.com/rss/space_time.xml",
    "https://www.sciencedaily.com/rss/computers_math.xml",
    "https://www.sciencedaily.com/rss/matter_energy.xml",
    "https://www.sciencedaily.com/rss/mind_brain.xml",
    "https://feeds.npr.org/1001/rss.xml",
    "https://feeds.npr.org/1007/rss.xml",
    "https://phys.org/rss-feed/",
    "https://www.quantamagazine.org/feed/",
    "https://news.mit.edu/rss/feed",
    "https://hnrss.org/frontpage",
    "https://www.smithsonianmag.com/rss/latest_articles/",
    "https://arstechnica.com/science/feed/",
    "https://arstechnica.com/information-technology/feed/",
]

ARXIV_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CL", "astro-ph", "physics.pop-ph", "q-bio", "math.HO",
]


# ---------------------------------------------------------------------------
# 1. Terminal-Safe Progress Bar
# ---------------------------------------------------------------------------
def _fmt_eta(seconds: float) -> str:
    secs = max(0, int(round(seconds)))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def print_progress(
    current: int,
    total: int,
    label: str = "",
    extra: str = "",
    start_time: float | None = None,
) -> None:
    total = max(1, total)
    pct = min(1.0, current / total)
    cols = shutil.get_terminal_size((80, 24)).columns
    max_w = max(50, cols - 1)

    bar_w = 20
    filled = int(bar_w * pct)
    bar = "#" * filled + "-" * (bar_w - filled)
    eta_str = ""
    if start_time is not None and current > 0:
        elapsed = max(1e-6, time.monotonic() - start_time)
        if current >= total:
            eta_str = f" | Done {_fmt_eta(elapsed)}"
        else:
            rem = elapsed * (total - current) / current
            eta_str = f" | ETA {_fmt_eta(rem)}"

    msg = f"{label:<12} [{bar}] {pct * 100:5.1f}% ({current}/{total}){eta_str}"
    if extra:
        msg += f" | {extra}"

    if len(msg) > max_w:
        msg = msg[: max_w - 3] + "..."
    else:
        msg = msg.ljust(max_w)

    sys.stdout.write("\r" + msg)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def make_log_cb(label: str):
    def _cb(step: int, total: int, loss: float, ema: float, tok_s: float, t0: float) -> None:
        print_progress(
            step,
            total,
            label,
            f"Loss={loss:.2f} EMA={ema:.2f} {tok_s:.0f} tok/s",
            start_time=t0,
        )
    return _cb


# ---------------------------------------------------------------------------
# 2. Full-Paragraph Web Crawlers (Wikipedia, Reddit, RSS)
# ---------------------------------------------------------------------------
class _WikiPageParser(HTMLParser):
    SKIP_PREFIXES = (
        "File:", "Help:", "Special:", "Wikipedia:", "Category:",
        "Portal:", "Talk:", "Template:", "Template_talk:", "Main_Page",
    )

    def __init__(self, paragraphs_only: bool = True):
        super().__init__()
        self._paragraphs_only = paragraphs_only
        self._skip_depth = 0
        self._in_p = False
        self._cur_p: list[str] = []
        self.paragraphs: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript", "head", "nav", "footer", "table", "sup"}:
            self._skip_depth += 1
            return
        if t == "p":
            self._in_p = True
            self._cur_p = []
        elif t == "a" and self._in_p and self._skip_depth == 0:
            for k, v in attrs:
                if k == "href" and v and v.startswith("/wiki/") and "#" not in v:
                    slug = urllib.parse.unquote(v[6:])
                    if slug and ":" not in slug and not slug.startswith(self.SKIP_PREFIXES):
                        self.links.append(slug)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript", "head", "nav", "footer", "table", "sup"}:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif t == "p":
            self._in_p = False
            txt = "".join(self._cur_p).strip()
            if txt:
                self.paragraphs.append(txt)
            self._cur_p = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._paragraphs_only and not self._in_p:
            return
        self._cur_p.append(data)


def http_get_text(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLError):
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset, errors="replace")
        raise


def clean_paragraphs(raw_paragraphs: list[str], min_words: int = 12, max_words: int = 140) -> list[str]:
    """Keeps clean multi-sentence paragraphs for richer context learning."""
    out: list[str] = []
    for p in raw_paragraphs:
        s = re.sub(r"\[[0-9a-zA-Z]+\]", "", p)
        s = re.sub(r"\s+", " ", s).strip()
        if not s or not s[0].isupper() or s[-1] not in ".!?":
            continue
        if any(bad in s for bad in ("http://", "https://", "{", "}", "ISBN", "Wikipedia", "^ ", "r/")):
            continue
        ascii_ratio = sum(32 <= ord(ch) <= 126 for ch in s) / max(1, len(s))
        if ascii_ratio < 0.95:
            continue
        alpha_ratio = sum(ch.isalpha() or ch == " " for ch in s) / max(1, len(s))
        if alpha_ratio < 0.78:
            continue
        words = re.findall(r"[A-Za-z0-9']+", s)
        if min_words <= len(words) <= max_words:
            out.append(s)
    return out


def fetch_random_wiki_titles(count: int = 20, simple: bool = False) -> list[tuple[str, str]]:
    """Uses Wikipedia's API to fetch genuinely random English (or Simple English) article titles."""
    host = "simple.wikipedia.org" if simple else "en.wikipedia.org"
    url = f"https://{host}/w/api.php?action=query&list=random&rnnamespace=0&rnlimit={min(50, count)}&format=json"
    data = json.loads(http_get_text(url))
    return [
        (host, item["title"].replace(" ", "_"))
        for item in data.get("query", {}).get("random", [])
        if item.get("title")
    ]


def scrape_arxiv_abstracts(category: str, max_results: int = 12) -> list[str]:
    url = f"http://export.arxiv.org/api/query?search_query=cat:{urllib.parse.quote(category)}&start={random.randint(0, 60)}&max_results={max_results}"
    xml_text = http_get_text(url)
    root = ET.fromstring(xml_text)
    chunks: list[str] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag == "summary" and elem.text:
            chunks.append(elem.text.strip())
    return clean_paragraphs(chunks, min_words=15, max_words=160)


def scrape_rss_feeds(feed_urls: list[str]) -> list[str]:
    paragraphs: list[str] = []
    t0 = time.monotonic()
    for idx, feed_url in enumerate(feed_urls, 1):
        try:
            xml_text = http_get_text(feed_url)
            root = ET.fromstring(xml_text)
            chunks: list[str] = []
            for elem in root.iter():
                tag = elem.tag.split("}")[-1].lower()
                if tag in {"title", "description", "summary"} and elem.text:
                    txt = html_lib.unescape(elem.text)
                    if "<" in txt and ">" in txt:
                        p = _WikiPageParser(paragraphs_only=False)
                        p.feed(txt)
                        txt = " ".join(p.paragraphs)
                    txt = txt.strip()
                    if txt:
                        chunks.append(txt if txt[-1] in ".!?" else txt + ".")
            paras = clean_paragraphs(chunks, min_words=8, max_words=120)
            paragraphs.extend(paras)
            print_progress(idx, len(feed_urls), "RSS Feeds", f"+{len(paras)} items", start_time=t0)
        except Exception:
            print_progress(idx, len(feed_urls), "RSS Feeds", "skip", start_time=t0)
    return paragraphs


def crawl_wikipedia(
    start_topic: str,
    max_pages: int = 20,
    max_paras_per_page: int = 25,
    visited_topics: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    visited = visited_topics if visited_topics is not None else set()
    queue: deque[tuple[str, str]] = deque()

    # Mix random English Wikipedia + Simple English Wikipedia articles
    for use_simple in (False, True):
        try:
            for host, t in fetch_random_wiki_titles(max_pages // 2 + 4, simple=use_simple):
                key = f"{host}:{t.lower()}"
                if key not in visited and t.lower() not in visited:
                    visited.add(key)
                    queue.append((host, t))
        except Exception:
            pass

    # Fallback if API failed: seed from unvisited DEFAULT_TOPICS + Special:Random
    if not queue:
        unvisited = [t for t in DEFAULT_TOPICS if t.lower() not in visited]
        random.shuffle(unvisited)
        for t in unvisited[:max_pages]:
            visited.add(t.lower())
            queue.append(("en.wikipedia.org", t))
        while len(queue) < max_pages:
            queue.append(("en.wikipedia.org", "Special:Random"))

    start_slug = start_topic.strip().replace(" ", "_")
    if start_slug and start_slug.lower() not in visited and start_slug != "Special:Random":
        visited.add(start_slug.lower())
        queue.appendleft(("en.wikipedia.org", start_slug))

    collected: list[str] = []
    discovered_links: list[str] = []
    pages_done = 0
    t0 = time.monotonic()

    while pages_done < max_pages:
        if not queue:
            queue.append(("en.wikipedia.org", "Special:Random"))

        host, slug = queue.popleft()
        pages_done += 1
        url = (
            f"https://{host}/wiki/Special:Random"
            if slug == "Special:Random"
            else f"https://{host}/wiki/{urllib.parse.quote(slug)}"
        )
        try:
            html = http_get_text(url)
            parser = _WikiPageParser(paragraphs_only=True)
            parser.feed(html)
            paras = clean_paragraphs(parser.paragraphs)[:max_paras_per_page]
            collected.extend(paras)

            # Follow at most 1 random link per page so every cycle spans many distinct topics
            new_links = [lnk for lnk in dict.fromkeys(parser.links) if lnk.lower() not in visited]
            random.shuffle(new_links)
            discovered_links.extend(new_links[:6])
            for lnk in new_links[:1]:
                if len(queue) + pages_done >= max_pages * 2:
                    break
                visited.add(lnk.lower())
                queue.append((host, lnk))

            print_progress(pages_done, max_pages, "Wiki Crawl", f"+{len(paras)} ({slug[:18]})", start_time=t0)
        except Exception:
            print_progress(pages_done, max_pages, "Wiki Crawl", f"skip ({slug[:18]})", start_time=t0)

    return collected, discovered_links


def scrape_reddit_subreddits(subreddits: list[str]) -> list[str]:
    paragraphs: list[str] = []
    t0 = time.monotonic()
    for idx, sub in enumerate(subreddits, 1):
        clean_sub = sub.strip().lstrip("r/").strip("/")
        try:
            url = f"https://www.reddit.com/r/{urllib.parse.quote(clean_sub)}/.rss"
            xml_text = http_get_text(url)
            root = ET.fromstring(xml_text)
            raw_list: list[str] = []
            for elem in root.iter():
                tag = elem.tag.split("}")[-1].lower()
                if tag in {"title", "content"} and elem.text:
                    txt = html_lib.unescape(elem.text)
                    if "<" in txt and ">" in txt:
                        p = _WikiPageParser(paragraphs_only=False)
                        p.feed(txt)
                        txt = " ".join(p.paragraphs)
                    txt = txt.strip()
                    if txt and "submitted by" not in txt.lower():
                        raw_list.append(txt if txt[-1] in ".!?" else txt + ".")
            paras = clean_paragraphs(raw_list, min_words=8, max_words=120)
            paragraphs.extend(paras)
            print_progress(idx, len(subreddits), "Reddit", f"+{len(paras)} (r/{clean_sub[:14]})", start_time=t0)
        except Exception:
            print_progress(idx, len(subreddits), "Reddit", f"skip (r/{clean_sub[:14]})", start_time=t0)
    return paragraphs


# ---------------------------------------------------------------------------
# 3. Deduplicated Corpus Storage
# ---------------------------------------------------------------------------
def _line_hash(line: str) -> str:
    return hashlib.sha256(line.strip().lower().encode("utf-8")).hexdigest()[:16]


def load_all_training_text(data_dir: str, existing_path: str) -> tuple[str, int, set[str]]:
    all_lines: list[str] = []
    seen_hashes: set[str] = set()
    file_count = 0

    candidates: list[str] = []
    if os.path.isfile(existing_path):
        candidates.append(existing_path)
    if os.path.isdir(data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith(".txt"):
                candidates.append(os.path.join(data_dir, fname))

    t_load = time.monotonic()
    total_files = max(1, len(candidates))
    for idx, p in enumerate(candidates, 1):
        file_count += 1
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                h = _line_hash(s)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_lines.append(s)
        print_progress(idx, total_files, "Loading", f"{len(all_lines):,} docs", start_time=t_load)

    return "\n".join(all_lines), file_count, seen_hashes


def save_batch(data_dir: str, prefix: str, lines: list[str], seen_hashes: set[str]) -> tuple[str | None, int]:
    os.makedirs(data_dir, exist_ok=True)
    unique_new: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        h = _line_hash(s)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_new.append(s)
    if not unique_new:
        return None, 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    batch_path = os.path.join(data_dir, f"{prefix}_{stamp}.txt")
    counter = 1
    while os.path.exists(batch_path):
        batch_path = os.path.join(data_dir, f"{prefix}_{stamp}_{counter}.txt")
        counter += 1

    with open(batch_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unique_new) + "\n")
    return batch_path, len(unique_new)


# ---------------------------------------------------------------------------
# 4. Fact-Grounded Self-Play, SFT Q&A Pairs, & GRPO Verification
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of",
    "and", "or", "in", "on", "at", "for", "with", "as", "by", "that", "this", "it",
}


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 3 and w not in _STOPWORDS}


def extract_grounded_prompts(docs: list[str], count: int = 12) -> list[tuple[str, str]]:
    """Returns (prompt_prefix, full_source_fact) pairs for fact-grounded self-play."""
    if not docs:
        return [("The universe is", "The universe is vast and contains billions of galaxies.")]
    sample = random.sample(docs, min(len(docs), count * 3))
    pairs: list[tuple[str, str]] = []
    for d in sample:
        words = d.split()
        if len(words) >= 10:
            k = random.randint(3, min(6, len(words) - 5))
            pairs.append((" ".join(words[:k]), d))
        if len(pairs) >= count:
            break
    return pairs or [("The universe is", "The universe is vast and contains billions of galaxies.")]


CORE_CHAT_PAIRS: list[tuple[str, str]] = [
    ("Hi", "Hello! How can I help you today?"),
    ("Hello", "Hi there! What would you like to talk about?"),
    ("Hey", "Hey! Ask me anything you are curious about."),
    ("How are you?", "I am doing well and ready to help! What is on your mind?"),
    ("Who are you?", "I am NanoLlamaPro, a custom Transformer language model trained from scratch."),
    ("What is your name?", "My name is NanoLlamaPro."),
    ("What can you do?", "I can answer questions, explain topics from science and history, and chat with you."),
    ("Thanks", "You are welcome! Let me know if you need anything else."),
    ("Thank you", "Happy to help!"),
    ("Good morning", "Good morning! How can I help you today?"),
    ("Good night", "Good night! Sleep well."),
    ("Help me", "Sure! Tell me what topic or question you need help with."),
    ("Tell me a fun fact", "Mars has the largest volcano in the solar system, called Olympus Mons."),
    ("What is artificial intelligence?", "Artificial intelligence is the science of building machines that can learn, reason, and solve problems."),
    ("What is a planet?", "A planet is a large astronomical object that orbits a star and has cleared its orbital path."),
    ("Why is the sky blue?", "Earth's atmosphere scatters shorter blue wavelengths of sunlight more than longer red wavelengths."),
    ("What is gravity?", "Gravity is the fundamental force that attracts objects with mass toward one another."),
    ("What is evolution?", "Evolution is the process by which living organisms change over generations through natural selection."),
    ("What is DNA?", "DNA is the molecule that carries the genetic instructions for living organisms."),
    ("What is water made of?", "Water is a molecule made of two hydrogen atoms bonded to one oxygen atom."),
    ("hi", "Hello! How can I help you today?"),
    ("hello", "Hi there! What would you like to talk about?"),
    ("hello!", "Hey! How can I help you?"),
    ("hey", "Hey! Ask me anything you are curious about."),
    ("how are you", "I am doing well and ready to help! What is on your mind?"),
    ("who are you", "I am NanoLlamaPro, a custom Transformer language model trained from scratch."),
    ("what is your name", "My name is NanoLlamaPro."),
    ("what can you do", "I can answer questions, explain topics from science and history, and chat with you."),
    ("thanks", "You are welcome! Let me know if you need anything else."),
    ("what", "Could you tell me what topic or question you would like me to explain?"),
]


MULTI_TURN_CHATS: list[list[tuple[str, str]]] = [
    [("hi", "Hello! How can I help you today?"), ("who are you?", "I am NanoLlamaPro, a custom Transformer language model.")],
    [("hello", "Hi there! What would you like to talk about?"), ("what can you do?", "I can answer questions about science, history, space, and chat with you.")],
    [("how are you?", "I am doing well and ready to help!"), ("tell me a fun fact", "Mars has the largest volcano in the solar system, called Olympus Mons.")],
    [("what is gravity?", "Gravity is the fundamental force that attracts objects with mass toward one another."), ("thanks", "You are welcome! Let me know if you have more questions.")],
]


def build_sft_qa_pairs(docs: list[str], count: int = 40) -> list:
    """Builds conversational, multi-turn, and <THINK>-grounded SFT examples."""
    qa: list = list(CORE_CHAT_PAIRS) + list(MULTI_TURN_CHATS)
    if not docs:
        return qa
    sample = random.sample(docs, min(len(docs), count))
    templates = [
        "What is {topic}?",
        "Tell me about {topic}.",
        "Can you explain {topic}?",
        "What do you know about {topic}?",
    ]
    for d in sample:
        m = re.match(r"^([A-Z][A-Za-z0-9'\- ]{2,35}?)\s+(?:is|are|was|were)\b", d)
        if m:
            topic = m.group(1).strip()
        else:
            words = d.split()
            if len(words) < 8:
                continue
            topic = " ".join(words[: min(4, len(words) // 2)]).rstrip(".,;:")
        q = random.choice(templates).format(topic=topic)
        kws = sorted(list(_keywords(d) - _keywords(topic)))[:4]
        if kws:
            think = f"Topic: {topic}. Key facts: {', '.join(kws)}."
            qa.append((q, think, d))
        else:
            qa.append((q, d))
    return qa


def grounded_quality_reward(
    model: NanoLlamaUltra,
    prompt: str,
    completion: str,
    gen_ids: list[int],
    source_fact: str | None = None,
) -> float:
    """
    Fact-Grounded Critic:
      - Penalizes short/repetitive gibberish
      - Rewards lexical & bigram diversity
      - Rewards factual keyword overlap with source_fact (prevents model collapse)
      - Rewards stable perplexity
    """
    words = re.findall(r"[A-Za-z0-9']+", completion)
    if len(words) < 5 or len(gen_ids) < 5:
        return -3.0

    unique_ratio = len(set(gen_ids)) / max(1, len(gen_ids))
    bigrams = list(zip(gen_ids, gen_ids[1:]))
    bigram_div = len(set(bigrams)) / max(1, len(bigrams))
    punct_bonus = 0.5 if completion.strip().endswith((".", "!", "?")) else -0.3
    len_score = min(1.0, len(gen_ids) / 20.0)

    fact_bonus = 0.0
    if source_fact:
        src_kw = _keywords(source_fact) - _keywords(prompt)
        gen_kw = _keywords(completion)
        if src_kw:
            overlap = len(src_kw & gen_kw) / min(6, len(src_kw))
            fact_bonus = 2.4 * min(1.0, overlap)

    ce, _ = model.score_text_perplexity((prompt + " " + completion).strip())
    ppl_score = -0.30 * max(0.0, ce - 2.3)

    return float(1.5 * unique_ratio + 1.4 * bigram_div + 0.7 * len_score + punct_bonus + fact_bonus + ppl_score)


def run_grounded_self_play(
    model: NanoLlamaUltra,
    num_prompts: int = 12,
    candidates_per_prompt: int = 3,
    min_reward: float = 2.4,
) -> list[str]:
    pairs = extract_grounded_prompts(model.docs, count=num_prompts)
    accepted: list[str] = []
    t0 = time.monotonic()

    for idx, (prompt, src_fact) in enumerate(pairs, 1):
        best_text = ""
        best_rew = -1e9
        for _ in range(candidates_per_prompt):
            comp = model.generate(prompt, max_tokens=44, temperature=0.65, top_k=24, min_p=0.06)
            gen_ids = model.tokenizer.encode(comp)
            rew = grounded_quality_reward(model, prompt, comp, gen_ids, source_fact=src_fact)
            if rew > best_rew:
                best_rew = rew
                best_text = (prompt + " " + comp).strip()
        if best_rew >= min_reward and len(best_text.split()) >= 8:
            if not best_text.endswith((".", "!", "?")):
                best_text += "."
            accepted.append(best_text)
        print_progress(idx, len(pairs), "Self-Play", f"verified={len(accepted)} bestR={best_rew:.2f}", start_time=t0)

    return accepted


# ---------------------------------------------------------------------------
# 5. Autonomous Self-Training Cycle
# ---------------------------------------------------------------------------
def run_self_training_cycle(
    model: NanoLlamaUltra,
    cycle_idx: int,
    data_dir: str,
    seen_hashes: set[str],
    visited_topics: set[str],
    topic_queue: deque[str],
    wiki_pages: int = 20,
    use_reddit: bool = False,
    train_steps: int = 500,
    sft_steps: int = 80,
    batch_size: int = 8,
    lr: float = 0.005,
    grpo_rounds: int = 3,
    save_path: str = "nanollama_pro.npz",
) -> None:
    print(f"\n=== Self-Training Cycle {cycle_idx} (Backend: {model.device}) ===")
    new_lines: list[str] = []

    # 1. Web Discovery (Full Paragraphs)
    if wiki_pages > 0:
        unvisited_pool = [t for t in DEFAULT_TOPICS if t.lower() not in visited_topics]
        if unvisited_pool:
            topic = random.choice(unvisited_pool)
        elif topic_queue:
            topic = topic_queue.popleft()
        else:
            topic = "Special:Random"
        print(f"Crawling Wikipedia from '{topic}' ({wiki_pages} pages)...")
        try:
            paras, links = crawl_wikipedia(topic, max_pages=wiki_pages, visited_topics=visited_topics)
            new_lines.extend(paras)
            for lnk in links:
                if lnk.lower() not in visited_topics:
                    topic_queue.append(lnk)
        except Exception as exc:
            print(f"  ! Web crawl skipped ({exc})")

    if wiki_pages > 0:
        subs_pick = random.sample(DEFAULT_SUBREDDITS, min(3, len(DEFAULT_SUBREDDITS)))
        new_lines.extend(scrape_reddit_subreddits(subs_pick))
        rss_pick = random.sample(DEFAULT_RSS_FEEDS, min(3, len(DEFAULT_RSS_FEEDS)))
        new_lines.extend(scrape_rss_feeds(rss_pick))
        try:
            arx_cat = random.choice(ARXIV_CATEGORIES)
            arx_paras = scrape_arxiv_abstracts(arx_cat, max_results=10)
            new_lines.extend(arx_paras)
        except Exception:
            pass
    elif use_reddit:
        new_lines.extend(scrape_reddit_subreddits(DEFAULT_SUBREDDITS[:3]))

    if new_lines:
        b_path, added = save_batch(data_dir, "wiki", new_lines, seen_hashes)
        if b_path:
            print(f"Ingested {added} new unique paragraphs -> {os.path.basename(b_path)}")

    last_loss = model.metrics_history[-1]["ema_loss"] if model.metrics_history else 99.0

    # 2. Fact-Grounded Self-Play Distillation (gated until basic English fluency)
    if model.step_count > 0 and last_loss < 4.0:
        synth = run_grounded_self_play(model, num_prompts=12, candidates_per_prompt=3)
        if synth:
            s_path, s_added = save_batch(data_dir, "selfplay", synth, seen_hashes)
            if s_path:
                print(f"Distilled {s_added} fact-verified examples -> {os.path.basename(s_path)}")
            new_lines.extend(synth)

    # 3. Experience Replay Pre-Training (Paragraph Packed)
    extra_seqs = None
    if new_lines:
        model.docs.extend(new_lines)
        t_pack = time.monotonic()
        extra_seqs = model.encode_docs(
            new_lines,
            pack=True,
            progress_cb=lambda c, t, e: print_progress(c, t, "Packing", e, start_time=t_pack),
        )

    ema_loss = model.train_steps(
        num_steps=train_steps,
        max_lr=lr,
        batch_size=batch_size,
        extra_seqs=extra_seqs,
        extra_ratio=0.40 if extra_seqs else 0.0,
        log_fn=make_log_cb("Training"),
        log_every=max(1, train_steps // 20),
        save_every=200,
        save_path=save_path,
    )

    # 4. Fact-Grounded GRPO RL (gated until basic English fluency)
    mean_rew = 0.0
    if grpo_rounds > 0 and ema_loss < 4.0:
        t0 = time.monotonic()
        for r in range(grpo_rounds):
            pairs = extract_grounded_prompts(model.docs, count=4)
            prompt_to_fact = dict(pairs)
            stats = model.grpo_update(
                prompts=[p for p, _ in pairs],
                reward_fn=lambda p, txt, ids: grounded_quality_reward(
                    model, p, txt, ids, source_fact=prompt_to_fact.get(p)
                ),
                group_size=4,
                lr=lr * 0.25,
            )
            mean_rew = stats["mean_reward"]
            print_progress(r + 1, grpo_rounds, "GRPO RL", f"Reward={mean_rew:.2f}", start_time=t0)

    # 5. Instruction/Chat SFT AFTER GRPO (Loss Masked on <USER> Tokens)
    if sft_steps > 0 and model.docs:
        qa_pairs = build_sft_qa_pairs(model.docs, count=30) + list(CORE_CHAT_PAIRS)
        model.train_sft_steps(
            qa_pairs=qa_pairs,
            num_steps=sft_steps,
            lr=lr * 0.45,
            log_fn=make_log_cb("SFT Chat"),
        )

    # 6. Validation & Save
    t_eval = time.monotonic()
    val_loss = model.eval_val_loss(
        progress_cb=lambda c, t, e: print_progress(c, t, "Val Eval", e, start_time=t_eval)
    )
    val_str = f" | Val Loss={val_loss:.4f}" if val_loss is not None else ""
    model.metrics_history.append({
        "cycle": cycle_idx,
        "step": model.step_count,
        "ema_loss": round(ema_loss, 4),
        "val_loss": round(val_loss, 4) if val_loss is not None else None,
        "grpo_reward": round(mean_rew, 4),
        "docs": len(model.docs),
    })

    t_save = time.monotonic()
    print_progress(0, 1, "Saving", os.path.basename(save_path), start_time=t_save)
    model.save(save_path)
    print_progress(1, 1, "Saving", os.path.basename(save_path), start_time=t_save)

    if visited_topics:
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "visited_topics.txt"), "w", encoding="utf-8") as vf:
            vf.write("\n".join(sorted(visited_topics)) + "\n")

    print(
        f"Cycle {cycle_idx} Complete | Steps={model.step_count} | "
        f"EMA Loss={ema_loss:.4f}{val_str} | GRPO Reward={mean_rew:.2f} | Saved -> {save_path}"
    )


# ---------------------------------------------------------------------------
# 5b. LAN Distributed Training (Host FedAvg Server + Auto-Discover Worker)
# ---------------------------------------------------------------------------
CLUSTER_PORT = 8765
BEACON_PORT = 8766


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _probe_host_port(ip: str, port: int = CLUSTER_PORT, timeout: float = 0.15) -> str | None:
    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            s.sendall(b"GET /status HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n")
            resp = s.recv(512).decode("utf-8", errors="ignore")
            if '"step"' in resp and '"merges"' in resp:
                return f"{ip}:{port}"
    except Exception:
        return None
    return None


def _discover_lan_host(timeout: float = 2.5) -> str | None:
    """Auto-discovers a NanoLlamaPro host via UDP beacon + fast /24 LAN TCP scan."""
    t_scan = time.monotonic()
    print_progress(1, 3, "LAN Scan", "listening for host beacon...", start_time=t_scan)
    # 1. Fast UDP beacon check
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)
    try:
        sock.bind(("", BEACON_PORT))
        data, addr = sock.recvfrom(1024)
        msg = data.decode("utf-8", errors="ignore")
        if msg.startswith("NANOLLAMA_HOST:"):
            port = msg.split(":", 1)[1].strip()
            return f"{addr[0]}:{port}"
    except Exception:
        pass
    finally:
        sock.close()

    print_progress(2, 3, "LAN Scan", "scanning local subnet...", start_time=t_scan)
    # 2. Fast parallel /24 subnet TCP scan (works even if UDP broadcast is blocked)
    local_ip = _get_local_ip()
    if local_ip == "127.0.0.1" or "." not in local_ip:
        return None
    prefix = ".".join(local_ip.split(".")[:3])
    found: list[str] = []
    lock = threading.Lock()

    def _scan_range(ips: list[str]):
        for ip in ips:
            if found:
                return
            hit = _probe_host_port(ip, CLUSTER_PORT, timeout=0.12)
            if hit:
                with lock:
                    found.append(hit)
                return

    targets = [f"{prefix}.{i}" for i in range(1, 255) if f"{prefix}.{i}" != local_ip]
    threads = []
    chunk = 16
    for i in range(0, len(targets), chunk):
        t = threading.Thread(target=_scan_range, args=(targets[i:i + chunk],), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=1.5)
    res = found[0] if found else None
    print_progress(3, 3, "LAN Scan", f"found {res}" if res else "no host found", start_time=t_scan)
    return res


def start_cluster_host(model: NanoLlamaUltra, save_path: str, lock: threading.Lock, port: int = CLUSTER_PORT):
    """Starts an HTTP FedAvg parameter server + UDP LAN discovery beacon."""
    state = {"workers_seen": set(), "merges": 0}

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path == "/checkpoint":
                with lock:
                    if not os.path.isfile(save_path):
                        model.save(save_path)
                    with open(save_path, "rb") as f:
                        payload = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/status":
                body = json.dumps({
                    "step": model.step_count,
                    "workers": len(state["workers_seen"]),
                    "merges": state["merges"],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/submit_weights":
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                worker_ip = self.client_address[0]
                state["workers_seen"].add(worker_ip)
                try:
                    w_data = numpy.load(io.BytesIO(raw), allow_pickle=False)
                    with lock:
                        if model.use_torch and model._pt_model is not None:
                            model._sync_torch_to_numpy()
                        alpha = 0.35
                        merged_keys = 0
                        for key in w_data.files:
                            if key.startswith("w::"):
                                k = key[3:]
                                if k in model.weights and model.weights[k].shape == w_data[key].shape:
                                    model.weights[k] = (1.0 - alpha) * model.weights[k] + alpha * w_data[key].astype(numpy.float32)
                                    merged_keys += 1
                        if model.use_torch and model._pt_model is not None:
                            model._sync_numpy_to_torch()
                        model.save(save_path)
                        state["merges"] += 1
                    resp = json.dumps({"status": "merged", "keys": merged_keys, "merges": state["merges"]}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(resp)
                    print(f"\n[Cluster Host] Merged weights from {worker_ip} (total merges={state['merges']})")
                except Exception as exc:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(str(exc).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    stop_beacon = threading.Event()

    def _beacon_loop():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        msg = f"NANOLLAMA_HOST:{port}".encode("utf-8")
        while not stop_beacon.is_set():
            try:
                sock.sendto(msg, ("255.255.255.255", BEACON_PORT))
            except Exception:
                pass
            stop_beacon.wait(2.0)
        sock.close()

    threading.Thread(target=_beacon_loop, daemon=True).start()
    local_ip = _get_local_ip()
    print(f"[Cluster Host] Ready! Other PCs can join at: {local_ip}:{port} (UDP auto-discovery active)")
    return server, stop_beacon


def run_cluster_worker(host_addr: str, args, data_dir: str, seen_hashes: set[str], visited_topics: set[str], topic_queue: deque[str]) -> None:
    base_url = host_addr if host_addr.startswith("http") else f"http://{host_addr}"
    local_ckpt = os.path.join(SCRIPT_DIR, "_worker_sync.npz")
    print(f"[Cluster Worker] Connecting to host {base_url}...")

    cycle = 0
    while args.forever or cycle < max(1, args.cycles):
        cycle += 1
        # 1. Pull latest global checkpoint from host
        req = urllib.request.Request(f"{base_url}/checkpoint", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(local_ckpt, "wb") as f:
                f.write(resp.read())

        model = NanoLlamaUltra.from_checkpoint(local_ckpt, device=get_best_device())
        combined_text, _, _ = load_all_training_text(data_dir, os.path.join(SCRIPT_DIR, args.data))
        if combined_text.strip():
            model.load_text(combined_text, fit_new=False)

        # 2. Train 1 local cycle
        run_self_training_cycle(
            model=model,
            cycle_idx=cycle,
            data_dir=data_dir,
            seen_hashes=seen_hashes,
            visited_topics=visited_topics,
            topic_queue=topic_queue,
            wiki_pages=args.pages,
            use_reddit=args.reddit,
            train_steps=args.steps,
            sft_steps=args.sft_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            grpo_rounds=args.grpo,
            save_path=local_ckpt,
        )

        # 3. Push trained weights back to host for FedAvg merge
        with open(local_ckpt, "rb") as f:
            payload = f.read()
        up_req = urllib.request.Request(
            f"{base_url}/submit_weights",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/octet-stream", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(up_req, timeout=30) as up_resp:
            info = json.loads(up_resp.read().decode("utf-8"))
            print(f"[Cluster Worker] Synced weights to host -> {info}")


# ---------------------------------------------------------------------------
# 6. Main CLI & Interactive Loop
# ---------------------------------------------------------------------------
def run() -> None:
    device = get_best_device()
    is_gpu = device in ("cuda", "mps")

    parser = argparse.ArgumentParser(description="NanoLlamaPro Autonomous Self-Training LLM")
    parser.add_argument("--cycles", type=int, default=1, help="Number of autonomous self-training cycles.")
    parser.add_argument("--forever", action="store_true", help="Run self-training continuously until Ctrl+C.")
    parser.add_argument("--steps", type=int, default=800 if is_gpu else 400, help="Pre-training steps per cycle.")
    parser.add_argument("--sft-steps", type=int, default=160, help="Loss-masked SFT steps per cycle.")
    parser.add_argument("--batch-size", type=int, default=16 if is_gpu else 8, help="Batch size.")
    parser.add_argument("--pages", type=int, default=20, help="Wikipedia pages to crawl per cycle (0=offline).")
    parser.add_argument("--reddit", action="store_true", help="Also scrape Reddit subreddits.")
    parser.add_argument("--grpo", type=int, default=3, help="GRPO RL update rounds per cycle.")
    parser.add_argument("--topic", default="Artificial_intelligence", help="Starting Wikipedia topic.")
    parser.add_argument("--dim", type=int, default=256 if is_gpu else 128, help="Embedding dimension.")
    parser.add_argument("--layers", type=int, default=6 if is_gpu else 4, help="Transformer layers.")
    parser.add_argument("--heads", type=int, default=8, help="Query attention heads.")
    parser.add_argument("--kv-heads", type=int, default=2, help="GQA KV heads.")
    parser.add_argument("--block-size", type=int, default=192 if is_gpu else 128, help="Context window length.")
    parser.add_argument("--vocab", type=int, default=3072 if is_gpu else 2048, help="Target BPE vocabulary size.")
    parser.add_argument("--lr", type=float, default=0.004 if HAS_TORCH else 0.006, help="Peak learning rate.")
    parser.add_argument("--data", default="training_data.txt", help="Seed training text file.")
    parser.add_argument("--save", default="nanollama_pro.npz", help="Checkpoint file path (.npz).")
    parser.add_argument("--resume", default=None, help="Existing .npz checkpoint to resume.")
    parser.add_argument("--no-chat", action="store_true", help="Exit after training cycles.")
    parser.add_argument("--host", action="store_true", help="Host a LAN distributed training cluster.")
    parser.add_argument("--join", default=None, help="Join a LAN cluster host (e.g. 192.168.1.42:8765 or 'auto').")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        print(f"NanoLlamaPro — Autonomous Self-Training LLM (Backend: {device})")
        print("  [1] Continuous Self-Training (--forever)")
        print("  [2] Train 2 Cycles & Chat")
        print("  [3] Chat Only (skip training)")
        print("  [4] Host Multi-PC Cluster (--host --forever)")
        print("  [5] Join Multi-PC Cluster (--join)")
        try:
            mode = input("Choose mode [1-5, default=2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            mode = "2"
        if mode == "1":
            args.forever = True
            print("Scanning local network for another NanoLlamaPro PC...")
            found = _discover_lan_host(timeout=2.5)
            if found:
                print(f"[Auto-Cluster] Found running PC at {found}! Joining to train together...")
                args.join = found
            else:
                print("[Auto-Cluster] No other PC found yet — starting as Host so others can auto-join!")
                args.host = True
        elif mode == "3":
            args.cycles = 0
        elif mode == "4":
            args.host = True
            args.forever = True
        elif mode == "5":
            args.forever = True
            print("Searching local Wi-Fi/LAN for a host (4s)...")
            found = _discover_lan_host(timeout=4.0)
            if found:
                print(f"Found host at {found}!")
                args.join = found
            else:
                try:
                    manual = input("Enter Host IP:Port (e.g. 192.168.1.42:8765): ").strip()
                except (EOFError, KeyboardInterrupt):
                    manual = ""
                args.join = manual or "127.0.0.1:8765"
        else:
            args.cycles = 2

    data_path = args.data if os.path.isabs(args.data) else os.path.join(SCRIPT_DIR, args.data)
    data_dir = os.path.join(SCRIPT_DIR, "web_training_data")
    save_path = args.save if os.path.isabs(args.save) else os.path.join(SCRIPT_DIR, args.save)

    combined_text, file_count, seen_hashes = load_all_training_text(data_dir, data_path)
    if not combined_text.strip() and args.pages <= 0:
        raise RuntimeError("No local training data found and --pages 0 specified.")

    resume_path = args.resume
    if not resume_path and os.path.isfile(save_path) and len(sys.argv) == 1:
        resume_path = save_path

    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming checkpoint: {resume_path} ({device})")
        model = NanoLlamaUltra.from_checkpoint(resume_path, device=device)
        if combined_text.strip():
            t_pack = time.monotonic()
            model.load_text(
                combined_text,
                fit_new=False,
                seq_progress_cb=lambda c, t, e: print_progress(c, t, "Packing", e, start_time=t_pack),
            )
    else:
        print(
            f"Initializing NanoLlamaPro ({args.dim}-dim, {args.layers}L, "
            f"{args.heads}Q/{args.kv_heads}KV GQA, ctx={args.block_size}, vocab={args.vocab}, backend={device})..."
        )
        model = NanoLlamaUltra(
            n_embd=args.dim,
            n_head=args.heads,
            n_kv_head=args.kv_heads,
            n_layer=args.layers,
            block_size=args.block_size,
            vocab_size=args.vocab,
            device=device,
        )
        if combined_text.strip():
            t_bpe = time.monotonic()
            t_pack = [None]

            def _bpe_cb(c, t, e):
                print_progress(c, t, "BPE Vocab", e, start_time=t_bpe)

            def _pack_cb(c, t, e):
                if t_pack[0] is None:
                    t_pack[0] = time.monotonic()
                print_progress(c, t, "Packing", e, start_time=t_pack[0])

            model.load_text(combined_text, fit_new=True, bpe_progress_cb=_bpe_cb, seq_progress_cb=_pack_cb)
            print(f"Loaded {len(model.docs):,} docs from {file_count} file(s).")

    print(f"Model Parameters: {model.param_count():,} | Vocab: {model.tokenizer.vocab_size} | Seqs: {len(model.train_seqs):,}")

    visited_file = os.path.join(data_dir, "visited_topics.txt")
    visited_topics: set[str] = set()
    if os.path.isfile(visited_file):
        with open(visited_file, "r", encoding="utf-8", errors="replace") as vf:
            visited_topics = {ln.strip().lower() for ln in vf if ln.strip()}
    shuffled_defaults = list(DEFAULT_TOPICS)
    random.shuffle(shuffled_defaults)
    topic_queue: deque[str] = deque([args.topic] + shuffled_defaults)
    if args.join:
        host_target = _discover_lan_host(timeout=4.0) if args.join == "auto" else args.join
        if not host_target:
            raise RuntimeError("Could not auto-discover a LAN host. Pass --join <IP:8765>.")
        run_cluster_worker(host_target, args, data_dir, seen_hashes, visited_topics, topic_queue)
        return

    cluster_lock = threading.Lock()
    cluster_srv = None
    cluster_stop = None
    if args.host:
        cluster_srv, cluster_stop = start_cluster_host(model, save_path, cluster_lock)

    cycle = 0

    try:
        while args.forever or cycle < args.cycles:
            cycle += 1
            run_self_training_cycle(
                model=model,
                cycle_idx=cycle,
                data_dir=data_dir,
                seen_hashes=seen_hashes,
                visited_topics=visited_topics,
                topic_queue=topic_queue,
                wiki_pages=args.pages,
                use_reddit=args.reddit,
                train_steps=args.steps,
                sft_steps=args.sft_steps,
                batch_size=args.batch_size,
                lr=args.lr,
                grpo_rounds=args.grpo,
                save_path=save_path,
            )
    except KeyboardInterrupt:
        print("\nSelf-training interrupted by user. Saving checkpoint...")
        with cluster_lock:
            model.save(save_path)
    finally:
        if cluster_stop:
            cluster_stop.set()
        if cluster_srv:
            cluster_srv.shutdown()

    if args.no_chat:
        return

    print("\nChatbot Mode — Commands: /clear, /think, /raw <text>, /selftrain, /grpo, /crawl <topic>, /stats, /save, /quit\n")
    chat_history: list[tuple[str, str]] = []
    show_thinking = False
    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not prompt:
            continue
        low = prompt.lower()
        if low in {"/quit", "quit", "exit"}:
            print("Goodbye!")
            break
        if low == "/clear":
            chat_history.clear()
            print("Conversation memory cleared.")
            continue
        if low == "/think":
            show_thinking = not show_thinking
            print(f"Show <THINK> reasoning: {'ON' if show_thinking else 'OFF'}")
            continue
        if low == "/stats":
            val = model.eval_val_loss()
            print(
                f"Backend={model.device} | Params={model.param_count():,} | Steps={model.step_count} | "
                f"Docs={len(model.docs):,} | Vocab={model.tokenizer.vocab_size} | "
                f"ValLoss={f'{val:.4f}' if val is not None else 'N/A'}"
            )
            continue
        if low == "/save":
            model.save(save_path)
            print(f"Saved checkpoint to {save_path}")
            continue
        if low == "/selftrain":
            cycle += 1
            run_self_training_cycle(
                model=model,
                cycle_idx=cycle,
                data_dir=data_dir,
                seen_hashes=seen_hashes,
                visited_topics=visited_topics,
                topic_queue=topic_queue,
                wiki_pages=10,
                use_reddit=False,
                train_steps=200,
                sft_steps=40,
                batch_size=args.batch_size,
                lr=args.lr,
                grpo_rounds=2,
                save_path=save_path,
            )
            continue
        if low == "/grpo":
            pairs = extract_grounded_prompts(model.docs, count=6)
            p_map = dict(pairs)
            stats = model.grpo_update(
                prompts=[p for p, _ in pairs],
                reward_fn=lambda p, txt, ids: grounded_quality_reward(model, p, txt, ids, source_fact=p_map.get(p)),
                group_size=4,
                lr=args.lr * 0.25,
            )
            print(f"GRPO step done | Mean Reward={stats['mean_reward']:.2f} | Best: {stats['best_text']}")
            continue
        if low.startswith("/crawl "):
            topic = prompt[7:].strip().replace(" ", "_")
            try:
                paras, _ = crawl_wikipedia(topic, max_pages=10, visited_topics=visited_topics)
                if paras:
                    _, added = save_batch(data_dir, "wiki", paras, seen_hashes)
                    model.docs.extend(paras)
                    extra = model.encode_docs(paras, pack=True)
                    model.train_steps(
                        num_steps=120,
                        max_lr=args.lr,
                        batch_size=args.batch_size,
                        extra_seqs=extra,
                        log_fn=make_log_cb("Training"),
                        log_every=10,
                    )
                    print(f"Crawled & trained on {added} new paragraphs from '{topic}'.")
            except Exception as exc:
                print(f"Crawl failed: {exc}")
            continue

        if low.startswith("/raw "):
            raw_p = prompt[5:].strip()
            print(raw_p, end="", flush=True)
            for chunk in model.generate_stream(raw_p, max_tokens=64, temperature=0.42, top_k=12, min_p=0.10, rep_penalty=1.35, chat_mode=False):
                print(chunk, end="", flush=True)
            print("\n")
            continue

        print("AI> ", end="", flush=True)
        reply_chunks: list[str] = []
        for chunk in model.generate_stream(
            prompt,
            max_tokens=72,
            temperature=0.38,
            top_k=10,
            min_p=0.10,
            rep_penalty=1.35,
            chat_mode=True,
            history=chat_history,
            show_thinking=show_thinking,
        ):
            print(chunk, end="", flush=True)
            reply_chunks.append(chunk)
        print("\n")
        chat_history.append((prompt, "".join(reply_chunks).strip()))
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]


def main() -> None:
    try:
        run()
    except SystemExit:
        raise
    except Exception:
        print("\n--- ERROR ---")
        traceback.print_exc()
        try:
            input("\nPress Enter to exit...")
        except (EOFError, KeyboardInterrupt):
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()