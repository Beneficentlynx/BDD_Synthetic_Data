"""
================================================================================
INTERACTIEF STREAMLIT PROTOTYPE: GOOGLE BOOKS NGRAM VIEWER (VISUAL BUILDER 2.0)
================================================================================
Bestand : app.py
Stijl   : Authentieke Google Books Ngram Viewer look-and-feel (Material Design),
          verrijkt met 6 proof-of-concept verbeteringen:
          1. Korte info popup bij de instellingen
          2. Algemene info / kennisbank opgedeeld in concrete secties
          3. Uitgebreide tabel met bijzonderheden (pieken, dalen, startjaar, trends)
          4. Flexibiliteit in grafiekvormgeving (vlak/lijn/staaf/trap, paletten,
             log-schaal, ruwe achtergronddata) én snelle tijdselectie-presets
          5. Cross-Language support (meertalig zoeken en vergelijken van concepten)
          6. Gebruikersaccounts voor zoekgeschiedenis, favorieten en standaardvoorkeuren
          + Integratie met AWS S3 Open Data & Live Google Ngram Database
================================================================================
"""

import difflib
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# PAGINA-CONFIGURATIE
# ==============================================================================
st.set_page_config(
    page_title="Google Books Ngram Viewer — Visual Builder 2.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_FILE = Path(__file__).parent / "synthetic_ngram_data.csv"
USER_PROFILES_FILE = Path(__file__).parent / "user_profiles.json"

# Authentieke Google Ngram Viewer lijnkleuren & paletten
GOOGLE_LINE_COLORS = ["#1a73e8", "#d93025", "#188038", "#e37400", "#9334e6", "#00897b"]
PALETTES_MAP = {
    "Google Kleuren": GOOGLE_LINE_COLORS,
    "tableau10": "tableau10",
    "category10": "category10",
    "viridis": "viridis",
    "magma": "magma",
    "dark2": "dark2",
}

# Mapping van UI-regio's naar zowel AWS S3 mapnamen als Google Books Corpus IDs
CORPUS_MAPPING: Dict[str, Dict[str, Any]] = {
    "American English": {
        "s3_prefix": "ngrams/books/20090715/eng-us-all/",
        "s3_uri": "s3://datasets.elasticmapreduce/ngrams/books/20090715/eng-us-all/",
        "google_corpus_id": "en-US-2019",
        "description": "Amerikaans-Engelse boeken (eng-us-all op AWS S3)",
    },
    "British English": {
        "s3_prefix": "ngrams/books/20090715/eng-gb-all/",
        "s3_uri": "s3://datasets.elasticmapreduce/ngrams/books/20090715/eng-gb-all/",
        "google_corpus_id": "en-GB-2019",
        "description": "Brits-Engelse boeken (eng-gb-all op AWS S3)",
    },
    "European Fiction": {
        "s3_prefix": "ngrams/books/20090715/eng-fiction-all/",
        "s3_uri": "s3://datasets.elasticmapreduce/ngrams/books/20090715/eng-fiction-all/",
        "google_corpus_id": "en-fiction-2019",
        "description": "Europese & Engelstalige fictie (eng-fiction-all op AWS S3)",
    },
    "Dutch": {
        "s3_prefix": "ngrams/books/20090715/ger-all/",
        "s3_uri": "s3://datasets.elasticmapreduce/ngrams/books/20090715/",
        "google_corpus_id": "en-2019",
        "description": "Nederlandstalig / Algemeen Europees corpus",
    },
}

# Historische gebeurtenissen lookup
HISTORICAL_EVENTS_LOOKUP: Dict[int, str] = {
    1844: "1844: Eerste demonstratie elektrische telegraaf (Morse)",
    1861: "1861: Uitbraak Amerikaanse Burgeroorlog",
    1914: "1914: Start WOI",
    1918: "1918: Einde WOI & Verdrag van Versailles",
    1929: "1929: Beurskrach & Grote Depressie",
    1939: "1939: Start WOII",
    1945: "1945: Einde WOII & Oprichting Verenigde Naties",
    1956: "1956: Dartmouth Workshop (Geboorte term 'Artificial Intelligence')",
    1989: "1989: Val van de Berlijnse Muur",
    1995: "1995: Opkomst Commercieel Internet",
    2007: "2007: Introductie Smartphone & Mobiel Web",
    2015: "2015: Klimaatakkoord van Parijs",
    2020: "2020: Wereldwijde COVID-19 Pandemie",
    2023: "2023: Doorbraak Generatieve AI & Large Language Models",
}

# Cross-Language Mapping (Verbetering 5)
CROSS_LANGUAGE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "climate": {"en": "climate", "nl": "klimaat", "fr": "climat", "label": "Klimaat / Climate"},
    "klimaat": {"en": "climate", "nl": "klimaat", "fr": "climat", "label": "Klimaat / Climate"},
    "democracy": {"en": "democracy", "nl": "democratie", "fr": "démocratie", "label": "Democratie / Democracy"},
    "democratie": {"en": "democracy", "nl": "democratie", "fr": "démocratie", "label": "Democratie / Democracy"},
    "artificial intelligence": {
        "en": "artificial intelligence",
        "nl": "kunstmatige intelligentie",
        "fr": "intelligence artificielle",
        "label": "AI / Kunstmatige Intelligentie",
    },
    "kunstmatige intelligentie": {
        "en": "artificial intelligence",
        "nl": "kunstmatige intelligentie",
        "fr": "intelligence artificielle",
        "label": "AI / Kunstmatige Intelligentie",
    },
    "telegram": {"en": "telegram", "nl": "telegram", "fr": "télégramme", "label": "Telegraaf / Telegram"},
    "Paris": {"en": "Paris", "nl": "Parijs", "fr": "Paris", "label": "Parijs (Stad / Hoofdletter)"},
    "Parijs": {"en": "Paris", "nl": "Parijs", "fr": "Paris", "label": "Parijs (Stad / Hoofdletter)"},
    "paris": {"en": "paris", "nl": "parijs", "fr": "paris", "label": "paris (kleine letter)"},
    "parijs": {"en": "paris", "nl": "parijs", "fr": "paris", "label": "parijs (kleine letter)"},
}

# ==============================================================================
# CSS STYLING (AUTHENTIEKE GOOGLE MATERIAL LOOK)
# ==============================================================================
GOOGLE_NGRAM_CSS = """
<style>
    .stApp {
        background-color: #ffffff !important;
        color: #202124 !important;
        font-family: 'Google Sans', 'Roboto', 'Segoe UI', Arial, sans-serif !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.5rem !important;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-left: 3.5rem !important;
        padding-right: 3.5rem !important;
        max-width: 1400px !important;
    }
    .ngram-header {
        display: flex;
        align-items: baseline;
        font-size: 22px;
        font-weight: 400;
        letter-spacing: -0.2px;
        padding-bottom: 12px;
        margin-bottom: 14px;
        border-bottom: 1px solid #f1f3f4;
        user-select: none;
    }
    .g-blue   { color: #4285F4; font-weight: 500; }
    .g-red    { color: #EA4335; font-weight: 500; }
    .g-yellow { color: #FBBC05; font-weight: 500; }
    .g-green  { color: #34A853; font-weight: 500; }
    .ngram-title-text {
        color: #5f6368;
        margin-left: 6px;
        font-weight: 400;
    }
    div[data-testid="stTextInput"] {
        max-width: 680px;
    }
    div[data-testid="stTextInput"] > label {
        display: none !important;
    }
    div[data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        border: 1px solid #dfe1e5 !important;
        border-radius: 8px !important;
        padding: 10px 14px 10px 38px !important;
        font-size: 15px !important;
        color: #202124 !important;
        box-shadow: none !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2370757a' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'%3E%3C/circle%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'%3E%3C/line%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: 12px center !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[data-testid="stTextInput"] input:hover,
    div[data-testid="stTextInput"] input:focus {
        border-color: #1a73e8 !important;
        box-shadow: 0 1px 6px rgba(32, 33, 36, 0.12) !important;
    }
    div[data-testid="stPopover"] > div > button {
        border-radius: 999px !important;
        padding: 3px 14px !important;
        min-height: 32px !important;
        height: 32px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        line-height: 1.2 !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
        background-color: #e8f0fe !important;
        color: #1967d2 !important;
        border: 1px solid transparent !important;
    }
    div[data-testid="stPopover"] > div > button:hover {
        background-color: #d2e3fc !important;
        color: #174ea6 !important;
    }
    div.stButton > button {
        border-radius: 999px !important;
        padding: 2px 12px !important;
        min-height: 28px !important;
        font-size: 12.5px !important;
        font-weight: 400 !important;
        background-color: #ffffff !important;
        color: #3c4043 !important;
        border: 1px solid #dadce0 !important;
    }
    div.stButton > button:hover {
        background-color: #f8f9fa !important;
        border-color: #1a73e8 !important;
        color: #1a73e8 !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #1a73e8 !important;
        color: #ffffff !important;
        border: 1px solid #1a73e8 !important;
        font-weight: 500 !important;
    }
    .ngram-chart-footer {
        text-align: center;
        color: #5f6368;
        font-size: 12px;
        margin-top: -8px;
        margin-bottom: 20px;
    }
    .ngram-filter-divider {
        border-bottom: 1px solid #e8eaed;
        margin-top: 8px;
        margin-bottom: 18px;
    }
    /* Tab styling */
    button[data-baseweb="tab"] {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #5f6368 !important;
    }
    button[aria-selected="true"] {
        color: #1a73e8 !important;
        border-bottom-color: #1a73e8 !important;
    }
</style>
"""


# ==============================================================================
# GEBRUIKERSACCOUNTS, GESCHIEDENIS & VOORKEUREN (VERBETERING 6)
# ==============================================================================
def load_user_profiles() -> Dict[str, Any]:
    """Laadt gebruikersprofielen, geschiedenis en voorkeuren uit JSON."""
    if USER_PROFILES_FILE.exists():
        try:
            with open(USER_PROFILES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    default_data = {
        "active_profile": "Lucas (Onderzoeker)",
        "profiles": {
            "Lucas (Onderzoeker)": {
                "preferences": {
                    "regions": ["American English", "Dutch"],
                    "chart_type": "Lijngrafiek",
                    "interpolation": "monotone",
                    "palette": "Google Kleuren",
                    "y_scale": "Lineair",
                    "smoothing": 3,
                    "year_range": [1800, 2026],
                    "show_raw": True,
                    "show_events": True,
                    "show_points": False,
                    "stroke_width": 2,
                    "cross_lang_mode": False,
                },
                "history": [],
                "favorites": [],
            },
            "Gastgebruiker": {
                "preferences": {
                    "regions": ["American English", "British English"],
                    "chart_type": "Lijngrafiek",
                    "interpolation": "monotone",
                    "palette": "Google Kleuren",
                    "y_scale": "Lineair",
                    "smoothing": 3,
                    "year_range": [1800, 2026],
                    "show_raw": True,
                    "show_events": True,
                    "show_points": False,
                    "stroke_width": 2,
                    "cross_lang_mode": False,
                },
                "history": [],
                "favorites": [],
            },
        },
    }
    save_user_profiles(default_data)
    return default_data


def save_user_profiles(data: Dict[str, Any]) -> None:
    """Slaat gebruikersprofielen op naar JSON."""
    with open(USER_PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_search_to_history(
    profile_name: str,
    query: str,
    regions: List[str],
    pos_tag: str,
    case_sensitive: bool,
    year_range: Tuple[int, int],
    smoothing: int,
    cross_lang: bool,
) -> None:
    """Voegt een zoekopdracht toe aan de geschiedenis van het actieve profiel."""
    profiles_data = load_user_profiles()
    if profile_name not in profiles_data["profiles"]:
        return

    history_list = profiles_data["profiles"][profile_name].get("history", [])
    entry = {
        "query": query,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "regions": regions,
        "pos_tag": pos_tag,
        "case_sensitive": case_sensitive,
        "year_range": list(year_range),
        "smoothing": smoothing,
        "cross_lang": cross_lang,
    }

    if not (history_list and history_list[0].get("query") == query and history_list[0].get("regions") == regions):
        history_list.insert(0, entry)
        profiles_data["profiles"][profile_name]["history"] = history_list[:25]
        save_user_profiles(profiles_data)


# ==============================================================================
# AWS S3 & LIVE GOOGLE NGRAM DATABASE INTEGRATIE
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def query_aws_s3_ngram_bucket(s3_subpath: str = "ngrams/books/20090715/") -> Dict[str, Any]:
    """
    Bevraagt de publieke Google Books Ngram dataset op AWS S3:
    `aws s3 ls --no-sign-request s3://datasets.elasticmapreduce/ngrams/books/`
    """
    s3_uri = f"s3://datasets.elasticmapreduce/{s3_subpath}"
    cli_cmd = f"aws s3 ls --no-sign-request {s3_uri}"

    if shutil.which("aws") is not None:
        try:
            proc = subprocess.run(
                ["aws", "s3", "ls", "--no-sign-request", s3_uri],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
                return {
                    "success": True,
                    "method": "AWS CLI (`aws s3 ls --no-sign-request`)",
                    "command": cli_cmd,
                    "entries": lines,
                }
        except Exception:
            pass

    try:
        params = urllib.parse.urlencode(
            {"list-type": "2", "prefix": s3_subpath, "delimiter": "/", "max-keys": "50"}
        )
        url = f"https://s3.amazonaws.com/datasets.elasticmapreduce?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 NgramVisualBuilder/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_bytes = resp.read()

        root = ET.fromstring(xml_bytes)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        entries: List[str] = []

        for cp in root.findall("s3:CommonPrefixes", ns):
            prefix_el = cp.find("s3:Prefix", ns)
            if prefix_el is not None and prefix_el.text:
                folder_name = prefix_el.text[len(s3_subpath) :]
                entries.append(f"                           PRE {folder_name}")

        for content in root.findall("s3:Contents", ns):
            key_el = content.find("s3:Key", ns)
            size_el = content.find("s3:Size", ns)
            mod_el = content.find("s3:LastModified", ns)
            if key_el is not None and key_el.text and key_el.text != s3_subpath:
                fname = key_el.text[len(s3_subpath) :]
                fsize_mb = int(size_el.text) / (1024 * 1024) if size_el is not None and size_el.text else 0.0
                mod_date = mod_el.text[:19].replace("T", " ") if mod_el is not None and mod_el.text else ""
                entries.append(f"{mod_date}  {fsize_mb:8.2f} MB  {fname}")

        if entries:
            return {
                "success": True,
                "method": "AWS S3 REST API (`--no-sign-request` via HTTPS)",
                "command": cli_cmd,
                "entries": entries,
            }
    except Exception as exc:
        return {
            "success": False,
            "method": "Offline (Geen directe verbinding met s3://datasets.elasticmapreduce)",
            "command": cli_cmd,
            "error": str(exc),
            "entries": [
                "                           PRE chi-sim-all/",
                "                           PRE eng-1m/",
                "                           PRE eng-all/",
                "                           PRE eng-fiction-all/",
                "                           PRE eng-gb-all/",
                "                           PRE eng-us-all/",
                "                           PRE fre-all/",
                "                           PRE ger-all/",
                "                           PRE heb-all/",
                "                           PRE rus-all/",
                "                           PRE spa-all/",
            ],
        }

    return {"success": False, "method": "Fallback S3 Overzicht", "command": cli_cmd, "entries": []}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_ngram_from_google(
    terms: Tuple[str, ...],
    pos_code: str,
    case_sensitive: bool,
    regions: Tuple[str, ...],
    start_year: int = 1800,
    end_year: int = 2026,
) -> Optional[pd.DataFrame]:
    """Haalt live echte Ngram-tijdreeksen op voor elk willekeurig woord via Google API."""
    records: List[Dict[str, Any]] = []
    api_end_year = min(end_year, 2019)

    for region in regions:
        corpus_id = CORPUS_MAPPING.get(region, {}).get("google_corpus_id", "en-US-2019")
        query_items = []
        for t in terms:
            if pos_code in ("_NOUN", "_VERB", "_INF"):
                q_str = f"{t}{pos_code}"
            elif pos_code == "_ADJ_SIM":
                q_str = f"{t}_ADJ"
            else:
                q_str = t
            query_items.append(q_str)

        params = {
            "content": ",".join(query_items),
            "year_start": str(start_year),
            "year_end": str(api_end_year),
            "corpus": corpus_id,
            "smoothing": "0",
        }
        if not case_sensitive:
            params["case_insensitive"] = "true"

        url = f"https://books.google.com/ngrams/json?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NgramVisualBuilder/1.0"},
            )
            with urllib.request.urlopen(req, timeout=6) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

        if not payload:
            continue

        for item in payload:
            if not case_sensitive and item.get("type") == "EXPANSION":
                continue

            timeseries = item.get("timeseries", [])
            if not timeseries:
                continue

            raw_ngram_name = item.get("ngram", "")
            clean_name = raw_ngram_name.replace(" (All)", "")
            for suffix in ["_NOUN", "_VERB", "_INF", "_ADJ"]:
                if clean_name.endswith(suffix):
                    clean_name = clean_name[: -len(suffix)]

            years_list = list(range(start_year, api_end_year + 1))
            for idx, yr in enumerate(years_list):
                if idx >= len(timeseries):
                    break
                rel_freq_pct = float(timeseries[idx]) * 100.0
                t_norm = (yr - 1800) / 226.0
                est_total_books = int(120_000 * np.exp(3.6 * t_norm))
                est_raw_matches = max(0, int(round((rel_freq_pct / 100.0) * est_total_books)))

                records.append(
                    {
                        "year": yr,
                        "ngram": clean_name,
                        "pos_tag": pos_code if pos_code != "_ADJ_SIM" else "_NOUN",
                        "case_sensitive": bool(case_sensitive),
                        "corpus_region": region,
                        "raw_match_count": est_raw_matches,
                        "total_books_in_corpus": est_total_books,
                        "relative_frequency": rel_freq_pct,
                        "historical_event": HISTORICAL_EVENTS_LOOKUP.get(yr, None),
                    }
                )

            if end_year > api_end_year and timeseries:
                last_val_pct = float(timeseries[-1]) * 100.0
                for yr in range(api_end_year + 1, end_year + 1):
                    t_norm = (yr - 1800) / 226.0
                    est_total_books = int(120_000 * np.exp(3.6 * t_norm))
                    est_raw_matches = max(0, int(round((last_val_pct / 100.0) * est_total_books)))
                    records.append(
                        {
                            "year": yr,
                            "ngram": clean_name,
                            "pos_tag": pos_code if pos_code != "_ADJ_SIM" else "_NOUN",
                            "case_sensitive": bool(case_sensitive),
                            "corpus_region": region,
                            "raw_match_count": est_raw_matches,
                            "total_books_in_corpus": est_total_books,
                            "relative_frequency": last_val_pct,
                            "historical_event": HISTORICAL_EVENTS_LOOKUP.get(yr, None),
                        }
                    )

    if not records:
        return None
    return pd.DataFrame(records)


# ==============================================================================
# DATA INLADEN (SYNTHETISCHE DATASET)
# ==============================================================================
@st.cache_data
def load_ngram_data(csv_path: Path) -> pd.DataFrame:
    """Laadt de synthetische Ngram-dataset in."""
    df = pd.read_csv(csv_path)
    df["historical_event"] = df["historical_event"].where(df["historical_event"].notna(), None)
    return df


# ==============================================================================
# CONSTRUCTIEVE FOUTAFHANDELING & FEEDFORWARD
# ==============================================================================
KNOWN_TYPOS_AND_SYNTAX: Dict[str, Tuple[str, str]] = {
    "climmate": ("climate", "Er zit een dubbele 'm' in je zoekterm."),
    "climat": ("climate", "Typfout of Franse spelling. We stellen 'climate' (EN) of 'klimaat' (NL) voor."),
    "democracie": ("democracy", "Franse/oud-Nederlandse spelling. In het Engelse corpus staat 'democracy'."),
    "artifical intelligence": ("artificial intelligence", "Veelvoorkomende typfout: ontbrekende 'i' in 'artificial'."),
    "ai": ("artificial intelligence", "De afkorting 'AI' staat onder de term 'artificial intelligence'."),
    "telegramm": ("telegram", "Duitse spelling met dubbel 'm'. Gebruik 'telegram'."),
    "parijs": ("Parijs", "Zoek je naar de Franse hoofdstad? Gebruik de hoofdletterversie 'Parijs'."),
}


def validate_single_term(
    raw_term: str,
    available_terms: List[str],
    case_sensitive: bool,
) -> Dict[str, Optional[str]]:
    """Valideert een zoekterm en genereert constructieve feedforward suggesties."""
    cleaned = raw_term.strip()

    # Check 1: Oude POS-operator syntax
    for legacy_tag in ["_NOUN", "_VERB", "_INF", "_ADJ", "_ADV"]:
        if cleaned.upper().endswith(legacy_tag):
            base_word = cleaned[: -len(legacy_tag)].strip()
            matched_suggestion = base_word if base_word in available_terms else (base_word or "climate")
            return {
                "is_valid": False,
                "input_term": cleaned,
                "matched_term": None,
                "suggestion": matched_suggestion,
                "reason": (
                    f"Je typte de oude programmeersyntax `{legacy_tag}` in de zoekbalk. "
                    f"Gebruik de visuele **Woordsoort**-knop in de filterbalk en zoek op **'{matched_suggestion}'**."
                ),
            }

    # Check 2: Match in dataset
    if case_sensitive:
        if cleaned in available_terms:
            return {"is_valid": True, "input_term": cleaned, "matched_term": cleaned, "suggestion": None, "reason": None}
        lower_map = {t.lower(): t for t in available_terms}
        if cleaned.lower() in lower_map:
            alt_case = lower_map[cleaned.lower()]
            return {
                "is_valid": False,
                "input_term": cleaned,
                "matched_term": None,
                "suggestion": alt_case,
                "reason": (
                    f"Hoofdlettergevoeligheid staat aan en `{cleaned}` verschilt in "
                    f"hoofdlettergebruik van **'{alt_case}'**."
                ),
            }
    else:
        for term in available_terms:
            if term.lower() == cleaned.lower():
                return {"is_valid": True, "input_term": cleaned, "matched_term": term, "suggestion": None, "reason": None}

    # Check 3: Bekende typfouten
    lookup = cleaned.lower()
    if lookup in KNOWN_TYPOS_AND_SYNTAX:
        suggested_word, explanation = KNOWN_TYPOS_AND_SYNTAX[lookup]
        return {
            "is_valid": False,
            "input_term": cleaned,
            "matched_term": None,
            "suggestion": suggested_word,
            "reason": explanation,
        }

    # Check 4: Fuzzy matching
    close_matches = difflib.get_close_matches(
        lookup,
        [t.lower() for t in available_terms],
        n=1,
        cutoff=0.55,
    )
    if close_matches:
        best_lower = close_matches[0]
        best_term = next(t for t in available_terms if t.lower() == best_lower)
        return {
            "is_valid": False,
            "input_term": cleaned,
            "matched_term": None,
            "suggestion": best_term,
            "reason": f"Geen exact resultaat gevonden voor `{cleaned}`, maar deze term lijkt er sterk op.",
        }

    return {
        "is_valid": False,
        "input_term": cleaned,
        "matched_term": None,
        "suggestion": "climate",
        "reason": (
            f"De term `{cleaned}` zit niet in de lokale synthetische CSV. "
            f"Schakel in de filterbalk **Databron** over naar **Live Ngram + AWS S3** om elk willekeurig "
            f"woord op te vragen, of kies een testterm ({', '.join(f'**{t}**' for t in available_terms[:6])})."
        ),
    }


# ==============================================================================
# ANALYSE VAN GRAFIEK-BIJZONDERHEDEN (VERBETERING 3)
# ==============================================================================
def analyze_chart_peculiarities(df_filtered: pd.DataFrame, group_col: str = "series_label") -> pd.DataFrame:
    """
    Berekent gedetailleerde bijzonderheden & anomalieën per tijdreeks:
    - Startjaar (eerste significante detectie)
    - Hoogste piek (jaar, %, absolute hits, historische context)
    - Diepste dal (jaar, %, absolute hits)
    - Steilste groeiperiode (5-jaars moving delta)
    - Gemiddelde frequentie (%)
    - Totaal aantal absolute vermeldingen
    """
    results = []

    for series_name, group in df_filtered.groupby(group_col):
        group = group.sort_values("year")
        if group.empty:
            continue

        first_significant_row = group[group["relative_frequency"] > 0.0001]
        start_year = int(first_significant_row["year"].iloc[0]) if not first_significant_row.empty else int(group["year"].min())

        peak_idx = group["relative_frequency"].idxmax()
        peak_row = group.loc[peak_idx]
        peak_year = int(peak_row["year"])
        peak_val = float(peak_row["relative_frequency"])
        peak_matches = int(peak_row["raw_match_count"])
        peak_event = peak_row["historical_event"] if pd.notna(peak_row["historical_event"]) else "Geen geregistreerd event"

        valley_idx = group["relative_frequency"].idxmin()
        valley_row = group.loc[valley_idx]
        valley_year = int(valley_row["year"])
        valley_val = float(valley_row["relative_frequency"])
        valley_matches = int(valley_row["raw_match_count"])

        group_growth = group.copy()
        group_growth["growth_5yr"] = group_growth["relative_frequency"].diff(5)
        if not group_growth["growth_5yr"].dropna().empty:
            fastest_rise_idx = group_growth["growth_5yr"].idxmax()
            fastest_rise_year = int(group_growth.loc[fastest_rise_idx, "year"])
            fastest_period = f"{fastest_rise_year - 5} – {fastest_rise_year}"
        else:
            fastest_period = "N.v.t."

        avg_freq = float(group["relative_frequency"].mean())
        total_raw = int(group["raw_match_count"].sum())
        total_books = int(group["total_books_in_corpus"].mean())

        results.append(
            {
                "Reeks / Corpus": series_name,
                "Startjaar": start_year,
                "Piekjaar": peak_year,
                "Piekwaarde (%)": f"{peak_val:.5f}%",
                "Treffers bij Piek": f"{peak_matches:,}".replace(",", "."),
                "Historische Context bij Piek": peak_event,
                "Daljaar": valley_year,
                "Dalwaarde (%)": f"{valley_val:.5f}%",
                "Steilste Groei": fastest_period,
                "Gemiddeld (%)": f"{avg_freq:.5f}%",
                "Totale Vermeldingen": f"{total_raw:,}".replace(",", "."),
                "Gem. Boeken/Jaar": f"{total_books:,}".replace(",", "."),
            }
        )

    return pd.DataFrame(results)


# ==============================================================================
# PLACEHOLDER GRAFIEK BIJ LEEG CANVAS
# ==============================================================================
def render_google_placeholder_chart(year_range: Tuple[int, int]) -> alt.Chart:
    """Toont een lege Google Books Ngram Viewer grafiek bij ongeldige of lege invoer."""
    placeholder_df = pd.DataFrame(
        {
            "year": [year_range[0], year_range[1]],
            "relative_frequency": [0.0, 0.00014],
        }
    )

    base = (
        alt.Chart(placeholder_df)
        .mark_line(opacity=0)
        .encode(
            x=alt.X(
                "year:Q",
                title=None,
                scale=alt.Scale(domain=[int(year_range[0]), int(year_range[1])], zero=False, nice=False),
                axis=alt.Axis(
                    format="d",
                    grid=False,
                    domainColor="#dadce0",
                    tickColor="#dadce0",
                    labelColor="#5f6368",
                    labelFontSize=11,
                ),
            ),
            y=alt.Y(
                "relative_frequency:Q",
                title=None,
                scale=alt.Scale(domain=[0, 0.00014]),
                axis=alt.Axis(
                    format=".6f",
                    labelExpr="datum.label + '%'",
                    grid=True,
                    gridColor="#f1f3f4",
                    domain=False,
                    tickColor="#dadce0",
                    labelColor="#5f6368",
                    labelFontSize=11,
                ),
            ),
        )
        .properties(height=450)
    )

    msg = (
        alt.Chart(
            pd.DataFrame(
                {
                    "year": [(year_range[0] + year_range[1]) / 2],
                    "relative_frequency": [0.00007],
                    "text": ["Kies hierboven een voorgestelde zoekterm om de trendlijn te tonen"],
                }
            )
        )
        .mark_text(color="#80868b", fontSize=14)
        .encode(
            x=alt.X(
                "year:Q",
                scale=alt.Scale(domain=[int(year_range[0]), int(year_range[1])], zero=False, nice=False),
            ),
            y="relative_frequency:Q",
            text="text:N",
        )
    )

    return (base + msg).configure_view(strokeWidth=0)


# ==============================================================================
# HOOFDAPPLICATIE
# ==============================================================================
def main() -> None:
    st.markdown(GOOGLE_NGRAM_CSS, unsafe_allow_html=True)

    if not DATA_FILE.exists():
        st.error(f"Bestand `{DATA_FILE.name}` niet gevonden! Voer eerst `python generate_synthetic_ngram_data.py` uit.")
        st.stop()

    synthetic_df = load_ngram_data(DATA_FILE)
    available_ngrams: List[str] = sorted(
        synthetic_df["ngram"].unique().tolist(), key=lambda s: (s.lower(), s)
    )
    available_regions: List[str] = synthetic_df["corpus_region"].unique().tolist()
    profiles_data = load_user_profiles()

    # Initialiseer profiel & voorkeuren (Verbetering 6)
    if "current_profile" not in st.session_state:
        st.session_state["current_profile"] = profiles_data.get("active_profile", "Lucas (Onderzoeker)")

    active_user = st.session_state["current_profile"]
    if active_user not in profiles_data["profiles"]:
        active_user = list(profiles_data["profiles"].keys())[0]
        st.session_state["current_profile"] = active_user

    user_prefs = profiles_data["profiles"][active_user]["preferences"]

    if "search_query" not in st.session_state:
        st.session_state["search_query"] = "climate, democracy, telegram"
    if "year_range" not in st.session_state:
        st.session_state["year_range"] = tuple(user_prefs.get("year_range", [1800, 2026]))
    if "smoothing_val" not in st.session_state:
        st.session_state["smoothing_val"] = user_prefs.get("smoothing", 3)
    if "selected_regions" not in st.session_state:
        st.session_state["selected_regions"] = user_prefs.get("regions", ["American English"])
    if "selected_pos_label" not in st.session_state:
        st.session_state["selected_pos_label"] = "Alle woordsoorten"
    if "case_sensitive_val" not in st.session_state:
        st.session_state["case_sensitive_val"] = False
    if "data_source_mode" not in st.session_state:
        st.session_state["data_source_mode"] = "Hybride (Synthetisch + Live Ngram / AWS S3)"
    if "chart_type" not in st.session_state:
        st.session_state["chart_type"] = user_prefs.get("chart_type", "Lijngrafiek")
    if "palette" not in st.session_state:
        st.session_state["palette"] = user_prefs.get("palette", "Google Kleuren")
    if "y_scale" not in st.session_state:
        st.session_state["y_scale"] = user_prefs.get("y_scale", "Lineair")
    if "interpolation" not in st.session_state:
        st.session_state["interpolation"] = user_prefs.get("interpolation", "monotone")
    if "show_raw" not in st.session_state:
        st.session_state["show_raw"] = user_prefs.get("show_raw", True)
    if "show_events" not in st.session_state:
        st.session_state["show_events"] = user_prefs.get("show_events", True)
    if "show_points" not in st.session_state:
        st.session_state["show_points"] = user_prefs.get("show_points", False)
    if "stroke_width" not in st.session_state:
        st.session_state["stroke_width"] = user_prefs.get("stroke_width", 2)
    if "cross_lang_mode" not in st.session_state:
        st.session_state["cross_lang_mode"] = user_prefs.get("cross_lang_mode", False)

    # --------------------------------------------------------------------------
    # 1. HEADER MET GOOGLE LOGO & ACCOUNT WISSELAAR
    # --------------------------------------------------------------------------
    header_left, header_right = st.columns([3.8, 1.2])
    with header_left:
        st.markdown(
            """
            <div class="ngram-header">
                <span class="g-blue">G</span><span class="g-red">o</span><span class="g-yellow">o</span><span class="g-blue">g</span><span class="g-green">l</span><span class="g-red">e</span>
                <span class="ngram-title-text">Books Ngram Viewer — Visual Builder 2.0</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        p_col1, p_col2 = st.columns([2.5, 1.0])
        with p_col1:
            all_profs = list(profiles_data["profiles"].keys())
            c_prof = st.selectbox(
                "👤 Profiel",
                options=all_profs,
                index=all_profs.index(active_user),
                key="head_prof_select",
                label_visibility="collapsed",
            )
            if c_prof != active_user:
                st.session_state["current_profile"] = c_prof
                st.rerun()
        with p_col2:
            with st.popover("⚙️ Account", icon="👤"):
                st.write(f"Ingelogd als: **{active_user}**")
                st.caption(f"Zoekgeschiedenis: {len(profiles_data['profiles'][active_user].get('history', []))} items")
                st.caption(f"Favorieten: {len(profiles_data['profiles'][active_user].get('favorites', []))} items")
                st.info("Ga naar het tabblad **'👤 Account & Geschiedenis'** voor alle profielopties.")

    # ==========================================================================
    # HOOFDNAVIGATIE TABS (VERBETERING 2: STRUCTURELE CONCRETE INDELING)
    # ==========================================================================
    tab_visual, tab_peculiarities, tab_cross_lang, tab_documentation, tab_account = st.tabs(
        [
            "📊 Visual Builder (Grafiek & Zoeken)",
            "🔍 Grafiek-Bijzonderheden (Pieken & Dalen)",
            "🌐 Meertalig Zoeken (Cross-Language)",
            "📖 Kennisbank & Documentatie",
            "👤 Account, Geschiedenis & Instellingen",
        ]
    )

    # ==========================================================================
    # TAB 1: VISUAL BUILDER (DE HOOFDAPPLICATIE)
    # ==========================================================================
    with tab_visual:
        # Zoekbalk & Meertalige Toggle
        search_col, cross_col = st.columns([4.2, 1.0])
        with search_col:
            user_query = st.text_input(
                label="Search Ngrams",
                key="search_query",
                placeholder="Search Ngrams (bijv. climate, democracy, Sherlock Holmes, Frankenstein)",
                help="Typ één of meerdere zoektermen gescheiden door een komma.",
            )
        with cross_col:
            cross_lang_toggle = st.checkbox(
                "🌐 Cross-Language",
                value=st.session_state["cross_lang_mode"],
                help="Koppelt automatisch vertalingen tussen Engels en Nederlands (bijv. climate <-> klimaat).",
            )
            st.session_state["cross_lang_mode"] = cross_lang_toggle

        # ----------------------------------------------------------------------
        # FILTER-PILLS BALK DIRECT ONDER DE ZOEKBALK (MET VERBETERING 1 & 4)
        # ----------------------------------------------------------------------
        pos_options_map = {
            "Alle woordsoorten": "NONE",
            "Zelfstandig naamwoord (Noun)": "_NOUN",
            "Werkwoord - vervoegd (Verb)": "_VERB",
            "Hele werkwoord / Infinitief (Inf)": "_INF",
            "Bijvoeglijk naamwoord (Adj)": "_ADJ_SIM",
        }

        yr_start, yr_end = st.session_state["year_range"]
        current_regions = st.session_state["selected_regions"]
        region_pill_label = (
            current_regions[0]
            if len(current_regions) == 1
            else f"{len(current_regions)} Corpora"
            if len(current_regions) > 1
            else "Kies Corpus"
        )
        current_pos_label = st.session_state["selected_pos_label"]
        current_pos_code = pos_options_map[current_pos_label]
        case_pill_label = "Case-Sensitive" if st.session_state["case_sensitive_val"] else "Case-Insensitive"
        smoothing_pill_label = f"Smoothing ({st.session_state['smoothing_val']})"

        # Filter Pills layout
        p_col1, p_col2, p_col3, p_col4, p_col5, p_col6, p_col7, p_col8 = st.columns(
            [1.0, 1.25, 1.2, 1.15, 1.35, 1.45, 1.2, 1.4]
        )

        # PILL 1: Tijdlijn-slider
        with p_col1:
            with st.popover(f"{yr_start} - {yr_end}", use_container_width=True):
                st.markdown("**📅 Tijdsperiode selecteren**")
                year_range: Tuple[int, int] = st.slider(
                    label="Jaartallen (`year`)",
                    min_value=1800,
                    max_value=2026,
                    key="year_range",
                    step=1,
                    help="Versleep de schuifregelaar om in te zoomen op specifieke jaartallen tussen 1800 en 2026.",
                )

        # PILL 2: Geografisch Corpus
        with p_col2:
            with st.popover(region_pill_label, use_container_width=True):
                st.markdown("**🌍 Corpus & Geografische Regio**")
                selected_regions = st.multiselect(
                    label="Selecteer corpora (`corpus_region`)",
                    options=available_regions,
                    key="selected_regions",
                    help="Kies corpora om taal- en cultuurverschillen te vergelijken.",
                )

        # PILL 3: Hoofdlettergevoeligheid
        with p_col3:
            with st.popover(case_pill_label, use_container_width=True):
                st.markdown("**🔤 Hoofdlettergevoeligheid**")
                case_sensitive = st.checkbox(
                    label="Hoofdlettergevoelig (`Case-Sensitive`)",
                    key="case_sensitive_val",
                    help="Aan: strikt onderscheid tussen bijv. 'Paris' en 'paris'. Uit: combineert alle schrijfwijzen.",
                )

        # PILL 4: Smoothing Slider
        with p_col4:
            with st.popover(smoothing_pill_label, use_container_width=True):
                st.markdown("**🌊 Trend-afvlakking (Smoothing)**")
                smoothing_window: int = st.slider(
                    label="Smoothing (voortschrijdend gemiddelde)",
                    min_value=0,
                    max_value=50,
                    key="smoothing_val",
                    step=1,
                    help="0 = ruwe data; 5-15 = gebalanceerd; 25-50 = macro-trend.",
                )

        # PILL 5: Woordsoort-selector
        with p_col5:
            pos_short = "Woordsoort: Alle" if current_pos_code == "NONE" else f"Tag: {current_pos_code}"
            with st.popover(pos_short, use_container_width=True):
                st.markdown("**🏷️ Woordsoort & Vervoeging (`pos_tag`)**")
                selected_pos_label = st.selectbox(
                    label="Kies grammaticale rol",
                    options=list(pos_options_map.keys()),
                    key="selected_pos_label",
                    help="Vervangt oude _NOUN / _VERB syntax door visuele selectie.",
                )
                selected_pos_code = pos_options_map[selected_pos_label]

        # PILL 6: Databron (Hybride / S3 / Synthetisch)
        with p_col6:
            mode_short = (
                "Bron: Synthetisch"
                if st.session_state["data_source_mode"].startswith("Alleen Synthetisch")
                else "Bron: S3 / Live + CSV"
            )
            with st.popover(mode_short, use_container_width=True):
                st.markdown("**☁️ Ngram Database & AWS S3 Koppeling**")
                data_source_mode = st.radio(
                    label="Kies databron:",
                    options=[
                        "Hybride (Synthetisch + Live Ngram / AWS S3)",
                        "Live Google Ngram Database (Online / S3)",
                        "Alleen Synthetische CSV (Test-modus)",
                    ],
                    key="data_source_mode",
                )

        # PILL 7: Grafiek-Vormgeving (Verbetering 4)
        with p_col7:
            with st.popover("🎨 Vormgeving", use_container_width=True):
                st.markdown("**🎨 Grafiek-Vormgeving & Stijl**")
                chart_type = st.selectbox(
                    "Vorm van de grafiek",
                    options=["Lijngrafiek", "Vlakgrafiek", "Staafgrafiek", "Trapgrafiek"],
                    index=["Lijngrafiek", "Vlakgrafiek", "Staafgrafiek", "Trapgrafiek"].index(
                        st.session_state["chart_type"]
                    ),
                    key="chart_type_sel",
                )
                st.session_state["chart_type"] = chart_type

                palette_choice = st.selectbox(
                    "Kleurenpalet",
                    options=list(PALETTES_MAP.keys()),
                    index=list(PALETTES_MAP.keys()).index(st.session_state["palette"]),
                    key="palette_choice_sel",
                )
                st.session_state["palette"] = palette_choice

                y_scale_choice = st.radio(
                    "Y-as schaal",
                    options=["Lineair", "Logaritmisch"],
                    index=["Lineair", "Logaritmisch"].index(st.session_state["y_scale"]),
                    horizontal=True,
                    key="yscale_sel",
                )
                st.session_state["y_scale"] = y_scale_choice

                interp_choice = st.selectbox(
                    "Lijncurve (Interpolatie)",
                    options=["monotone", "linear", "basis"],
                    index=["monotone", "linear", "basis"].index(st.session_state["interpolation"]),
                    key="interp_sel",
                )
                st.session_state["interpolation"] = interp_choice

                stroke_w = st.slider("Lijndikte (px)", 1, 6, st.session_state["stroke_width"], key="sw_sel")
                st.session_state["stroke_width"] = stroke_w

                st.divider()
                s_raw = st.checkbox("Toon ruwe data op achtergrond", value=st.session_state["show_raw"], key="sr_cb")
                st.session_state["show_raw"] = s_raw
                s_ev = st.checkbox("Toon historische annotatielijnen", value=st.session_state["show_events"], key="se_cb")
                st.session_state["show_events"] = s_ev
                s_pt = st.checkbox("Toon datapunten op lijn", value=st.session_state["show_points"], key="sp_cb")
                st.session_state["show_points"] = s_pt

        # PILL 8: Korte Info Popup bij de Instellingen (Verbetering 1)
        with p_col8:
            with st.popover("ℹ️ Hulp & Uitleg", use_container_width=True):
                st.markdown(
                    """
                    ### 💡 Hulp bij Instellingen & Filters
                    * **📅 Tijdsperiode:** In- en uitzoomen tussen 1800 en 2026.
                    * **🌍 Corpus:** Normalisatie per taalgebied (voorkomt bias door publicatievolume).
                    * **🔤 Hoofdlettergevoelig:** Onderscheid tussen bijv. 'Paris' (stad) en 'paris'.
                    * **🌊 Smoothing:** Berekent een moving average om jaarlijkse ruis glad te strijken.
                    * **🏷️ Woordsoort:** Filtert op zelfstandig naamwoord, werkwoord of infinitief.
                    * **☁️ Databron:** Schakel tussen synthetische testdata en live Google Books / S3 queries.
                    * **🎨 Vormgeving:** Pas grafiektype (vlak/lijn/staaf/trap) en kleurenpalet aan.
                    """
                )

        st.markdown('<div class="ngram-filter-divider"></div>', unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # VALIDATIE & DATA OPHALEN
        # ----------------------------------------------------------------------
        raw_terms = [t.strip() for t in user_query.split(",") if t.strip()]
        if not raw_terms:
            raw_terms = [""]

        if not selected_regions:
            st.warning("Geen corpus geselecteerd. Klik hierboven op **Kies Corpus** en vink ten minste één regio aan.")
            st.altair_chart(render_google_placeholder_chart(year_range), use_container_width=True)
            st.stop()

        # Als Cross-Language actief is, vul querytermen aan met vertalingen (Verbetering 5)
        augmented_terms = list(raw_terms)
        if st.session_state["cross_lang_mode"]:
            for t in raw_terms:
                c_info = CROSS_LANGUAGE_MAPPINGS.get(t, {})
                if c_info:
                    en_t = c_info.get("en")
                    nl_t = c_info.get("nl")
                    if en_t and en_t not in augmented_terms:
                        augmented_terms.append(en_t)
                    if nl_t and nl_t not in augmented_terms:
                        augmented_terms.append(nl_t)

        validations = [
            validate_single_term(t, available_ngrams, case_sensitive) for t in augmented_terms
        ]

        known_error_items = []
        unknown_but_fetchable_terms = []
        valid_synthetic_terms = []

        for v in validations:
            if v["is_valid"]:
                if v["matched_term"] not in valid_synthetic_terms:
                    valid_synthetic_terms.append(v["matched_term"])
            else:
                inp_lower = (v["input_term"] or "").lower()
                has_legacy_tag = any(
                    (v["input_term"] or "").upper().endswith(tag)
                    for tag in ["_NOUN", "_VERB", "_INF", "_ADJ", "_ADV"]
                )
                is_close_typo = bool(
                    difflib.get_close_matches(inp_lower, [t.lower() for t in available_ngrams], n=1, cutoff=0.68)
                )
                if inp_lower in KNOWN_TYPOS_AND_SYNTAX or has_legacy_tag or is_close_typo:
                    known_error_items.append(v)
                else:
                    unknown_but_fetchable_terms.append(v)

        if known_error_items:
            for bad in known_error_items:
                suggested = bad["suggestion"]
                st.warning(
                    f"💡 **Geen exacte resultaten voor `{bad['input_term']}`** — {bad['reason']}  \n"
                    f"Bedoelde je misschien **`{suggested}`**?"
                )

            corrected_terms = [
                v["matched_term"] if v["is_valid"] else v["suggestion"] for v in validations
            ]
            corrected_query_str = ", ".join(dict.fromkeys(corrected_terms))

            btn_col, _ = st.columns([2.5, 5.5])
            with btn_col:
                if st.button(f"Bedoelde je: '{corrected_query_str}'? (Klik om toe te passen)", type="primary"):
                    st.session_state["search_query"] = corrected_query_str
                    st.rerun()

        frames_to_combine: List[pd.DataFrame] = []
        actual_pos_tag = "_NOUN" if selected_pos_code == "_ADJ_SIM" else selected_pos_code

        if data_source_mode == "Live Google Ngram Database (Online / S3)":
            live_candidates = [
                v["input_term"] for v in validations if v not in known_error_items and v["input_term"]
            ]
            if live_candidates:
                with st.spinner("Bezig met ophalen uit de Google Books Ngram Database..."):
                    live_df = fetch_live_ngram_from_google(
                        terms=tuple(live_candidates),
                        pos_code=selected_pos_code,
                        case_sensitive=case_sensitive,
                        regions=tuple(selected_regions),
                        start_year=1800,
                        end_year=2026,
                    )
                if live_df is not None and not live_df.empty:
                    frames_to_combine.append(live_df)
        else:
            if valid_synthetic_terms:
                syn_part = synthetic_df[
                    (synthetic_df["ngram"].isin(valid_synthetic_terms))
                    & (synthetic_df["pos_tag"] == actual_pos_tag)
                    & (synthetic_df["case_sensitive"] == case_sensitive)
                    & (synthetic_df["corpus_region"].isin(selected_regions))
                ].copy()
                if selected_pos_code == "_ADJ_SIM":
                    syn_part["relative_frequency"] = (syn_part["relative_frequency"] * 0.42).round(6)
                    syn_part["raw_match_count"] = (syn_part["raw_match_count"] * 0.42).round().astype(int)
                frames_to_combine.append(syn_part)

            if data_source_mode.startswith("Hybride") and unknown_but_fetchable_terms:
                fetch_terms = [v["input_term"] for v in unknown_but_fetchable_terms if v["input_term"]]
                if fetch_terms:
                    with st.spinner(f"Zoekterm(en) {', '.join(fetch_terms)} live opvragen bij Google Books..."):
                        live_df = fetch_live_ngram_from_google(
                            terms=tuple(fetch_terms),
                            pos_code=selected_pos_code,
                            case_sensitive=case_sensitive,
                            regions=tuple(selected_regions),
                            start_year=1800,
                            end_year=2026,
                        )
                    if live_df is not None and not live_df.empty:
                        frames_to_combine.append(live_df)
                    else:
                        for un in unknown_but_fetchable_terms:
                            st.warning(
                                f"💡 **Geen resultaten voor `{un['input_term']}`** — {un['reason']}  \n"
                                f"Bedoelde je misschien **`{un['suggestion']}`**?"
                            )
            elif data_source_mode.startswith("Alleen Synthetisch") and unknown_but_fetchable_terms:
                for un in unknown_but_fetchable_terms:
                    st.warning(
                        f"💡 **Geen exacte resultaten voor `{un['input_term']}`** — {un['reason']}  \n"
                        f"Bedoelde je misschien **`{un['suggestion']}`**?"
                    )

        if not frames_to_combine:
            st.altair_chart(render_google_placeholder_chart(year_range), use_container_width=True)
            st.stop()

        combined_df = pd.concat(frames_to_combine, ignore_index=True)
        filtered_df = combined_df[
            (combined_df["year"] >= year_range[0]) & (combined_df["year"] <= year_range[1])
        ].copy()

        if filtered_df.empty:
            st.altair_chart(render_google_placeholder_chart(year_range), use_container_width=True)
            st.stop()

        # Label per lijn
        if len(selected_regions) == 1 and not st.session_state["cross_lang_mode"]:
            filtered_df["series_label"] = filtered_df["ngram"]
        else:
            filtered_df["series_label"] = filtered_df["ngram"] + " (" + filtered_df["corpus_region"] + ")"

        filtered_df.sort_values(by=["series_label", "year"], inplace=True)

        # Voeg zoekopdracht toe aan account geschiedenis (Verbetering 6)
        add_search_to_history(
            profile_name=active_user,
            query=user_query,
            regions=selected_regions,
            pos_tag=selected_pos_code,
            case_sensitive=case_sensitive,
            year_range=year_range,
            smoothing=smoothing_window,
            cross_lang=st.session_state["cross_lang_mode"],
        )

        # Smoothing berekenen
        window_size = max(1, smoothing_window * 2 + 1) if smoothing_window > 0 else 1
        filtered_df["smoothed_frequency"] = filtered_df.groupby("series_label")[
            "relative_frequency"
        ].transform(lambda s: s.rolling(window=window_size, min_periods=1, center=True).mean())

        # ----------------------------------------------------------------------
        # GRAFIEK RENDEREN MET VORMGEVINGSOPTIES (VERBETERING 4)
        # ----------------------------------------------------------------------
        series_order = filtered_df["series_label"].unique().tolist()
        palette_val = PALETTES_MAP.get(st.session_state["palette"], GOOGLE_LINE_COLORS)
        if isinstance(palette_val, list):
            color_scale = alt.Scale(domain=series_order, range=palette_val[: max(1, len(series_order))])
        else:
            color_scale = alt.Scale(scheme=palette_val)

        focus_selection = alt.selection_point(fields=["series_label"], on="click", clear="dblclick")

        x_min = int(year_range[0])
        x_max = int(year_range[1])
        year_span = max(1, x_max - x_min)

        if year_span > 160:
            tick_step = 20
        elif year_span > 80:
            tick_step = 10
        elif year_span > 35:
            tick_step = 5
        elif year_span > 15:
            tick_step = 2
        else:
            tick_step = 1

        first_tick = ((x_min + tick_step - 1) // tick_step) * tick_step
        x_tick_values = list(range(first_tick, x_max + 1, tick_step))
        if x_min not in x_tick_values:
            x_tick_values.insert(0, x_min)

        shared_x_scale = alt.Scale(domain=[x_min, x_max], zero=False, nice=False)

        max_y_val = float(filtered_df["smoothed_frequency"].max()) if not filtered_df.empty else 0.01
        if max_y_val < 0.00005:
            y_format = ".7f"
        elif max_y_val < 0.0005:
            y_format = ".6f"
        elif max_y_val < 0.01:
            y_format = ".5f"
        else:
            y_format = ".4f"

        y_scale_type = "log" if st.session_state["y_scale"] == "Logaritmisch" else "linear"
        y_scale_conf = alt.Scale(type=y_scale_type) if y_scale_type == "log" else alt.Scale(zero=True)

        base_chart = alt.Chart(filtered_df).encode(
            x=alt.X(
                "year:Q",
                title=None,
                scale=shared_x_scale,
                axis=alt.Axis(
                    format="d",
                    values=x_tick_values,
                    grid=False,
                    domainColor="#dadce0",
                    tickColor="#dadce0",
                    labelColor="#5f6368",
                    labelFontSize=11,
                    labelPadding=6,
                ),
            ),
            color=alt.Color("series_label:N", scale=color_scale, legend=None),
            opacity=alt.condition(focus_selection, alt.value(1.0), alt.value(0.2)),
        )

        tooltip_config = [
            alt.Tooltip("year:Q", title="Jaar"),
            alt.Tooltip("series_label:N", title="Term / Corpus"),
            alt.Tooltip("smoothed_frequency:Q", title="Frequentie (Smoothed %)", format=".6f"),
            alt.Tooltip("relative_frequency:Q", title="Frequentie (Ruw %)", format=".6f"),
            alt.Tooltip("raw_match_count:Q", title="Absolute treffers", format=","),
            alt.Tooltip("total_books_in_corpus:Q", title="Boeken in corpus", format=","),
            alt.Tooltip("historical_event:N", title="Historische gebeurtenis"),
        ]

        chart_shape = st.session_state["chart_type"]
        curv = "step-after" if chart_shape == "Trapgrafiek" else st.session_state["interpolation"]

        if chart_shape in ["Lijngrafiek", "Trapgrafiek"]:
            main_mark = base_chart.mark_line(
                strokeWidth=st.session_state["stroke_width"],
                interpolate=curv,
                point=st.session_state["show_points"],
            ).encode(
                y=alt.Y(
                    "smoothed_frequency:Q",
                    title=None,
                    scale=y_scale_conf,
                    axis=alt.Axis(
                        format=y_format,
                        labelExpr="datum.label + '%'",
                        grid=True,
                        gridColor="#f1f3f4",
                        domain=False,
                        tickColor="#dadce0",
                        tickSize=4,
                        labelColor="#5f6368",
                        labelFontSize=11,
                    ),
                ),
                tooltip=tooltip_config,
            ).add_params(focus_selection)
        elif chart_shape == "Vlakgrafiek":
            main_mark = base_chart.mark_area(
                opacity=0.45,
                interpolate=curv,
                line={"strokeWidth": st.session_state["stroke_width"]},
            ).encode(
                y=alt.Y(
                    "smoothed_frequency:Q",
                    title=None,
                    scale=y_scale_conf,
                    axis=alt.Axis(
                        format=y_format,
                        labelExpr="datum.label + '%'",
                        grid=True,
                        gridColor="#f1f3f4",
                        domain=False,
                        tickColor="#dadce0",
                        tickSize=4,
                        labelColor="#5f6368",
                        labelFontSize=11,
                    ),
                ),
                tooltip=tooltip_config,
            ).add_params(focus_selection)
        elif chart_shape == "Staafgrafiek":
            main_mark = base_chart.mark_bar(opacity=0.75).encode(
                y=alt.Y(
                    "smoothed_frequency:Q",
                    title=None,
                    scale=y_scale_conf,
                    axis=alt.Axis(
                        format=y_format,
                        labelExpr="datum.label + '%'",
                        grid=True,
                        gridColor="#f1f3f4",
                        domain=False,
                        tickColor="#dadce0",
                        tickSize=4,
                        labelColor="#5f6368",
                        labelFontSize=11,
                    ),
                ),
                tooltip=tooltip_config,
            ).add_params(focus_selection)
        else:
            main_mark = base_chart.mark_line(strokeWidth=st.session_state["stroke_width"]).encode(
                y=alt.Y("smoothed_frequency:Q", scale=y_scale_conf), tooltip=tooltip_config
            )

        end_points_df = filtered_df[filtered_df["year"] == x_max].copy()
        direct_labels = (
            alt.Chart(end_points_df)
            .mark_text(align="left", baseline="middle", dx=8, fontSize=13, fontWeight=500, clip=False)
            .encode(
                x=alt.X("year:Q", scale=shared_x_scale),
                y="smoothed_frequency:Q",
                text="series_label:N",
                color=alt.Color("series_label:N", scale=color_scale, legend=None),
                opacity=alt.condition(focus_selection, alt.value(1.0), alt.value(0.25)),
            )
            .add_params(focus_selection)
        )

        chart_layers = [main_mark, direct_labels]

        # Ruwe achtergrond data lijn bij smoothing > 0 (Verbetering 4)
        if st.session_state["show_raw"] and smoothing_window > 0:
            raw_background = base_chart.mark_line(
                strokeWidth=1.0, opacity=0.22, strokeDash=[3, 3]
            ).encode(y=alt.Y("relative_frequency:Q", scale=y_scale_conf))
            chart_layers.insert(0, raw_background)

        # Historische gebeurtenissen marker-lijnen
        events_in_view = (
            filtered_df[filtered_df["historical_event"].notna()][["year", "historical_event"]]
            .drop_duplicates(subset=["year"])
            .sort_values("year")
        )
        if st.session_state["show_events"] and not events_in_view.empty:
            event_rules = (
                alt.Chart(events_in_view)
                .mark_rule(color="#dadce0", strokeDash=[3, 3], opacity=0.75)
                .encode(
                    x=alt.X("year:Q", scale=shared_x_scale),
                    tooltip=[
                        alt.Tooltip("year:Q", title="Historisch jaar"),
                        alt.Tooltip("historical_event:N", title="Gebeurtenis"),
                    ],
                )
            )
            chart_layers.insert(0, event_rules)

        final_chart = (
            alt.layer(*chart_layers)
            .properties(
                height=460,
                padding={"left": 10, "top": 12, "right": 160, "bottom": 10},
            )
            .configure_view(strokeWidth=0)
        )

        st.altair_chart(final_chart, use_container_width=True)

        st.markdown(
            '<div class="ngram-chart-footer">(klik op een lijn/label voor focus • hover voor exacte aantallen & historische pieken)</div>',
            unsafe_allow_html=True,
        )

        # Snel Acties & KPI metrics
        kpi_col1, kpi_col2, kpi_col3, fav_save_col = st.columns([1.5, 1.5, 1.5, 1.5])
        with kpi_col1:
            tot_raw = int(filtered_df["raw_match_count"].sum())
            st.metric("Totale Treffers", f"{tot_raw:,}".replace(",", "."))
        with kpi_col2:
            pk_row = filtered_df.loc[filtered_df["relative_frequency"].idxmax()]
            st.metric("Piekjaar", f"{int(pk_row['year'])}", delta=f"{pk_row['relative_frequency']:.5f}%")
        with kpi_col3:
            avg_b = int(filtered_df["total_books_in_corpus"].mean())
            st.metric("Gem. Boeken/Jaar", f"{avg_b:,}".replace(",", "."))
        with fav_save_col:
            if st.button("⭐ Bewaar Zoekopdracht", help="Sla deze zoekopdracht op in je profiel-favorieten"):
                favs = profiles_data["profiles"][active_user].get("favorites", [])
                favs.append(
                    {
                        "query": user_query,
                        "label": f"{user_query} ({year_range[0]}-{year_range[1]})",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "regions": selected_regions,
                        "pos_tag": selected_pos_code,
                        "case_sensitive": case_sensitive,
                        "year_range": list(year_range),
                        "smoothing": smoothing_window,
                    }
                )
                profiles_data["profiles"][active_user]["favorites"] = favs
                save_user_profiles(profiles_data)
                st.success("Opgeslagen als favoriet!")

        # ----------------------------------------------------------------------
        # BEKNOPTE BIJZONDERHEDEN TABEL (VERBETERING 3 IN HOOFDVIEW)
        # ----------------------------------------------------------------------
        with st.expander("🔍 Bekijk beknopte tabel met grafiek-bijzonderheden (pieken, dalen, startjaar)", expanded=True):
            pec_df = analyze_chart_peculiarities(filtered_df, group_col="series_label")
            st.dataframe(pec_df, use_container_width=True, hide_index=True)

        # Historische context onder de grafiek
        st.markdown("##### Historische context in deze tijdsperiode (`historical_event`)")
        if events_in_view.empty:
            st.info(f"Geen gemarkeerde historische gebeurtenissen tussen {year_range[0]} en {year_range[1]}.")
        else:
            ev_cols = st.columns(3)
            for idx, (_, ev_row) in enumerate(events_in_view.iterrows()):
                yr = int(ev_row["year"])
                desc = ev_row["historical_event"]
                yr_data = filtered_df[filtered_df["year"] == yr]
                top_match = yr_data.sort_values("relative_frequency", ascending=False).iloc[0]
                with ev_cols[idx % 3]:
                    st.info(
                        f"**{desc}**  \n"
                        f"Hoogste frequentie in {yr}: `{top_match['series_label']}` "
                        f"({top_match['relative_frequency']:.5f}% • "
                        f"{int(top_match['raw_match_count']):,} treffers)"
                    )

        # ----------------------------------------------------------------------
        # AWS S3 OPEN DATA BROWSER
        # ----------------------------------------------------------------------
        with st.expander("☁️ AWS S3 Google Books Ngram Database Explorer (`s3://datasets.elasticmapreduce/ngrams/books/`)"):
            st.markdown(
                "Hieronder wordt rechtstreeks de publieke **Google Books Ngrams S3-bucket** op AWS Open Data "
                "bevraagd met het commando `--no-sign-request`."
            )
            s3_col1, s3_col2 = st.columns([2.2, 2.8])
            with s3_col1:
                selected_s3_path = st.selectbox(
                    label="Kies een pad in `s3://datasets.elasticmapreduce/`:",
                    options=[
                        "ngrams/books/",
                        "ngrams/books/20090715/",
                        "ngrams/books/20090715/eng-us-all/",
                        "ngrams/books/20090715/eng-us-all/1gram/",
                        "ngrams/books/20090715/eng-gb-all/1gram/",
                        "ngrams/books/20090715/eng-fiction-all/1gram/",
                    ],
                    index=1,
                    key="s3_path_sel",
                )
            s3_result = query_aws_s3_ngram_bucket(selected_s3_path)
            with s3_col2:
                st.markdown("**Uitgevoerd AWS S3 commando:**")
                st.code(s3_result["command"], language="bash")
                st.caption(f"Verbindingsmethode: **{s3_result['method']}**")
            if s3_result["entries"]:
                st.code("\n".join(s3_result["entries"][:25]), language="text")

    # ==========================================================================
    # TAB 2: UITGEBREIDE GRAFIEK-BIJZONDERHEDEN (VERBETERING 3)
    # ==========================================================================
    with tab_peculiarities:
        st.header("🔍 Diepte-Analyse: Grafiek-Bijzonderheden, Pieken & Dalen")
        st.markdown(
            """
            Deze tabel analyseert automatisch de wiskundige eigenschappen van elke actieve curve:
            wanneer de term voor het eerst opkwam, waar het absolute hoogtepunt ligt, wat het diepste dal was,
            en welke historische gebeurtenissen samenvallen met piekjaren.
            """
        )

        if "filtered_df" in locals() and not filtered_df.empty:
            pec_full_df = analyze_chart_peculiarities(filtered_df, group_col="series_label")
            st.dataframe(pec_full_df, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download Bijzonderheden Tabel (CSV)",
                data=pec_full_df.to_csv(index=False).encode("utf-8"),
                file_name="grafiek_bijzonderheden.csv",
                mime="text/csv",
            )

            st.subheader("📌 Gemarkeerde Mijlpalen in het Geselecteerde Bereik")
            if not events_in_view.empty:
                pec_ev_cols = st.columns(2)
                for i, (_, ev_row) in enumerate(events_in_view.iterrows()):
                    yr = int(ev_row["year"])
                    desc = ev_row["historical_event"]
                    yr_df = filtered_df[filtered_df["year"] == yr]
                    mean_f = yr_df["relative_frequency"].mean()
                    total_m = yr_df["raw_match_count"].sum()
                    with pec_ev_cols[i % 2]:
                        st.info(f"📅 **{desc}**\n\n* Gemiddelde frequentie in {yr}: **{mean_f:.5f}%**\n* Absolute vermeldingen: **{total_m:,}**")
        else:
            st.info("Voer eerst een geldige zoekopdracht uit in het tabblad 'Visual Builder'.")

    # ==========================================================================
    # TAB 3: MEERTALIG ZOEKEN (CROSS-LANGUAGE - VERBETERING 5)
    # ==========================================================================
    with tab_cross_lang:
        st.header("🌐 Cross-Language Zoeken & Vergelijken")
        st.markdown(
            """
            In klassieke Ngram viewers ben je gebonden aan één taal. Met **Cross-Language Support**
            worden concepten over taal- en landsgrenzen gekoppeld (bijv. het Engelse *climate* versus het Nederlandse *klimaat*).
            """
        )

        cl_col1, cl_col2 = st.columns([2.0, 3.0])
        with cl_col1:
            st.subheader("Kies een Meertalig Concept")
            concept_choices = [
                ("Klimaat (climate / klimaat)", "climate"),
                ("Democratie (democracy / democratie)", "democracy"),
                ("Kunstmatige Intelligentie (artificial intelligence / kunstmatige intelligentie)", "artificial intelligence"),
                ("Telegraaf (telegram)", "telegram"),
                ("Parijs (Paris / Parijs)", "Paris"),
            ]
            selected_concept_key = st.radio(
                "Kies een concept om over talen heen te onderzoeken:",
                options=[c[1] for c in concept_choices],
                format_func=lambda k: next(c[0] for c in concept_choices if c[1] == k),
                key="concept_radio_sel",
            )

            mapping = CROSS_LANGUAGE_MAPPINGS.get(selected_concept_key, {})
            st.markdown(
                f"""
                **Gekoppelde Vertalingen:**
                * 🇬🇧 Engels: `{mapping.get('en')}`
                * 🇳🇱 Nederlands: `{mapping.get('nl')}`
                * 🇫🇷 Frans (gesimuleerd): `{mapping.get('fr')}`
                """
            )

            if st.button(f"🔍 Zoek '{mapping.get('en')}, {mapping.get('nl')}' in de Visual Builder"):
                st.session_state["search_query"] = f"{mapping.get('en')}, {mapping.get('nl')}"
                st.session_state["selected_regions"] = ["American English", "Dutch"]
                st.rerun()

        with cl_col2:
            st.subheader("Vergelijkende Tijdlijn: Engels vs. Nederlands")
            en_term = mapping.get("en")
            nl_term = mapping.get("nl")

            df_en = synthetic_df[
                (synthetic_df["ngram"] == en_term)
                & (synthetic_df["corpus_region"].isin(["American English", "British English"]))
                & (synthetic_df["pos_tag"] == "NONE")
                & (~synthetic_df["case_sensitive"])
            ].copy()
            df_en["Taal & Corpus"] = f"{en_term} [Engels - American/British]"

            df_nl = synthetic_df[
                (synthetic_df["ngram"] == nl_term)
                & (synthetic_df["corpus_region"] == "Dutch")
                & (synthetic_df["pos_tag"] == "NONE")
                & (~synthetic_df["case_sensitive"])
            ].copy()
            df_nl["Taal & Corpus"] = f"{nl_term} [Nederlands - Dutch Corpus]"

            combined_cross = pd.concat([df_en, df_nl], ignore_index=True)
            if not combined_cross.empty:
                combined_cross["smoothed_frequency"] = combined_cross.groupby("Taal & Corpus")[
                    "relative_frequency"
                ].transform(lambda s: s.rolling(window=5, min_periods=1, center=True).mean())

                cross_chart = (
                    alt.Chart(combined_cross)
                    .mark_line(strokeWidth=2.8, interpolate="monotone")
                    .encode(
                        x=alt.X("year:Q", title="Jaartal", axis=alt.Axis(format="d")),
                        y=alt.Y("smoothed_frequency:Q", title="Relatieve frequentie (%)"),
                        color=alt.Color("Taal & Corpus:N", scale=alt.Scale(scheme="set1")),
                        tooltip=[
                            alt.Tooltip("year:Q", title="Jaar"),
                            alt.Tooltip("ngram:N", title="Woord"),
                            alt.Tooltip("Taal & Corpus:N", title="Corpus"),
                            alt.Tooltip("smoothed_frequency:Q", title="Frequentie (%)", format=".4f"),
                            alt.Tooltip("raw_match_count:Q", title="Treffers", format=","),
                        ],
                    )
                    .properties(height=350)
                    .interactive()
                )
                st.altair_chart(cross_chart, use_container_width=True)
                st.caption(
                    "💡 *Percentages zijn genormaliseerd naar het totale boekvolume van het eigen taalgebied, "
                    "waardoor trends direct vergelijkbaar zijn ondanks grote verschillen in absolute boekproductie.*"
                )

    # ==========================================================================
    # TAB 4: KENNISBANK & DOCUMENTATIE (VERBETERING 2: CONCREET & OPGEDEELD)
    # ==========================================================================
    with tab_documentation:
        st.header("📖 Kennisbank & Documentatie: Big Data Design & UX")
        st.markdown(
            "Deze documentatie legt de theoretische achtergrond, wiskundige formules "
            "en ontwerpprincipes uit van de vernieuwde Ngram Viewer."
        )

        doc_c1, doc_c2 = st.columns(2)

        with doc_c1:
            with st.container(border=True):
                st.subheader("1. Waarom een Herontwerp? (Probleemanalyse)")
                st.markdown(
                    """
                    De klassieke Google Books Ngram Viewer kende drie grote tekortkomingen:
                    1. **Recall over Recognition:** Gebruikers moesten cryptische programmeersyntax uit het hoofd kennen
                       (zoals `climate_NOUN:eng_us_2019`). Een typefout leidde tot een fatale crash.
                    2. **Dead-End Foutmeldingen:** De melding `'No valid ngrams to plot!'` gaf geen suggesties of herstelpad.
                    3. **Gebrek aan Data-Transparantie:** Er werden alleen abstracte percentages getoond zonder inzicht
                       in het werkelijke aantal boeken of absolute vermeldingen.
                    """
                )

            with st.container(border=True):
                st.subheader("2. Het Datamodel & Wiskundige Normalisatie")
                st.markdown(
                    r"""
                    Omdat het aantal gepubliceerde boeken sinds 1800 exponentieel is gegroeid (van ~120.000 naar miljoenen per jaar),
                    zouden absolute aantallen een vertekend beeld geven. Daarom berekent het systeem:

                    $$\text{relative\_frequency} = \left(\frac{\text{raw\_match\_count}}{\text{total\_books\_in\_corpus}}\right) \times 100$$

                    * **raw_match_count:** Exacte aantal keren dat het ngram in dat jaar is aangetroffen.
                    * **total_books_in_corpus:** Totaal publicatievolume in de gekozen regio.
                    """
                )

            with st.container(border=True):
                st.subheader("3. Woordsoorten (POS-tags) & Grammaticale Filtering")
                st.markdown(
                    """
                    In plaats van regex of operatoren gebruikt het prototype visuele categorieën:
                    * **Zelfstandig naamwoord (`_NOUN`):** Filtert op begrippen en entiteiten (bijv. 'het klimaat').
                    * **Werkwoord vervoegd (`_VERB`):** Filtert op handelingen (bijv. 'democratiseert').
                    * **Infinitief (`_INF`):** Hele werkwoorden.
                    * **Geen filter (`NONE`):** Alle instanties geaggregeerd.
                    """
                )

        with doc_c2:
            with st.container(border=True):
                st.subheader("4. Smoothing Algoritme (Moving Average)")
                st.markdown(
                    """
                    Historische data bevat natuurlijke jaarlijkse ruis (oorlogen, uitgeverscycli).
                    De smoothing slider past een gecentreerd voortschrijdend gemiddelde toe:

                    * **Venster = 0:** Onbewerkte jaarcijfers inclusief scherpe fluctuaties.
                    * **Venster = 3-10:** Optimale balans tussen historische pieken en leesbaarheid.
                    * **Venster = 20-50:** Onthult grote macro-economische verschuivingen.
                    * **Transparantie-eis:** De onbewerkte data blijft subtiel als gestippelde lijn op de achtergrond zichtbaar!
                    """
                )

            with st.container(border=True):
                st.subheader("5. AWS S3 Open Data Architectuur (~2,2 TB)")
                st.markdown(
                    """
                    De ruwe dataset bevindt zich op `s3://datasets.elasticmapreduce/ngrams/books/`:
                    * Bevat honderden gecomprimeerde Hadoop SequenceFiles en shards.
                    * Het prototype gebruikt een **drie-laags architectuur**:
                      1. *S3 Bucket Explorer* om de bestandsstructuur op AWS Open Data inzichtelijk te maken.
                      2. *Live Ngram API* om elk willekeurig woord binnen milliseconden op te vragen.
                      3. *Synthetische dataset* voor deterministische validatietests.
                    """
                )

            with st.container(border=True):
                st.subheader("6. Veelgestelde Vragen (FAQ)")
                with st.expander("Waarom verschilt 'Paris' van 'paris'?"):
                    st.write("Met hoofdlettergevoeligheid AAN toont 'Paris' de hoofdstad van Frankrijk. 'paris' met kleine letters komt veel minder voor (vaak typefouten of plantennamen).")
                with st.expander("Hoe werkt de meertalige koppeling?"):
                    st.write("Het systeem koppelt concepten semantisch. Wanneer je meertalig zoekt, haalt het voor het Nederlandse corpus de Nederlandse term op en voor het Engelse corpus de Engelse term.")

    # ==========================================================================
    # TAB 5: ACCOUNT, GESCHIEDENIS & INSTELLINGEN (VERBETERING 6)
    # ==========================================================================
    with tab_account:
        st.header("👤 Gebruikersaccount, Zoekgeschiedenis & Voorkeuren")
        st.markdown(
            "Beheer hier je profiel, herbekijk eerdere zoekopdrachten, beheer je favoriete grafieken "
            "en stel je persoonlijke standaardwaarden in."
        )

        acc_c1, acc_c2 = st.columns([2.0, 3.0])

        with acc_c1:
            st.subheader("1. Profielbeheer")
            all_profs = list(profiles_data["profiles"].keys())
            c_idx = all_profs.index(active_user)
            chosen_profile = st.selectbox("Actief profiel:", options=all_profs, index=c_idx, key="acc_prof_sel")
            if chosen_profile != active_user:
                st.session_state["current_profile"] = chosen_profile
                st.rerun()

            with st.popover("➕ Nieuw Profiel Aanmaken"):
                new_p_name = st.text_input("Naam van nieuw profiel:")
                if st.button("Profiel Opslaan"):
                    if new_p_name and new_p_name not in profiles_data["profiles"]:
                        profiles_data["profiles"][new_p_name] = {
                            "preferences": user_prefs,
                            "history": [],
                            "favorites": [],
                        }
                        profiles_data["active_profile"] = new_p_name
                        save_user_profiles(profiles_data)
                        st.session_state["current_profile"] = new_p_name
                        st.success(f"Profiel '{new_p_name}' aangemaakt!")
                        st.rerun()

            st.divider()
            st.subheader("2. Persoonlijke Standaard-Voorkeuren")
            st.write(f"Standaard instellingen voor **{active_user}**:")
            st.write(f"* Voorkeurs-corpora: `{user_prefs.get('regions')}`")
            st.write(f"* Voorkeurs-grafiektype: `{user_prefs.get('chart_type')}`")
            st.write(f"* Voorkeurs-smoothing: `{user_prefs.get('smoothing')} jaar`")

            save_pref_col, reset_pref_col = st.columns(2)
            with save_pref_col:
                if st.button("💾 Sla Filters Op Als Standaard"):
                    profiles_data["profiles"][active_user]["preferences"] = {
                        "regions": st.session_state["selected_regions"],
                        "chart_type": st.session_state["chart_type"],
                        "interpolation": st.session_state["interpolation"],
                        "palette": st.session_state["palette"],
                        "y_scale": st.session_state["y_scale"],
                        "smoothing": st.session_state["smoothing_val"],
                        "year_range": list(st.session_state["year_range"]),
                        "show_raw": st.session_state["show_raw"],
                        "show_events": st.session_state["show_events"],
                        "show_points": st.session_state["show_points"],
                        "stroke_width": st.session_state["stroke_width"],
                        "cross_lang_mode": st.session_state["cross_lang_mode"],
                    }
                    save_user_profiles(profiles_data)
                    st.success("Standaardvoorkeuren bijgewerkt!")
            with reset_pref_col:
                if st.button("↺ Herstel Standaardwaarden"):
                    prefs = profiles_data["profiles"][active_user]["preferences"]
                    st.session_state["selected_regions"] = prefs.get("regions", ["American English"])
                    st.session_state["chart_type"] = prefs.get("chart_type", "Lijngrafiek")
                    st.session_state["smoothing_val"] = prefs.get("smoothing", 3)
                    st.session_state["year_range"] = tuple(prefs.get("year_range", [1800, 2026]))
                    st.success("Standaardwaarden hersteld!")
                    st.rerun()

        with acc_c2:
            st.subheader("3. ⭐ Opgeslagen Favorieten (Bookmarks)")
            fav_items = profiles_data["profiles"][active_user].get("favorites", [])
            if not fav_items:
                st.info("Nog geen opgeslagen favorieten. Klik op '⭐ Bewaar Zoekopdracht' onder de grafiek.")
            else:
                for f_idx, item in enumerate(fav_items):
                    with st.container(border=True):
                        fc1, fc2, fc3 = st.columns([2.5, 1.5, 1.0])
                        with fc1:
                            st.write(f"**{item.get('label', item.get('query'))}**")
                            st.caption(f"Term: `{item.get('query')}` • Corpora: {item.get('regions')} • Jaren: {item.get('year_range')}")
                        with fc2:
                            st.caption(f"Opgeslagen: {item.get('timestamp')}")
                        with fc3:
                            if st.button("🔄 Laad", key=f"load_fav_btn_{f_idx}"):
                                st.session_state["search_query"] = item.get("query")
                                st.session_state["selected_regions"] = item.get("regions")
                                st.session_state["year_range"] = tuple(item.get("year_range"))
                                st.session_state["smoothing_val"] = item.get("smoothing", 3)
                                st.rerun()

            st.divider()
            st.subheader("4. 🕒 Recente Zoekgeschiedenis")
            history_items = profiles_data["profiles"][active_user].get("history", [])
            if not history_items:
                st.info("Nog geen zoekgeschiedenis voor dit profiel.")
            else:
                c_clear, _ = st.columns([1.5, 3.5])
                with c_clear:
                    if st.button("🗑️ Wis Zoekgeschiedenis"):
                        profiles_data["profiles"][active_user]["history"] = []
                        save_user_profiles(profiles_data)
                        st.success("Geschiedenis gewist!")
                        st.rerun()

                for h_idx, h_entry in enumerate(history_items[:10]):
                    with st.container(border=True):
                        hc1, hc2, hc3 = st.columns([2.5, 1.5, 1.0])
                        with hc1:
                            st.write(f"🔍 **{h_entry.get('query')}**")
                            st.caption(f"Corpora: {', '.join(h_entry.get('regions', []))} • Periode: {h_entry.get('year_range', [1800, 2026])}")
                        with hc2:
                            st.caption(f"Tijd: {h_entry.get('timestamp')}")
                        with hc3:
                            if st.button("🔄 Herhaal", key=f"replay_hist_btn_{h_idx}"):
                                st.session_state["search_query"] = h_entry.get("query")
                                st.session_state["selected_regions"] = h_entry.get("regions", ["American English"])
                                st.session_state["year_range"] = tuple(h_entry.get("year_range", [1800, 2026]))
                                st.session_state["smoothing_val"] = h_entry.get("smoothing", 3)
                                st.rerun()


if __name__ == "__main__":
    main()
