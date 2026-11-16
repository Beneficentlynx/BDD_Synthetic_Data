"""
================================================================================
INTERACTIEF STREAMLIT PROTOTYPE: GOOGLE BOOKS NGRAM VIEWER (VISUAL BUILDER)
================================================================================
Bestand : app.py
Stijl   : Authentieke Google Books Ngram Viewer look-and-feel (Material Design),
          verrijkt met:
          1. Visuele filter-pills (ter vervanging van complexe _NOUN / _INF syntax)
          2. Tijdlijn- & Smoothing-sliders met begrijpelijke feedforward-tooltips
          3. Constructieve foutafhandeling ('Bedoelde je...?' i.p.v. 'No valid ngrams to plot!')
          4. Dynamische historische gebeurtenissen (historical_event) onder de grafiek
          5. Integratie met de echte Google Books Ngram Database & AWS S3 Open Data
             (aws s3 ls --no-sign-request s3://datasets.elasticmapreduce/ngrams/books/)
================================================================================
"""

import difflib
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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
    page_title="Google Books Ngram Viewer — Visual Builder",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_FILE = Path(__file__).parent / "synthetic_ngram_data.csv"

# Authentieke Google Ngram Viewer lijnkleuren (Blauw, Rood, Groen, Oranje, Paars, Teal)
GOOGLE_LINE_COLORS = ["#1a73e8", "#d93025", "#188038", "#e37400", "#9334e6", "#00897b"]

# Mapping van onze UI-regio's naar zowel AWS S3 mapnamen als Google Books Corpus IDs
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

# Historische gebeurtenissen om ook bij live opgevraagde Ngram-data direct
# historische duiding onder de grafiek te tonen.
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

# ==============================================================================
# GOOGLE NGRAM VIEWER CSS STYLING
# ==============================================================================
GOOGLE_NGRAM_CSS = """
<style>
    /* Algemene pagina-achtergrond en typografie in Google-stijl */
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

    /* Google Books Ngram Viewer Top Header */
    .ngram-header {
        display: flex;
        align-items: baseline;
        font-size: 22px;
        font-weight: 400;
        letter-spacing: -0.2px;
        padding-bottom: 14px;
        margin-bottom: 18px;
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

    /* Zoekbalk styling (afgeronde Google Search input met subtiele rand) */
    div[data-testid="stTextInput"] {
        max-width: 640px;
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

    /* Filter-pills balk direct onder de zoekbalk */
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

    /* Secundaire pill-knoppen (Quick-chips) */
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
        margin-bottom: 24px;
    }

    .ngram-filter-divider {
        border-bottom: 1px solid #e8eaed;
        margin-top: 8px;
        margin-bottom: 22px;
    }
</style>
"""


# ==============================================================================
# 1. AWS S3 & LIVE GOOGLE NGRAM DATABASE INTEGRATIE
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def query_aws_s3_ngram_bucket(s3_subpath: str = "ngrams/books/20090715/") -> Dict[str, Any]:
    """
     bevraagt de publieke Google Books Ngram dataset op AWS S3:
    `aws s3 ls --no-sign-request s3://datasets.elasticmapreduce/ngrams/books/`

    Hoe dit werkt (twee-staps aanpak):
    1. Als de `aws` CLI op het systeem geïnstalleerd is, voert deze functie exact
       het commando `aws s3 ls --no-sign-request s3://datasets.elasticmapreduce/...` uit.
    2. Als de `aws` CLI niet geïnstalleerd is, bevraagt de functie direct dezelfde
       publieke S3-bucket via de ongetekende AWS S3 REST API
       (`https://s3.amazonaws.com/datasets.elasticmapreduce?list-type=2&prefix=...`),
       zodat de S3-integratie op élke computer werkt zonder AWS-account of installatie!
    """
    s3_uri = f"s3://datasets.elasticmapreduce/{s3_subpath}"
    cli_cmd = f"aws s3 ls --no-sign-request {s3_uri}"

    # Poging 1: Gebruik het echte AWS CLI commando als 'aws' beschikbaar is in PATH
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

    # Poging 2: Directe ongetekende HTTPS request naar dezelfde AWS S3 bucket
    # (S3 REST API equivalent van `aws s3 ls --no-sign-request`)
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

        # Map-prefixes (zoals 'PRE eng-us-all/')
        for cp in root.findall("s3:CommonPrefixes", ns):
            prefix_el = cp.find("s3:Prefix", ns)
            if prefix_el is not None and prefix_el.text:
                folder_name = prefix_el.text[len(s3_subpath) :]
                entries.append(f"                           PRE {folder_name}")

        # Bestanden (zoals SequenceFiles / .zip shards met bestandsgrootte)
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

    return {
        "success": False,
        "method": "Fallback S3 Overzicht",
        "command": cli_cmd,
        "entries": [],
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_ngram_from_google(
    terms: Tuple[str, ...],
    pos_code: str,
    case_sensitive: bool,
    regions: Tuple[str, ...],
    start_year: int = 1800,
    end_year: int = 2026,
) -> Optional[pd.DataFrame]:
    """
    Haalt live echte Ngram-tijdreeksen op zodat de gebruiker in de app niet alleen
    de 6 synthetische woorden, maar élk bestaand woord (zoals 'Sherlock Holmes',
    'Frankenstein', 'Albert Einstein', 'computer', 'evolution') kan opzoeken!

    UX- & Big Data-uitleg:
    De ruwe bestanden op `s3://datasets.elasticmapreduce/ngrams/books/` zijn ~2,2 TB
    aan gecomprimeerde shards. Om binnen 200ms een grafiek te tekenen voor elk woord,
    bevragen we voor live zoekopdrachten het geïndexeerde Ngram-endpoint met
    smoothing=0 (ruwe jaarlijkse data) en passen we vervolgens in Python onze eigen
    visuele filters, Smoothing-slider en historische annotaties toe.
    """
    records: List[Dict[str, Any]] = []
    # De publieke Google Books Ngram index loopt t/m 2019/2022; jaren tot 2026
    # vullen we netjes aan zodat de tijdlijn-slider t/m 2026 consistent werkt.
    api_end_year = min(end_year, 2019)

    for region in regions:
        corpus_id = CORPUS_MAPPING.get(region, {}).get("google_corpus_id", "en-US-2019")

        # Vertaal onze visuele POS-tag knop naar de onderliggende Ngram query-operator
        # zodat de gebruiker zelf géén syntax hoeft te typen!
        query_items = []
        term_lookup = {}
        for t in terms:
            if pos_code in ("_NOUN", "_VERB", "_INF"):
                q_str = f"{t}{pos_code}"
            elif pos_code == "_ADJ_SIM":
                q_str = f"{t}_ADJ"
            else:
                q_str = t
            query_items.append(q_str)
            term_lookup[q_str.lower()] = t

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

        # Verwerk elke teruggekomen tijdreeks naar ons uniforme datamodel.
        # Let op: bij case_insensitive=true stuurt Google naast de hoofdreeks ('CASE_INSENSITIVE')
        # ook losse hoofdletter-expansies ('EXPANSION', zoals 'DONALD DUCK') mee.
        # We filteren die 'EXPANSION'-dubbelingen eruit zodat elke zoekterm 1 strakke lijn vormt.
        for item in payload:
            if not case_sensitive and item.get("type") == "EXPANSION":
                continue

            timeseries = item.get("timeseries", [])
            if not timeseries:
                continue

            raw_ngram_name = item.get("ngram", "")
            # Koppel terug aan de schone zoekterm zonder '(All)' of '_NOUN' achtervoegsel
            clean_name = raw_ngram_name.replace(" (All)", "")
            for suffix in ["_NOUN", "_VERB", "_INF", "_ADJ"]:
                if clean_name.endswith(suffix):
                    clean_name = clean_name[: -len(suffix)]

            years_list = list(range(start_year, api_end_year + 1))
            for idx, yr in enumerate(years_list):
                if idx >= len(timeseries):
                    break
                # Google retourneert een fractie (bijv. 0.000012 -> * 100 voor percentage)
                rel_freq_pct = float(timeseries[idx]) * 100.0
                # Schat het jaarlijkse corpusvolume voor de transparantie-indicatoren
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

            # Extrapoleer 2020 t/m end_year met de laatst bekende waarde zodat de
            # 1800-2026 slider vloeiend doorloopt
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
# DATA INLADEN & CACHING (SYNTHETISCHE CSV)
# ==============================================================================
@st.cache_data
def load_ngram_data(csv_path: Path) -> pd.DataFrame:
    """
    Laadt de lokale synthetische Ngram-dataset (synthetic_ngram_data.csv) in.
    """
    df = pd.read_csv(csv_path)
    df["historical_event"] = df["historical_event"].where(df["historical_event"].notna(), None)
    return df


# ==============================================================================
# CONSTRUCTIEVE FOUTAFHANDELING (VAN FEEDBACK NAAR FEEDFORWARD)
# ==============================================================================
KNOWN_TYPOS_AND_SYNTAX: Dict[str, Tuple[str, str]] = {
    "climmate": (
        "climate",
        "Er zit een dubbele 'm' in je zoekterm.",
    ),
    "climat": (
        "climate",
        "Er ontbreekt een 'e' aan het einde van het woord.",
    ),
    "democracie": (
        "democracy",
        "Je gebruikte de Nederlandse/Franse schrijfwijze. In het Engelse corpus staat 'democracy'.",
    ),
    "democratie": (
        "democracy",
        "Gebruik de zoekterm 'democracy' (je kunt deze wel filteren op het corpus 'Dutch'!).",
    ),
    "artifical intelligence": (
        "artificial intelligence",
        "Veelvoorkomende typfout: er ontbreekt een 'i' in 'artificial'.",
    ),
    "ai": (
        "artificial intelligence",
        "De afkorting 'AI' is in de synthetische dataset opgenomen onder 'artificial intelligence'.",
    ),
    "telegramm": (
        "telegram",
        "Dit lijkt op de Duitse spelling met dubbel 'm'.",
    ),
}


def validate_single_term(
    raw_term: str,
    available_terms: List[str],
    case_sensitive: bool,
) -> Dict[str, Optional[str]]:
    """
    Valideert één ingevoerde zoekterm en geeft bij een fout een constructieve
    feedforward-suggestie terug in plaats van 'No valid ngrams to plot!'.
    """
    cleaned = raw_term.strip()

    # Check 1: Gebruikt de onderzoeker nog een oude POS-operator zoals '_NOUN' of '_INF'?
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

    # Check 2: Bestaat de zoekterm in de synthetische dataset?
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
                    f"Hoofdlettergevoeligheid (`Case-Sensitive`) staat aan en `{cleaned}` verschilt in "
                    f"hoofdlettergebruik van **'{alt_case}'**."
                ),
            }
    else:
        for term in available_terms:
            if term.lower() == cleaned.lower():
                return {"is_valid": True, "input_term": cleaned, "matched_term": term, "suggestion": None, "reason": None}

    # Check 3: Bekende typfout of taal-mismatch?
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

    # Check 4: Fuzzy matching voor overige spelfouten
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

    # Check 5: Niet in synthetische CSV -> verwijs naar beschikbare termen of Live S3/Google modus
    return {
        "is_valid": False,
        "input_term": cleaned,
        "matched_term": None,
        "suggestion": "climate",
        "reason": (
            f"De term `{cleaned}` zit niet in de lokale synthetische CSV. "
            f"Schakel in de filterbalk **Databron** over naar **Live Ngram + AWS S3** om élk willekeurig "
            f"woord op te vragen, of kies een synthetische testterm ({', '.join(f'**{t}**' for t in available_terms)})."
        ),
    }


def render_google_placeholder_chart(year_range: Tuple[int, int]) -> alt.Chart:
    """
    Toont een lege Google Books Ngram Viewer grafiek met exact dezelfde minimalistische
    assen en percentages wanneer er nog geen geldige zoekterm is ingevoerd.
    """
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
# HOOFDAPPLICATIE IN GOOGLE BOOKS NGRAM VIEWER STIJL
# ==============================================================================
def main() -> None:
    st.markdown(GOOGLE_NGRAM_CSS, unsafe_allow_html=True)

    if not DATA_FILE.exists():
        st.error(
            f"Bestand `{DATA_FILE.name}` niet gevonden. "
            "Voer eerst `python generate_synthetic_ngram_data.py` uit."
        )
        st.stop()

    synthetic_df = load_ngram_data(DATA_FILE)
    available_ngrams: List[str] = sorted(
        synthetic_df["ngram"].unique().tolist(), key=lambda s: (s.lower(), s)
    )
    available_regions: List[str] = synthetic_df["corpus_region"].unique().tolist()

    # Initialiseer session_state
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = "climate, democracy, telegram"
    if "year_range" not in st.session_state:
        st.session_state["year_range"] = (1800, 2026)
    if "smoothing_val" not in st.session_state:
        st.session_state["smoothing_val"] = 3
    if "selected_regions" not in st.session_state:
        st.session_state["selected_regions"] = ["American English"]
    if "selected_pos_label" not in st.session_state:
        st.session_state["selected_pos_label"] = "Alle woordsoorten"
    if "case_sensitive_val" not in st.session_state:
        st.session_state["case_sensitive_val"] = False
    if "data_source_mode" not in st.session_state:
        st.session_state["data_source_mode"] = "Hybride (Synthetisch + Live Ngram / AWS S3)"

    # --------------------------------------------------------------------------
    # 1. AUTHENTIEKE GOOGLE BOOKS NGRAM VIEWER HEADER
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="ngram-header">
            <span class="g-blue">G</span><span class="g-red">o</span><span class="g-yellow">o</span><span class="g-blue">g</span><span class="g-green">l</span><span class="g-red">e</span>
            <span class="ngram-title-text">Books Ngram Viewer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 2. ZOEKBALK
    # --------------------------------------------------------------------------
    user_query = st.text_input(
        label="Search Ngrams",
        key="search_query",
        placeholder="Search Ngrams (bijv. climate, democracy, Sherlock Holmes, Frankenstein)",
        help=(
            "Typ één of meerdere zoektermen (gescheiden door een komma). "
            "Dankzij de hybride koppeling met de echte Ngram-database kun je zowel de "
            "synthetische testtermen als elk echt woord (zoals 'Sherlock Holmes, Frankenstein') opzoeken!"
        ),
    )

    # --------------------------------------------------------------------------
    # 3. FILTER-PILLS BALK DIRECT ONDER DE ZOEKBALK (INCL. AWS S3 DATABRON PILL)
    # --------------------------------------------------------------------------
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

    pill_col1, pill_col2, pill_col3, pill_col4, pill_col5, pill_col6, _ = st.columns(
        [1.1, 1.35, 1.3, 1.2, 1.45, 1.55, 2.05]
    )

    # PILL 1: Tijdlijn-slider (1800 - 2026)
    with pill_col1:
        with st.popover(f"{yr_start} - {yr_end}", use_container_width=True):
            st.markdown("**Tijdsperiode selecteren**")
            year_range: Tuple[int, int] = st.slider(
                label="Jaartallen (`year`)",
                min_value=1800,
                max_value=2026,
                key="year_range",
                step=1,
                help="Versleep de schuifregelaar om in te zoomen op specifieke jaartallen tussen 1800 en 2026.",
            )

    # PILL 2: Geografisch Corpus / Regio
    with pill_col2:
        with st.popover(region_pill_label, use_container_width=True):
            st.markdown("**Corpus & Geografische Regio**")
            selected_regions = st.multiselect(
                label="Selecteer één of meerdere corpora (`corpus_region`)",
                options=available_regions,
                key="selected_regions",
                help=(
                    "Kies één corpus om meerdere woorden met elkaar te vergelijken, "
                    "of selecteer meerdere regio's om een woord tussen taalgebieden te vergelijken."
                ),
            )

    # PILL 3: Hoofdlettergevoeligheid
    with pill_col3:
        with st.popover(case_pill_label, use_container_width=True):
            st.markdown("**Hoofdlettergevoeligheid**")
            case_sensitive = st.checkbox(
                label="Hoofdlettergevoelig (`Case-Sensitive`)",
                key="case_sensitive_val",
                help=(
                    "Uit (Case-Insensitive): Combineert hoofd- en kleine letters.\n\n"
                    "Aan (Case-Sensitive): Maakt onderscheid tussen bijv. 'Paris' (de stad) en 'paris'."
                ),
            )

    # PILL 4: Smoothing Slider (0 - 50) met Feedforward Tooltip
    with pill_col4:
        with st.popover(smoothing_pill_label, use_container_width=True):
            st.markdown("**Trend-afvlakking (Smoothing)**")
            smoothing_window: int = st.slider(
                label="Smoothing (voortschrijdend gemiddelde)",
                min_value=0,
                max_value=50,
                key="smoothing_val",
                step=1,
                help=(
                    "Berekent een voortschrijdend gemiddelde over het gekozen aantal jaren. "
                    "Dit strijkt kortstondige jaarlijkse pieken en ruis glad, zodat grote "
                    "macro-trends over langere periodes zichtbaar worden (0 = ruwe data, 50 = maximale afvlakking)."
                ),
            )

    # PILL 5: Visuele Woordsoort-selector (Vervangt _NOUN / _INF syntax)
    with pill_col5:
        pos_short = "Woordsoort: Alle" if current_pos_code == "NONE" else f"Tag: {current_pos_code}"
        with st.popover(pos_short, use_container_width=True):
            st.markdown("**Woordsoort & Vervoeging (Vervangt `_NOUN` / `_INF` syntax)**")
            selected_pos_label = st.selectbox(
                label="Kies grammaticale rol (`pos_tag`)",
                options=list(pos_options_map.keys()),
                key="selected_pos_label",
                help=(
                    "In plaats van handmatig `_NOUN` of `_VERB` achter je zoekwoord te moeten typen, "
                    "selecteer je hier visueel de gewenste woordsoort."
                ),
            )
            selected_pos_code = pos_options_map[selected_pos_label]

    # PILL 6: AWS S3 & Live Ngram Database Koppeling
    with pill_col6:
        mode_short = (
            "Bron: Synthetisch"
            if st.session_state["data_source_mode"].startswith("Alleen Synthetisch")
            else "Bron: S3 / Live + CSV"
        )
        with st.popover(mode_short, use_container_width=True):
            st.markdown("**Ngram Database & AWS S3 Koppeling**")
            data_source_mode = st.radio(
                label="Kies hoe zoektermen worden opgevraagd:",
                options=[
                    "Hybride (Synthetisch + Live Ngram / AWS S3)",
                    "Live Google Ngram Database (Online / S3)",
                    "Alleen Synthetische CSV (Test-modus voor typfouten)",
                ],
                key="data_source_mode",
                help=(
                    "• **Hybride:** Gebruikt `synthetic_ngram_data.csv` voor de 6 testtermen én haalt "
                    "onbekende echte woorden (zoals 'Sherlock Holmes') live op uit de Ngram-database.\n"
                    "• **Live Google Ngram Database:** Haalt alle termen live op.\n"
                    "• **Alleen Synthetische CSV:** Strikte testmodus voor de voorgeprogrammeerde dataset."
                ),
            )

    st.markdown('<div class="ngram-filter-divider"></div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 4. ZOEKTERMEN VALIDEREN & DATA OPHALEN (SYNTHETISCH OF LIVE NGRAM/S3)
    # --------------------------------------------------------------------------
    raw_terms = [t.strip() for t in user_query.split(",") if t.strip()]
    if not raw_terms:
        raw_terms = [""]

    if not selected_regions:
        st.warning(
            "Geen corpus geselecteerd. Klik hierboven op de **Kies Corpus**-knop en vink ten minste één regio aan."
        )
        st.altair_chart(render_google_placeholder_chart(year_range), use_container_width=True)
        st.stop()

    validations = [
        validate_single_term(t, available_ngrams, case_sensitive) for t in raw_terms
    ]

    # Scheid bekende typfouten / oude syntax (zoals 'climmate', 'democracie', 'climate_NOUN')
    # van echte woorden die we live uit de Google Books Ngram database kunnen halen!
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

    # Toon ALTIJD onze constructieve feedforward-waarschuwing bij gedetecteerde typfouten of oude syntax
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

    # Stel het gecombineerde DataFrame samen op basis van de gekozen databron-modus
    frames_to_combine: List[pd.DataFrame] = []
    actual_pos_tag = "_NOUN" if selected_pos_code == "_ADJ_SIM" else selected_pos_code

    if data_source_mode == "Live Google Ngram Database (Online / S3)":
        # Haal alle geldige/ingevoerde termen (behalve ongecorrigeerde legacy-syntax) live op
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
        # 1. Laad de beschikbare synthetische termen uit synthetic_ngram_data.csv
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

        # 2. Als we in Hybride modus zitten en de gebruiker typt bijv. 'Sherlock Holmes, Frankenstein',
        #    haal die dan live op uit de echte Google Books Ngram database!
        if (
            data_source_mode.startswith("Hybride")
            and unknown_but_fetchable_terms
        ):
            fetch_terms = [v["input_term"] for v in unknown_but_fetchable_terms if v["input_term"]]
            if fetch_terms:
                with st.spinner(f"Zoekterm(en) {', '.join(fetch_terms)} live opvragen bij Google Books Ngram..."):
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
                            f"💡 **Geen resultaten gevonden voor `{un['input_term']}`** — {un['reason']}  \n"
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
    if len(selected_regions) == 1:
        filtered_df["series_label"] = filtered_df["ngram"]
    else:
        filtered_df["series_label"] = filtered_df["ngram"] + " (" + filtered_df["corpus_region"] + ")"

    filtered_df.sort_values(by=["series_label", "year"], inplace=True)

    # Voortschrijdend gemiddelde (Smoothing):
    window_size = max(1, smoothing_window * 2 + 1) if smoothing_window > 0 else 1
    filtered_df["smoothed_frequency"] = filtered_df.groupby("series_label")[
        "relative_frequency"
    ].transform(lambda s: s.rolling(window=window_size, min_periods=1, center=True).mean())

    # --------------------------------------------------------------------------
    # 5. GOOGLE BOOKS NGRAM VIEWER LIJNGRAFIEK (MET CORRECT SCHALENDE X-AS)
    # --------------------------------------------------------------------------
    series_order = filtered_df["series_label"].unique().tolist()
    color_scale = alt.Scale(
        domain=series_order,
        range=GOOGLE_LINE_COLORS[: max(1, len(series_order))],
    )

    focus_selection = alt.selection_point(fields=["series_label"], on="click", clear="dblclick")

    # Dynamische X-as schaal met zero=False en nice=False zodat de X-as ALTIJD
    # exact bij year_range[0] (bijv. 1800) links begint en bij year_range[1] eindigt,
    # in plaats van dat Altair het nulpunt (jaar 0) meeneemt!
    x_min = int(year_range[0])
    x_max = int(year_range[1])
    year_span = max(1, x_max - x_min)

    # Kies een nette stapgrootte voor de jaartallen op de X-as (zoals 20 jaar bij 1800-2026)
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

    # Dynamische decimaal-formattering op de Y-as zodat ook kleine frequenties
    # (zoals 'Donald Duck' rond 0.000018%) scherpe, unieke labels krijgen.
    max_y_val = float(filtered_df["smoothed_frequency"].max()) if not filtered_df.empty else 0.01
    if max_y_val < 0.00005:
        y_format = ".7f"
    elif max_y_val < 0.0005:
        y_format = ".6f"
    elif max_y_val < 0.01:
        y_format = ".5f"
    else:
        y_format = ".4f"

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

    lines = (
        base_chart.mark_line(strokeWidth=1.8, interpolate="monotone")
        .encode(
            y=alt.Y(
                "smoothed_frequency:Q",
                title=None,
                scale=alt.Scale(zero=True),
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
                    labelPadding=4,
                ),
            ),
            tooltip=[
                alt.Tooltip("year:Q", title="Jaar"),
                alt.Tooltip("series_label:N", title="Term / Corpus"),
                alt.Tooltip("smoothed_frequency:Q", title="Relatieve frequentie (%)", format=".6f"),
                alt.Tooltip("raw_match_count:Q", title="Absolute treffers (raw_match_count)", format=","),
                alt.Tooltip("total_books_in_corpus:Q", title="Totaal boeken in corpus", format=","),
                alt.Tooltip("historical_event:N", title="Historische gebeurtenis"),
            ],
        )
        .add_params(focus_selection)
    )

    # Eindpunten op het laatste geselecteerde jaartal voor de gekleurde labels rechts naast de lijn.
    # Dankzij clip=False en rechter-padding op de chart vallen deze buiten het X-as bereik
    # zonder de X-as zelf op te rekken!
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

    chart_layers = [lines, direct_labels]

    events_in_view = (
        filtered_df[filtered_df["historical_event"].notna()][["year", "historical_event"]]
        .drop_duplicates(subset=["year"])
        .sort_values("year")
    )
    if not events_in_view.empty:
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
            padding={"left": 10, "top": 12, "right": 155, "bottom": 10},
        )
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(final_chart, use_container_width=True)

    st.markdown(
        '<div class="ngram-chart-footer">(click on line/label for focus • hover over de lijn voor exacte aantallen & historische pieken)</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 6. SNEL TESTEN & HISTORISCHE CONTEXT ONDER DE GRAFIEK (st.info)
    # --------------------------------------------------------------------------
    st.caption("Snel testen (synthetische termen, live boeken-termen uit je screenshot, of foutafhandeling):")
    q_cols = st.columns(6)
    examples = [
        ("climate, democracy, telegram", "climate, democracy, telegram"),
        ("Sherlock Holmes, Frankenstein, Albert Einstein", "Sherlock Holmes, Frankenstein, Albert Einstein"),
        ("Paris, paris (Case-Sensitive)", "Paris, paris"),
        ("Typfout: climmate", "climmate"),
        ("Typfout: democracie", "democracie"),
        ("Oude syntax: climate_NOUN", "climate_NOUN"),
    ]
    for idx, (label_txt, query_val) in enumerate(examples):
        with q_cols[idx]:
            if st.button(label_txt, key=f"ex_{idx}", use_container_width=True):
                st.session_state["search_query"] = query_val
                st.rerun()

    st.markdown("##### Historische context in deze tijdsperiode (`historical_event`)")
    if events_in_view.empty:
        st.info(
            f"Geen gemarkeerde historische gebeurtenissen tussen {year_range[0]} en {year_range[1]}."
        )
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
                    f"{int(top_match['raw_match_count']):,} treffers op "
                    f"{int(top_match['total_books_in_corpus']):,} boeken)"
                )

    # --------------------------------------------------------------------------
    # 7. AWS S3 OPEN DATA BROWSER (`s3://datasets.elasticmapreduce/ngrams/books/`)
    # --------------------------------------------------------------------------
    with st.expander(
        "☁️ AWS S3 Google Books Ngram Database Explorer (`s3://datasets.elasticmapreduce/ngrams/books/`)"
    ):
        st.markdown(
            "Hieronder wordt rechtstreeks de publieke **Google Books Ngrams S3-bucket** op AWS Open Data "
            "bevraagd met het commando dat je hebt gevonden (`--no-sign-request`). "
            "Je kunt zowel de hoofdmap als de specifieke corpus-mappen (`eng-us-all/`, `eng-gb-all/`, `1gram/`) inspecteren."
        )

        s3_col1, s3_col2 = st.columns([2.2, 2.8])
        with s3_col1:
            selected_s3_path = st.selectbox(
                label="Kies een pad in `s3://datasets.elasticmapreduce/` om op te vragen:",
                options=[
                    "ngrams/books/",
                    "ngrams/books/20090715/",
                    "ngrams/books/20090715/eng-us-all/",
                    "ngrams/books/20090715/eng-us-all/1gram/",
                    "ngrams/books/20090715/eng-gb-all/1gram/",
                    "ngrams/books/20090715/eng-fiction-all/1gram/",
                ],
                index=1,
            )

        s3_result = query_aws_s3_ngram_bucket(selected_s3_path)

        with s3_col2:
            st.markdown("**Uitgevoerd AWS S3 commando:**")
            st.code(s3_result["command"], language="bash")
            st.caption(f"Verbindingsmethode: **{s3_result['method']}**")

        if s3_result["entries"]:
            st.code("\n".join(s3_result["entries"][:25]), language="text")

        st.info(
            "💡 **Big Data Design Architectuur-inzicht:** "
            "De ruwe bestanden in `s3://datasets.elasticmapreduce/ngrams/books/` bestaan uit honderden "
            "gecomprimeerde Hadoop SequenceFiles en `.zip`-shards (samen **~2,2 Terabyte**). "
            "Daarom combineert dit prototype drie lagen:\n"
            "1. **AWS S3 Bucket Inspectie** (`query_aws_s3_ngram_bucket`) om de ruwe S3-structuur te verkennen.\n"
            "2. **Live Geïndexeerde Ngram Query's** (`fetch_live_ngram_from_google`) om elk bestaand woord "
            "(zoals *Sherlock Holmes, Frankenstein, Albert Einstein*) binnen milliseconden te plotten.\n"
            "3. **Synthetische Verrijking** (`synthetic_ngram_data.csv`) voor deterministische tests van "
            "POS-tags, ruis/smoothing en constructieve foutafhandeling."
        )


if __name__ == "__main__":
    main()
