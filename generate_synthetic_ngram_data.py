"""
================================================================================
SYNTHETISCHE DATA GENERATOR VOOR DE REDESIGNED NGRAM VIEWER (BIG DATA DESIGN)
================================================================================
Dit script genereert een rijke, synthetische dataset om alle nieuwe UI- en UX-
functionaliteiten van de herontworpen Ngram Viewer te testen.

Ondersteunde UX-functionaliteiten in deze dataset:
1. Tijdlijn-sliders & Zoom: Volledige dekking van 1800 t/m 2026.
2. Visuele POS-tags (Woordsoorten): Vervangt complexe '_NOUN' / '_VERB' syntax
   door toegankelijke UI-checkboxes en tags.
3. Hoofdlettergevoeligheid (Case-Sensitivity toggle): Maakt onderscheid tussen
   bijv. 'Paris' (de stad, hoofdlettergevoelig) en 'paris' (algemeen/kleine letter).
4. Geografische & Corpus Filters: Kruisreferenties tussen 'American English',
   'British English', 'European Fiction' en 'Dutch'.
5. Transparantie & Normalisatie: Bevat zowel absolute aantallen (raw_match_count)
   als het totale publicatievolume (total_books_in_corpus) en het genormaliseerde
   percentage (relative_frequency).
6. Smoothing Slider (Moving Average 0-50): Bevat opzettelijke Gaussische +
   multiplicatieve jaarlijkse ruis bovenop logische basistrends, zodat het
   afvlakken van ruwe fluctuaties visueel direct merkbaar is.
7. Historische Annotaties & Pieken: Koppelt echte historische gebeurtenissen
   (bijv. WOI, WOII, Opkomst Internet) aan kunstmatige frequentiepieken.
8. Constructieve Foutafhandeling (Feedforward): Vervangt de cryptische foutmelding
   'No valid ngrams to plot!' door slimme suggesties bij typfouten en edge cases.
================================================================================
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# ==============================================================================
# 1. CONFIGURATIE VAN HET DATAMODEL & HISTORISCHE EVENTS
# ==============================================================================

# Tijdsbereik voor de X-as en de tijdlijn-range-sliders (1800 t/m 2026)
START_YEAR = 1800
END_YEAR = 2026
YEARS = np.arange(START_YEAR, END_YEAR + 1)

# Zoektermen (ngrams) waarmee verschillende soorten historische trends worden getest
NGRAMS = [
    "climate",
    "democracy",
    "artificial intelligence",
    "telegram",
    "Paris",
    "paris",
]

# Woordsoorten en vervoegingen (POS-tags) voor het testen van de visuele tag-filters
# (ter vervanging van de oude programmeersyntax in het zoekveld)
POS_TAGS = ["NONE", "_NOUN", "_VERB", "_INF"]

# Corpora en geografische regio's voor het testen van kruisreferenties en regioknoppen
CORPUS_REGIONS = [
    "American English",
    "British English",
    "European Fiction",
    "Dutch",
]

# Historische gebeurtenissen gekoppeld aan specifieke jaartallen.
# UX-doel: Testen van interactieve annotatie-markers (tooltips/badges) op de tijdlijn.
HISTORICAL_EVENTS: Dict[int, str] = {
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
# 2. WISKUNDIGE PATRONEN VOOR UX-INTERACTIE (TRENDS, RUIS & PIEKEN)
# ==============================================================================

def generate_corpus_volume(years: np.ndarray, region: str, rng: np.random.Generator) -> np.ndarray:
    """
    Simuleert het totale publicatievolume (total_books_in_corpus) per jaar.

    Wiskundige opbouw:
    - Het aantal gepubliceerde boeken groeit exponentieel van 1800 tot 2026,
      met een kleine dip tijdens de wereldoorlogen (1914-1918 en 1939-1945).
    - Elke regio (corpus_region) heeft een eigen schaalfactor zodat we in de UI
      kunnen demonstreren waarom normalisatie (relative_frequency) essentieel is:
      een kleiner corpus zoals 'Dutch' heeft minder boeken in absolute zin,
      maar kan procentueel een hogere relatieve frequentie vertonen.
    """
    t = (years - START_YEAR) / (END_YEAR - START_YEAR)  # Genormaliseerde tijd [0, 1]

    region_scale = {
        "American English": 1.4,
        "British English": 1.1,
        "European Fiction": 0.75,
        "Dutch": 0.45,
    }.get(region, 1.0)

    # Exponentiële groei van publicatievolume van ~120.000 naar ~4.500.000+ boeken/jaar
    # zodat raw_match_count ook in de 19e eeuw voldoende resolutie heeft voor ruis & smoothing.
    base_volume = 120_000 * np.exp(3.6 * t) * region_scale

    # Oorlogsdips in boekproductie (papiertekort en censuur)
    ww1_dip = 1.0 - 0.18 * np.exp(-0.5 * ((years - 1916) / 2.0) ** 2)
    ww2_dip = 1.0 - 0.25 * np.exp(-0.5 * ((years - 1943) / 2.5) ** 2)

    # Kleine jaarlijkse variatie in uitgeversvolume
    volume_noise = rng.normal(loc=1.0, scale=0.03, size=len(years))
    volumes = base_volume * ww1_dip * ww2_dip * np.clip(volume_noise, 0.9, 1.1)

    return np.round(volumes).astype(int)


def generate_base_trend(years: np.ndarray, ngram: str) -> np.ndarray:
    """
    Berekent de onderliggende 'gladde' basistrend (relatieve frequentie in %)
    voor elke zoekterm over de periode 1800-2026.

    UX-doel:
    Wanneer de gebruiker de Smoothing Slider in de UI op een hoge waarde zet
    (bijv. moving average window = 20 tot 50 jaar), filtert het algoritme de
    jaarlijkse ruis weg en komt deze herkenbare langetermijntrend tevoorschijn.
    """
    if ngram == "telegram":
        # Wiskundig patroon: Log-normale / asymmetrische Gaussische curve.
        # Vrijwel 0 vóór 1844, sterke opkomst in de late 19e eeuw, piek rond 1915-1925,
        # en een gestage daling na 1920 door opkomst telefoon, e-mail en internet.
        rise = 1.0 / (1.0 + np.exp(-(years - 1865) / 8.0))
        decay = np.exp(-np.maximum(0, years - 1920) / 38.0)
        baseline = 0.0005 + 0.038 * rise * decay

    elif ngram == "climate":
        # Wiskundig patroon: Lage stabiele basis in de 19e en vroege 20e eeuw,
        # gevolgd door een steile sigmoïde / exponentiële stijging na 1980.
        modern_surge = 0.065 / (1.0 + np.exp(-(years - 2002) / 9.5))
        early_science = 0.0035 + 0.002 * np.sin((years - 1800) / 30.0)
        baseline = np.maximum(0.001, early_science + modern_surge)

    elif ngram == "democracy":
        # Wiskundig patroon: Geleidelijke opmars sinds 1800 met golfbewegingen
        # rond grote democratiseringsgolven (1918, 1945, 1989, jaren 2010-2026).
        linear_growth = 0.004 + 0.025 * ((years - START_YEAR) / (END_YEAR - START_YEAR))
        wave_1945 = 0.018 * np.exp(-0.5 * ((years - 1945) / 10.0) ** 2)
        wave_1990 = 0.022 * np.exp(-0.5 * ((years - 1992) / 8.0) ** 2)
        wave_2020 = 0.015 * np.exp(-0.5 * ((years - 2020) / 7.0) ** 2)
        baseline = linear_growth + wave_1945 + wave_1990 + wave_2020

    elif ngram == "artificial intelligence":
        # Wiskundig patroon: Vrijwel onbestaand vóór 1956 (Dartmouth-conferentie),
        # eerste 'AI-lente' in jaren '80, gevolgd door een enorme exponentiële
        # explosie na 2012 (Deep Learning) en 2022-2026 (Generative AI).
        ai_winter_spring = 0.008 * np.exp(-0.5 * ((years - 1987) / 5.0) ** 2)
        deep_learning_boom = 0.075 / (1.0 + np.exp(-(years - 2021) / 2.8))
        pre_1955_mask = np.where(years < 1954, 0.00005, 1.0)
        baseline = (0.0002 + ai_winter_spring + deep_learning_boom) * pre_1955_mask

    elif ngram == "Paris":
        # Hoofdlettergevoelig ('Paris' als wereldstad): hoog in de 19e/20e eeuw,
        # met pieken rond wereldtentoonstellingen (1889, 1900), verdragen (1919) en 2015.
        belle_epoque = 0.022 * np.exp(-0.5 * ((years - 1900) / 22.0) ** 2)
        baseline = 0.018 + belle_epoque

    elif ngram == "paris":
        # Kleine letters ('paris' of case-insensitive tegenhanger):
        # veel lagere frequentie wanneer case_sensitive=True (bijv. botanische naam
        # of OCR-fouten), maar bij case_sensitive=False worden 'Paris' + 'paris' gecombineerd.
        baseline = 0.0025 + 0.001 * np.cos((years - 1800) / 25.0)

    else:
        baseline = np.full_like(years, 0.005, dtype=float)

    return baseline


def apply_historical_spikes(years: np.ndarray, trend: np.ndarray, ngram: str) -> np.ndarray:
    """
    Voegt kunstmatige, scherpe pieken toe op specifieke historische jaartallen
    die direct correleren met de kolom 'historical_event'.

    UX-doel:
    Hiermee testen we of een plotselinge uitschieter in de lijngrafiek visueel
    verklaard wordt door een klikbare annotatie op de tijdlijn (bijv. WOI, WOII,
    Klimaatakkoord van Parijs of de doorbraak van AI).
    """
    spiked = trend.copy()

    # Specifieke gebeurtenis-pieken per zoekterm (Gaussian impulses met smalle breedte)
    spike_definitions = {
        "telegram": [
            (1844, 0.012, 1.5),  # Morse demonstratie
            (1861, 0.018, 2.0),  # Amerikaanse Burgeroorlog
            (1914, 0.035, 1.8),  # Start WOI (intensief diplomatiek & militair telegraafverkeer)
            (1918, 0.028, 1.5),  # Einde WOI
            (1939, 0.020, 1.8),  # Start WOII
        ],
        "climate": [
            (1995, 0.015, 1.5),  # Eerste COP-toppen & Internet-verspreiding
            (2015, 0.042, 1.4),  # Klimaatakkoord van Parijs
            (2020, 0.025, 1.2),  # Klimaat- & pandemie-debatten
        ],
        "democracy": [
            (1861, 0.014, 1.8),  # Burgeroorlog
            (1914, 0.018, 1.5),  # Start WOI
            (1918, 0.032, 1.6),  # Einde WOI
            (1939, 0.024, 1.5),  # Start WOII
            (1945, 0.038, 1.6),  # Einde WOII
            (1989, 0.045, 1.4),  # Val van de Berlijnse Muur
        ],
        "artificial intelligence": [
            (1956, 0.012, 1.2),  # Dartmouth Workshop (geboorte van het vakgebied)
            (1995, 0.009, 1.5),  # Opkomst Internet & expert systems
            (2020, 0.022, 1.3),  # Digitaliseringsslag COVID-19
            (2023, 0.055, 1.2),  # Doorbraak Generatieve AI (ChatGPT/LLMs)
        ],
        "Paris": [
            (1914, 0.020, 1.5),  # WOI
            (1918, 0.034, 1.5),  # Vredesconferentie van Parijs
            (1945, 0.022, 1.5),  # Bevrijding & naoorlogse verdragen
            (2015, 0.038, 1.3),  # Klimaatakkoord van Parijs
        ],
        "paris": [
            (1918, 0.004, 1.5),
            (2015, 0.006, 1.3),
        ],
    }

    for event_year, amplitude, sigma in spike_definitions.get(ngram, []):
        spiked += amplitude * np.exp(-0.5 * ((years - event_year) / sigma) ** 2)

    return spiked


def inject_smoothing_noise(
    trend: np.ndarray,
    rng: np.random.Generator,
    noise_intensity: float = 0.38,
) -> np.ndarray:
    """
    Voegt opzettelijk flinke jaarlijkse fluctuaties (hoogfrequente ruis) toe met NumPy.

    Wiskundige opbouw:
    We combineren twee soorten ruis:
    1. Multiplicatieve Gaussische ruis: schaalt mee met de hoogte van de trend
       zodat grote trends ook grote zichtbare kartels (zaagtanden) vertonen.
    2. Additieve hoogfrequente jitter + incidentele uitschieters (Poisson/Laplace):
       zorgt ervoor dat ook lagere segmenten duidelijk 'ruw' ogen bij Smoothing = 0.

    UX-doel:
    Wanneer de gebruiker in de UI de Smoothing Slider van 0 naar bijv. 10, 25 of 50
    slept, berekent de frontend een voortschrijdend gemiddelde (Moving Average).
    Dankzij deze ruis met gemiddelde 0 wordt de grafiek bij hogere smoothing-waarden
    zichtbaar gladder zonder dat de onderliggende trendlijn verschuift.
    """
    n = len(trend)

    # 1. Multiplicatieve ruis (standaarddeviatie ~38% van de trendwaarde)
    multiplicative_noise = rng.normal(loc=1.0, scale=noise_intensity, size=n)

    # 2. Hoogfrequente alternerende jitter (zodat opeenvolgende jaren sterk contrasteren)
    alternating_jitter = rng.Laplace(loc=0.0, scale=0.0012, size=n) if hasattr(rng, "Laplace") else rng.laplace(loc=0.0, scale=0.0012, size=n)

    noisy_trend = trend * multiplicative_noise + alternating_jitter

    # Frequenties kunnen wiskundig nooit negatief zijn (ondergrens op een kleine positieve waarde)
    return np.clip(noisy_trend, a_min=0.00002, a_max=None)


# ==============================================================================
# 3. EDGE CASES & FOUTAFHANDELING (FEEDFORWARD SUGGESTIES)
# ==============================================================================

# Dictionary met veelvoorkomende gebruikersfouten, typfouten, oude programmeersyntax
# en anachronismen, gekoppeld aan constructieve feedforward-suggesties.
# UX-doel: Vervangt de frustrerende Google Ngram foutmelding "No valid ngrams to plot!"
# door begrijpelijke uitleg en direct aanklikbare herstelsuggesties in de UI.
EDGE_CASE_QUERY_SUGGESTIONS: Dict[str, Dict[str, Any]] = {
    "climmate": {
        "error_type": "TYPO_DETECTED",
        "user_message_nl": "Geen exacte resultaten gevonden voor 'climmate'. Bedoelde je misschien 'climate'?",
        "suggested_queries": ["climate", "climate change", "global warming"],
        "recommended_filters": {"corpus_region": "American English", "pos_tag": "_NOUN"},
        "did_you_mean": "climate",
    },
    "democracie": {
        "error_type": "LANGUAGE_OR_SPELLING_MISMATCH",
        "user_message_nl": "Geen resultaten in het geselecteerde Engelse corpus voor 'democracie'. Zoek je het Engelse 'democracy' of wil je wisselen naar het Nederlandstalige corpus ('Dutch')?",
        "suggested_queries": ["democracy", "democratie", "democratic"],
        "recommended_filters": {"corpus_region": "Dutch", "pos_tag": "_NOUN"},
        "did_you_mean": "democracy",
    },
    "artifical intelligence": {
        "error_type": "TYPO_DETECTED",
        "user_message_nl": "Let op: veelvoorkomende spelfout in 'artifical' (ontbrekende 'i'). We tonen suggesties voor 'artificial intelligence'.",
        "suggested_queries": ["artificial intelligence", "machine learning", "neural network"],
        "recommended_filters": {"start_year": 1950, "end_year": 2026, "pos_tag": "_NOUN"},
        "did_you_mean": "artificial intelligence",
    },
    "telegramm": {
        "error_type": "TYPO_DETECTED",
        "user_message_nl": "'telegramm' lijkt op de Duitse spelling. Gebruik 'telegram' voor de Engelse en Nederlandse corpora.",
        "suggested_queries": ["telegram", "telegraph", "morse code"],
        "recommended_filters": {"start_year": 1840, "end_year": 1960, "pos_tag": "_NOUN"},
        "did_you_mean": "telegram",
    },
    "climate_NOUN": {
        "error_type": "LEGACY_SYNTAX_USED",
        "user_message_nl": "Je hoeft geen '_NOUN' meer te typen in de zoekbalk! Gebruik de nieuwe visuele Woordsoort-checkboxes (Zelfstandig naamwoord) onder het zoekveld.",
        "suggested_queries": ["climate"],
        "recommended_filters": {"pos_tag": "_NOUN"},
        "did_you_mean": "climate",
    },
    "artificial intelligence makes": {
        "error_type": "NGRAM_TOO_SPECIFIC",
        "user_message_nl": "Deze woordcombinatie komt te weinig voor om een betrouwbare trendlijn te tekenen. Probeer de kernterm 'artificial intelligence' in combinatie met het werkwoord-filter ('_VERB').",
        "suggested_queries": ["artificial intelligence", "intelligence"],
        "recommended_filters": {"pos_tag": "_VERB"},
        "did_you_mean": "artificial intelligence",
    },
}


def get_query_feedforward(user_query: str, available_ngrams: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Functie voor proactieve en constructieve foutafhandeling (Feedforward & Feedback).

    In plaats van een doodlopende 'No valid ngrams to plot!' foutmelding te geven,
    controleert deze functie:
    1. Of de zoekterm al geldig is (inclusief hoofdlettergevoelige suggestie zoals 'paris' -> 'Paris').
    2. Of de gebruiker per ongeluk oude Ngram-syntax (zoals '_NOUN' of '_INF') heeft getypt.
    3. Of de zoekterm voorkomt in onze bekende typfouten/edge-cases dictionary.
    4. Anders: berekent een karakter-overeenkomst (fuzzy fallback) voor de beste suggestie.
    """
    if available_ngrams is None:
        available_ngrams = NGRAMS

    clean_query = user_query.strip()

    # Check 1: Exacte match in de dataset
    if clean_query in available_ngrams:
        return {
            "status": "VALID",
            "query": clean_query,
            "user_message_nl": f"Resultaten geladen voor '{clean_query}'.",
            "suggested_queries": [],
        }

    # Check 2: Bekende edge case of typfout uit onze dictionary (case-insensitive lookup)
    lookup_key = clean_query.lower()
    for key, metadata in EDGE_CASE_QUERY_SUGGESTIONS.items():
        if key.lower() == lookup_key:
            return {
                "status": "RECOVERABLE_ERROR",
                "query": clean_query,
                **metadata,
            }

    # Check 3: Gebruiker typte handmatig een oude POS-tag achter een woord (bijv. 'democracy_VERB')
    for tag in ["_NOUN", "_VERB", "_INF", "_ADJ", "_ADV"]:
        if clean_query.upper().endswith(tag):
            base_word = clean_query[: -len(tag)].strip()
            return {
                "status": "RECOVERABLE_ERROR",
                "query": clean_query,
                "error_type": "LEGACY_SYNTAX_USED",
                "user_message_nl": (
                    f"Handmatige tag '{tag}' gedetecteerd. In deze nieuwe interface kun je "
                    f"gewoon '{base_word}' zoeken en de visuele tag-knop '{tag}' aanvinken."
                ),
                "suggested_queries": [base_word] if base_word in available_ngrams else available_ngrams[:3],
                "recommended_filters": {"pos_tag": tag if tag in POS_TAGS else "NONE"},
                "did_you_mean": base_word,
            }

    # Check 4: Algemene fallback met constructieve suggesties uit beschikbare ngrams
    return {
        "status": "NOT_FOUND_WITH_SUGGESTIONS",
        "query": clean_query,
        "error_type": "OUT_OF_CORPUS_VOCABULARY",
        "user_message_nl": (
            f"De term '{clean_query}' komt niet voor in deze test-dataset. "
            "Kies een van de onderstaande verwante onderwerpen om de tijdlijn te verkennen:"
        ),
        "suggested_queries": ["climate", "democracy", "artificial intelligence", "telegram", "Paris"],
        "recommended_filters": {"pos_tag": "NONE", "case_sensitive": False},
        "did_you_mean": "climate",
    }


# ==============================================================================
# 4. DATASET OPBOUW & EXPORT FUNCTIES
# ==============================================================================

def build_synthetic_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Genereert het volledige dataframe met alle gevraagde kolommen voor de UI-tests:
    - year
    - ngram
    - pos_tag
    - case_sensitive
    - corpus_region
    - raw_match_count
    - total_books_in_corpus
    - relative_frequency
    - historical_event
    """
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    # We genereren eerst het jaarlijkse boekvolume per regio zodat dit consistent is
    # voor alle zoektermen binnen hetzelfde corpus en jaartal.
    region_volumes: Dict[str, np.ndarray] = {
        region: generate_corpus_volume(YEARS, region, rng)
        for region in CORPUS_REGIONS
    }

    # Wegingsfactoren voor POS-tags:
    # 'NONE' (ongefilterd totaal) heeft de hoogste waarde; specifieke woordsoorten
    # zoals '_NOUN', '_VERB', en '_INF' zijn deelverzamelingen met eigen verhoudingen.
    pos_multipliers = {
        "NONE": 1.00,
        "_NOUN": 0.68,
        "_VERB": 0.22,
        "_INF": 0.10,
    }

    # Regio-modifiers per zoekterm (bijv. 'Paris' komt vaker voor in 'European Fiction',
    # 'artificial intelligence' iets sterker in 'American English').
    region_affinity = {
        ("Paris", "European Fiction"): 1.45,
        ("climate", "American English"): 1.15,
        ("climate", "Dutch"): 1.25,
        ("democracy", "American English"): 1.20,
        ("telegram", "British English"): 1.30,
        ("artificial intelligence", "American English"): 1.35,
    }

    for ngram in NGRAMS:
        base_trend = generate_base_trend(YEARS, ngram)
        spiked_trend = apply_historical_spikes(YEARS, base_trend, ngram)

        for region in CORPUS_REGIONS:
            total_books_array = region_volumes[region]
            affinity = region_affinity.get((ngram, region), 1.0)

            for pos_tag in POS_TAGS:
                pos_factor = pos_multipliers[pos_tag]

                for case_sensitive in [True, False]:
                    # UX-logica voor de Case-Sensitivity toggle:
                    # Als case_sensitive=False is, worden hoofd- en kleine letters bij elkaar
                    # opgeteld (hogere frequentie, en 'Paris'/'paris' gedragen zich identiek).
                    # Als case_sensitive=True is, zien we het scherpe verschil tussen
                    # 'Paris' (de stad, veel treffers) en 'paris' (kleine letter, weinig treffers).
                    if not case_sensitive:
                        if ngram in ("Paris", "paris"):
                            combined_base = generate_base_trend(YEARS, "Paris") + generate_base_trend(YEARS, "paris")
                            effective_trend = apply_historical_spikes(YEARS, combined_base, "Paris")
                        else:
                            effective_trend = spiked_trend * 1.18  # Inclusief gekapitaliseerde varianten aan begin zin
                    else:
                        effective_trend = spiked_trend

                    # Pas regio- en woordsoort-verhoudingen toe
                    scaled_trend = effective_trend * affinity * pos_factor

                    # Voeg stevige jaarlijkse ruis toe voor het testen van de Smoothing Slider
                    noisy_rel_freq = inject_smoothing_noise(scaled_trend, rng, noise_intensity=0.35)

                    # Bereken raw_match_count op basis van relative_frequency (%) en total_books_in_corpus:
                    # relative_frequency = (raw_match_count / total_books_in_corpus) * 100
                    # => raw_match_count = round((relative_frequency / 100) * total_books_in_corpus)
                    raw_counts = np.maximum(
                        1,
                        np.round((noisy_rel_freq / 100.0) * total_books_array).astype(int),
                    )

                    # Herbereken de exacte relative_frequency uit de afgeronde raw_match_count
                    # zodat de wiskundige formule (raw_match_count / total_books_in_corpus * 100)
                    # in elke rij 100% klopt voor transparantie-checks in de UI.
                    exact_rel_freq = (raw_counts / total_books_array) * 100.0

                    for idx, year in enumerate(YEARS):
                        yr_int = int(year)
                        records.append(
                            {
                                "year": yr_int,
                                "ngram": ngram,
                                "pos_tag": pos_tag,
                                "case_sensitive": bool(case_sensitive),
                                "corpus_region": region,
                                "raw_match_count": int(raw_counts[idx]),
                                "total_books_in_corpus": int(total_books_array[idx]),
                                "relative_frequency": round(float(exact_rel_freq[idx]), 6),
                                "historical_event": HISTORICAL_EVENTS.get(yr_int, None),
                            }
                        )

    df = pd.DataFrame(records)
    return df


def export_data(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Exporteert de gegenereerde dataset naar zowel CSV als JSON om de data-export
    knoppen en de frontend data-loader van de Ngram Viewer te voeden.
    Exporteert daarnaast ook de feedforward-foutafhandelingsregels naar JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "synthetic_ngram_data.csv"
    json_path = output_dir / "synthetic_ngram_data.json"
    suggestions_path = output_dir / "ngram_error_suggestions.json"

    # 1. Exporteer naar CSV (voor spreadsheet-gebruikers & data-export knop)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # 2. Exporteer naar JSON (voor directe consumptie in web-UI / D3 / Chart.js / React)
    # Gebruik df.where(pd.notna(df), None) zodat ontbrekende historical_event waarden
    # als standaard JSON 'null' (Python None) worden weggeschreven i.p.v. NaN.
    clean_df = df.astype(object).where(pd.notna(df), None)
    records = clean_df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, allow_nan=False)

    # 3. Exporteer de edge-case & feedforward-suggesties (voor de zoekbalk-validatie)
    with open(suggestions_path, "w", encoding="utf-8") as f:
        json.dump(EDGE_CASE_QUERY_SUGGESTIONS, f, ensure_ascii=False, indent=2)

    print(f"[OK] CSV geëxporteerd naar : {csv_path.resolve()} ({len(df):,} rijen)")
    print(f"[OK] JSON geëxporteerd naar: {json_path.resolve()} ({len(records):,} objecten)")
    print(f"[OK] Feedforward suggesties: {suggestions_path.resolve()} ({len(EDGE_CASE_QUERY_SUGGESTIONS)} edge cases)")


if __name__ == "__main__":
    print("Genereren van synthetische Ngram Viewer dataset (1800 - 2026)...")
    ngram_df = build_synthetic_dataset(seed=42)
    export_data(ngram_df, Path("."))

    # Korte demonstratie van de Feedforward-foutafhandeling in de terminal
    print("\n--- Test van Feedforward Foutafhandeling (ter vervanging van 'No valid ngrams to plot!') ---")
    for test_query in ["climmate", "democracie", "climate_NOUN", "onbekendwoord"]:
        feedback = get_query_feedforward(test_query)
        print(f"Zoekopdracht: '{test_query}' -> [{feedback['status']}] {feedback['user_message_nl']}")
        print(f"  Suggesties: {feedback['suggested_queries']}")
