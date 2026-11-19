"""
================================================================================
GRAPHRAG ENGINE: KNOWLEDGE GRAPH & HYBRID RETRIEVAL-AUGMENTED GENERATION
================================================================================
Bestand : graph_rag_engine.py
Doel    : Biedt Local Search, Global Community Search, Multi-hop Pathfinding,
          Live Google Ngram Data Ingestion en Vis.js visualisatie.
================================================================================
"""

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# Kleurenpalet voor entiteit-categorieën (Google Material look)
CATEGORY_COLORS: Dict[str, str] = {
    "Concept": "#1a73e8",               # Google Blauw
    "Technologie": "#e37400",           # Google Oranje
    "Historisch Event": "#d93025",      # Google Rood
    "Sleutelfiguur": "#188038",         # Google Groen
    "Tijdperk / Stroming": "#9334e6",    # Google Paars
    "Literatuur & Cultuur": "#00897b",  # Google Teal
    "Wetenschap & Fysica": "#3949ab",   # Google Indigo
    "Live Ngram Concept": "#d81b60",    # Google Magenta (voor dynamisch toegevoegde termen)
    "Google Books Database": "#0288d1", # Google Ocean Blue (voor database geminede co-locaties)
}


# ==============================================================================
# 0. LIVE GOOGLE NGRAM DATA INGESTION
# ==============================================================================
def fetch_live_ngram_metrics(term: str, corpus: str = "en-US-2019") -> Dict[str, Any]:
    """
    Haalt realtime Google Books Ngram data op voor een entiteit en berekent:
    - Piekjaar
    - Piekwaarde (frequentie in %)
    - Huidige trend (stijgend / dalend / stabiel)
    - Tijdreekssteekproef
    """
    clean_term = term.strip()
    if not clean_term:
        return {"success": False, "term": term}

    params = {
        "content": clean_term,
        "year_start": "1800",
        "year_end": "2019",
        "corpus": corpus,
        "smoothing": "3",
    }
    url = f"https://books.google.com/ngrams/json?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NgramGraphRAG/2.0"},
        )
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        if payload and isinstance(payload, list) and payload[0].get("timeseries"):
            ts = payload[0]["timeseries"]
            years = list(range(1800, 1800 + len(ts)))
            
            # Piekberekening
            max_idx = max(range(len(ts)), key=lambda i: ts[i])
            peak_year = years[max_idx]
            peak_val_pct = float(ts[max_idx]) * 100.0

            # Trendberekening (vergelijk laatste 20 jaar met voorgaande 20 jaar)
            recent_avg = sum(ts[-20:]) / max(1, len(ts[-20:]))
            prior_avg = sum(ts[-40:-20]) / max(1, len(ts[-40:-20])) if len(ts) >= 40 else recent_avg

            if recent_avg > prior_avg * 1.25:
                trend_desc = "Sterk stijgend in recente decennia"
            elif recent_avg < prior_avg * 0.75:
                trend_desc = "Afnemend na historische piek"
            else:
                trend_desc = "Stabiel niveau"

            # Steekproefwaarden voor mini-visualisatie
            sample_points = {
                1850: round(float(ts[50]) * 100.0, 5) if len(ts) > 50 else 0.0,
                1900: round(float(ts[100]) * 100.0, 5) if len(ts) > 100 else 0.0,
                1950: round(float(ts[150]) * 100.0, 5) if len(ts) > 150 else 0.0,
                2000: round(float(ts[200]) * 100.0, 5) if len(ts) > 200 else 0.0,
                2019: round(float(ts[-1]) * 100.0, 5) if ts else 0.0,
            }

            return {
                "success": True,
                "term": clean_term,
                "peak_year": peak_year,
                "peak_val_pct": peak_val_pct,
                "trend": trend_desc,
                "sample_points": sample_points,
                "source": "Live Google Books Ngram Index",
            }
    except Exception:
        pass

    # Fallback op heuristische data als de Google API onverhoopt offline is
    return {
        "success": False,
        "term": clean_term,
        "peak_year": 1990,
        "peak_val_pct": 0.0045,
        "trend": "➡️ Stabiel (geschat)",
        "sample_points": {1850: 0.0001, 1900: 0.0005, 1950: 0.002, 2000: 0.004},
        "source": "Offline historische index",
    }


# Stopwords voor kwalitatieve bigram- en associatiefiltering
ENGLISH_STOPWORDS: Set[str] = {
    # Lidwoorden, voegwoorden, voorzetsels
    "the", "of", "and", "a", "an", "in", "to", "for", "with", "on", "at", "by", "from",
    "up", "about", "into", "over", "after", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "but", "if", "or", "because",
    "as", "until", "while", "that", "this", "these", "those", "then", "so", "than",
    "too", "very", "can", "will", "just", "should", "now", "our", "its", "their", "my",
    "your", "his", "her", "which", "what", "who", "whom", "whose", "it", "they", "them",
    "we", "us", "you", "i", "me", "he", "she", "him", "one", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "there", "here", "when", "where", "how", "also", "between", "under", "again", "further",
    # Modifiers, vage bijwoorden en comparatieven die geen historische concepten zijn
    "less", "much", "many", "good", "bad", "great", "small", "large", "high", "low",
    "first", "last", "new", "old", "next", "per", "even", "like", "well",
    "back", "still", "way", "may", "might", "must", "shall", "would", "could", "done",
    "make", "made", "get", "got", "take", "took", "see", "saw", "know", "known",
    "part", "parts", "case", "cases", "point", "points", "time", "times", "year", "years",
    "day", "days", "way", "ways", "man", "men", "two", "three", "four", "five"
}


def mine_google_ngram_associations(
    term: str,
    corpus: str = "en-US-2019",
    max_results: int = 6,
) -> List[Dict[str, Any]]:
    """
    Bevraagt de officiële Google Books Ngram Viewer database met wildcard query's
    om authentieke historische co-occurrenties, bigrams en conceptuele associaties
    te ontdekken die in miljoenen gescande boeken voorkomen.
    """
    clean_term = term.strip().lower()
    if not clean_term:
        return []

    # Wildcards: linker associatie (* term), rechter associatie (term *), en conjunctions (term and *)
    queries = [f"* {clean_term}", f"{clean_term} *"]
    if " " not in clean_term:
        queries.append(f"{clean_term} and *")

    query_str = ",".join(queries)
    params = {
        "content": query_str,
        "year_start": "1800",
        "year_end": "2019",
        "corpus": corpus,
        "smoothing": "3",
    }
    url = f"https://books.google.com/ngrams/json?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NgramDatabaseMiner/2.0"},
        )
        with urllib.request.urlopen(req, timeout=4.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    extracted: List[Dict[str, Any]] = []
    seen: Set[str] = {clean_term}

    for item in data:
        ngram = item.get("ngram", "").strip()
        # Filter ongeldige patronen, asterisken of leestekens
        if not ngram or "*" in ngram or not item.get("timeseries"):
            continue
        if any(char in ngram for char in ["'", '"', ",", ".", ":", ";", "/", "\\", "_"]):
            continue

        lower_ng = ngram.lower()
        words = lower_ng.split()
        if len(words) < 2:
            continue

        # Filter stopwords
        other_words = [w for w in words if w != clean_term and w != "and"]
        if not other_words or any(w in ENGLISH_STOPWORDS for w in other_words):
            continue

        if lower_ng in seen:
            continue
        seen.add(lower_ng)

        ts = item["timeseries"]
        max_val = max(ts) if ts else 0.0
        # Betrouwbaarheidsdrempel: filter toevallige OCR-ruis weg (< 1 op de 10 miljoen woorden)
        if max_val < 1e-7:
            continue

        peak_idx = ts.index(max_val)
        peak_year = 1800 + peak_idx
        peak_val_pct = float(max_val) * 100.0

        # Bereken gemiddelde frequentie
        avg_val_pct = (sum(ts) / max(1, len(ts))) * 100.0

        direction = "prefix" if words[0] != clean_term else "suffix"

        extracted.append(
            {
                "ngram": ngram,
                "clean_label": ngram.title(),
                "peak_year": peak_year,
                "peak_val_pct": peak_val_pct,
                "avg_val_pct": avg_val_pct,
                "direction": direction,
                "relation_type": "CO_OCCURS_IN_GOOGLE_BOOKS",
                "sample_points": {
                    1850: round(float(ts[50]) * 100.0, 5) if len(ts) > 50 else 0.0,
                    1900: round(float(ts[100]) * 100.0, 5) if len(ts) > 100 else 0.0,
                    1950: round(float(ts[150]) * 100.0, 5) if len(ts) > 150 else 0.0,
                    2000: round(float(ts[200]) * 100.0, 5) if len(ts) > 200 else 0.0,
                },
            }
        )

    # Rangschik op piekfrequentie (hoogste historische significantie)
    extracted.sort(key=lambda x: x["peak_val_pct"], reverse=True)
    return extracted[:max_results]


# ==============================================================================
# 1. KENNISGRAAF INITIALISATIE & UITGEBREIDE ONTOLOGIE (55+ ENTITEITEN)
# ==============================================================================
def build_historical_knowledge_graph() -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Construeert de NetworkX kennisgraaf met een rijke ontologie van historische entiteiten,
    getypeerde relaties, gewichten en community metadata over wetenschap, literatuur,
    technologie en politiek.
    """
    G = nx.DiGraph()

    entities = [
        # --- Ecologie, Energie & Klimaat ---
        {"id": "climate", "label": "Klimaat / Climate", "category": "Concept", "period": "1800-heden", "ngram": "climate", "desc": "Wetenschappelijk en maatschappelijk begrip van wereldwijde weersystemen en opwarming."},
        {"id": "fossil fuels", "label": "Fossiele Brandstoffen", "category": "Concept", "period": "1820-heden", "ngram": "fossil fuels", "desc": "Kolen, olie en gas als primaire energiebronnen van de moderne industriële wereld."},
        {"id": "sustainability", "label": "Duurzaamheid", "category": "Concept", "period": "1970-heden", "ngram": "sustainability", "desc": "Beleid en ecologische theorie gericht op het behoud van natuurlijke hulpbronnen."},
        {"id": "nuclear energy", "label": "Kernenergie", "category": "Technologie", "period": "1942-heden", "ngram": "nuclear energy", "desc": "Energie opgewekt door atoomkerndeling, ontwikkeld na de Tweede Wereldoorlog."},
        {"id": "steam engine", "label": "Stoommachine", "category": "Technologie", "period": "1769-1920", "ngram": "steam engine", "desc": "Thermische machine van James Watt die de Industriële Revolutie in gang zette."},
        {"id": "Industrial Revolution", "label": "Industriële Revolutie", "category": "Historisch Event", "period": "1760-1840", "ngram": "Industrial Revolution", "desc": "Overgang van handmatige productie naar machinale fabrieken en massale urbanisatie."},
        {"id": "Paris Climate Accord 2015", "label": "Klimaatakkoord Parijs (2015)", "category": "Historisch Event", "period": "2015", "ngram": "climate", "desc": "Internationaal bindend verdrag om opwarming te beperken tot ruim onder 2 graden Celsius."},
        {"id": "James Watt", "label": "James Watt", "category": "Sleutelfiguur", "period": "1736-1819", "ngram": "James Watt", "desc": "Schotse uitvinder die de stoommachine perfectioneerde met een aparte condensator."},
        {"id": "Alexander von Humboldt", "label": "Alexander von Humboldt", "category": "Sleutelfiguur", "period": "1769-1859", "ngram": "Alexander von Humboldt", "desc": "Pionier in ecologie en biogeografie; waarschuwde als eerste voor menselijke klimaatinvloed."},

        # --- Communicatie, Informatica & AI ---
        {"id": "telegraph", "label": "Elektrische Telegraaf", "category": "Technologie", "period": "1844-1970", "ngram": "telegraph", "desc": "Eerste technologie voor instantane langeafstandstelecommunicatie via morsecode."},
        {"id": "electricity", "label": "Elektriciteit", "category": "Technologie", "period": "1880-heden", "ngram": "electricity", "desc": "Universele energiedrager voor verlichting, fabrieken en elektronica."},
        {"id": "radio", "label": "Draadloze Radio", "category": "Technologie", "period": "1895-heden", "ngram": "radio", "desc": "Draadloze uitzending van geluidsgolven via elektromagnetische straling."},
        {"id": "computer", "label": "Elektronische Computer", "category": "Technologie", "period": "1940-heden", "ngram": "computer", "desc": "Programmeerbare digitale rekenmachine voor dataverwerking en automatie."},
        {"id": "internet", "label": "Internet & Web", "category": "Technologie", "period": "1990-heden", "ngram": "internet", "desc": "Wereldwijd computernetwerk dat massacommunicatie en kennistoegang democratiseerde."},
        {"id": "smartphone", "label": "Smartphone", "category": "Technologie", "period": "2007-heden", "ngram": "smartphone", "desc": "Mobiele pocketcomputer die internet, navigatie en communicatie alomtegenwoordig maakte."},
        {"id": "artificial intelligence", "label": "Artificial Intelligence", "category": "Concept", "period": "1956-heden", "ngram": "artificial intelligence", "desc": "Het wetenschapsgebied gericht op het creëren van machines die menselijke cognitie nabootsen."},
        {"id": "generative ai", "label": "Generatieve AI & LLMs", "category": "Concept", "period": "2020-heden", "ngram": "artificial intelligence", "desc": "Geavanceerde neurale netwerken die autonoom taal, beelden en code synthetiseren."},
        {"id": "Morse Telegraph 1844", "label": "Eerste Telegraaflijn (1844)", "category": "Historisch Event", "period": "1844", "ngram": "telegraph", "desc": "Samuel Morse verstuurt het eerste telegram: 'What hath God wrought'."},
        {"id": "Dartmouth Workshop 1956", "label": "Dartmouth AI Workshop (1956)", "category": "Historisch Event", "period": "1956", "ngram": "artificial intelligence", "desc": "Academische bijeenkomst waar John McCarthy de term 'Artificial Intelligence' introduceerde."},
        {"id": "Commercial Internet 1995", "label": "Commercieel Internet (1995)", "category": "Historisch Event", "period": "1995", "ngram": "internet", "desc": "Opkomst van browsers, zoekmachines en de dotcom-economie."},
        {"id": "Ada Lovelace", "label": "Ada Lovelace", "category": "Sleutelfiguur", "period": "1815-1852", "ngram": "Ada Lovelace", "desc": "Eerste computerprogrammeur die het universele potentieel van rekenmachines voorzag."},
        {"id": "Alan Turing", "label": "Alan Turing", "category": "Sleutelfiguur", "period": "1912-1954", "ngram": "Alan Turing", "desc": "Wiskundige, grondlegger van de informatica en bedenker van de Turingtest voor AI."},
        {"id": "Information Age", "label": "Informatietijdperk", "category": "Tijdperk / Stroming", "period": "1970-heden", "ngram": "Information Age", "desc": "Historische periode gekenmerkt door digitale data, snelle netwerken en computerautomatisering."},

        # --- Fysica & Ruimtevaart ---
        {"id": "gravity", "label": "Zwaartekracht (Gravitatie)", "category": "Wetenschap & Fysica", "period": "1687-heden", "ngram": "gravity", "desc": "Fundamentele natuurkracht beschreven door Newton en geherdefinieerd door Einstein."},
        {"id": "relativity", "label": "Relativiteitstheorie", "category": "Wetenschap & Fysica", "period": "1905-heden", "ngram": "relativity", "desc": "Einsteins theorie die ruimte, tijd en zwaartekracht verenigt in ruimtetijd."},
        {"id": "quantum mechanics", "label": "Kwantummechanica", "category": "Wetenschap & Fysica", "period": "1925-heden", "ngram": "quantum mechanics", "desc": "Fysische theorie over materie en energie op microscopische atomaire schaal."},
        {"id": "space exploration", "label": "Ruimtevaart", "category": "Technologie", "period": "1957-heden", "ngram": "space exploration", "desc": "Verkenning van het heelal met raketten, satellieten en bemande missies."},
        {"id": "Apollo 11", "label": "Apollo 11 Maanlanding (1969)", "category": "Historisch Event", "period": "1969", "ngram": "Apollo 11", "desc": "Eerste mens op de maan (Neil Armstrong & Buzz Aldrin) en bron van de beroemde 'Earthrise' foto."},
        {"id": "Isaac Newton", "label": "Isaac Newton", "category": "Sleutelfiguur", "period": "1643-1727", "ngram": "Isaac Newton", "desc": "Grondlegger van de klassieke mechanica en optica ('Principia Mathematica')."},
        {"id": "Albert Einstein", "label": "Albert Einstein", "category": "Sleutelfiguur", "period": "1879-1955", "ngram": "Albert Einstein", "desc": "Theoretisch fysicus, formuleerde de relativiteitstheorie en E=mc²."},

        # --- Literatuur & Cultuurgeschiedenis ---
        {"id": "Frankenstein", "label": "Frankenstein (Boek & Mythe)", "category": "Literatuur & Cultuur", "period": "1818-heden", "ngram": "Frankenstein", "desc": "Mary Shelley's klassieke roman over wetenschappelijke hoogmoed en kunstmatig leven."},
        {"id": "Sherlock Holmes", "label": "Sherlock Holmes", "category": "Literatuur & Cultuur", "period": "1887-heden", "ngram": "Sherlock Holmes", "desc": "Arthur Conan Doyle's meesterdetective die logische deductie en forensische wetenschap populariseerde."},
        {"id": "Mary Shelley", "label": "Mary Shelley", "category": "Sleutelfiguur", "period": "1797-1851", "ngram": "Mary Shelley", "desc": "Engelse schrijfster en grondlegger van de moderne sciencefictionliteratuur."},
        {"id": "Arthur Conan Doyle", "label": "Arthur Conan Doyle", "category": "Sleutelfiguur", "period": "1859-1930", "ngram": "Conan Doyle", "desc": "Schots arts en auteur, schepper van Sherlock Holmes en Dr. Watson."},
        {"id": "Romanticism", "label": "Romantiek", "category": "Tijdperk / Stroming", "period": "1790-1850", "ngram": "Romanticism", "desc": "Culturele stroming die emotie, ontzag voor de natuur en verzet tegen zielloze mechanisatie benadrukte."},
        {"id": "Victorian Era", "label": "Victoriaans Tijdperk", "category": "Tijdperk / Stroming", "period": "1837-1901", "ngram": "Victorian", "desc": "Periode van Britse industriële hegemonie, sociale hervormingen en literaire bloei."},

        # --- Democratie, Rechten & Filosofie ---
        {"id": "democracy", "label": "Democratie / Democracy", "category": "Concept", "period": "Klassiek-heden", "ngram": "democracy", "desc": "Politiek systeem gebaseerd op volkssoevereiniteit, gelijke rechten en vrije verkiezingen."},
        {"id": "freedom of speech", "label": "Vrijheid van Meningsuiting", "category": "Concept", "period": "1789-heden", "ngram": "freedom of speech", "desc": "Fundamenteel burgerrecht vastgelegd tijdens de Verlichting en moderne grondwetten."},
        {"id": "human rights", "label": "Mensenrechten", "category": "Concept", "period": "1948-heden", "ngram": "human rights", "desc": "Universele rechten voor elk individu, verankerd in het VN-handvest van 1948."},
        {"id": "capitalism", "label": "Kapitalisme", "category": "Concept", "period": "1776-heden", "ngram": "capitalism", "desc": "Economisch stelsel gebaseerd op privaat bezit van productiemiddelen en marktwerking."},
        {"id": "socialism", "label": "Socialisme", "category": "Concept", "period": "1848-heden", "ngram": "socialism", "desc": "Politiek-economische ideologie gericht op collectief bezit en bescherming van arbeiders."},
        {"id": "psychology", "label": "Psychologie", "category": "Concept", "period": "1879-heden", "ngram": "psychology", "desc": "Wetenschappelijke studie van de menselijke geest en gedragspatronen."},
        {"id": "evolution", "label": "Evolutietheorie", "category": "Concept", "period": "1859-heden", "ngram": "evolution", "desc": "Natuurwetenschappelijke verklaring voor het ontstaan en aanpassen van biologische soorten."},
        {"id": "globalization", "label": "Globalisering", "category": "Concept", "period": "1870-heden", "ngram": "globalization", "desc": "Toenemende internationale verwevenheid van economieën, culturen en informatie."},
        {"id": "Enlightenment", "label": "De Verlichting", "category": "Tijdperk / Stroming", "period": "1700-1800", "ngram": "Enlightenment", "desc": "Filosofische stroming waarin rede, wetenschappelijk scepticisme en burgerrechten centraal stonden."},
        {"id": "American Civil War", "label": "Amerikaanse Burgeroorlog", "category": "Historisch Event", "period": "1861-1865", "ngram": "Civil War", "desc": "Strijd over eenheid en de afschaffing van slavernij in de Verenigde Staten."},
        {"id": "WW1", "label": "Eerste Wereldoorlog", "category": "Historisch Event", "period": "1914-1918", "ngram": "World War", "desc": "Mondiaal conflict dat imperia ontmantelde en industriële oorlogsvoering introduceerde."},
        {"id": "WW2", "label": "Tweede Wereldoorlog", "category": "Historisch Event", "period": "1939-1945", "ngram": "World War", "desc": "Wereldoorlog tegen het fascisme, leidend tot de oprichting van de Verenigde Naties."},
        {"id": "Cold War", "label": "Koude Oorlog", "category": "Historisch Event", "period": "1947-1991", "ngram": "Cold War", "desc": "Geopolitieke spanning tussen de VS en de Sovjet-Unie, die de Space Race en het internet aanjaagde."},
        {"id": "Adam Smith", "label": "Adam Smith", "category": "Sleutelfiguur", "period": "1723-1790", "ngram": "Adam Smith", "desc": "Filosoof en econoom, grondlegger van de klassieke markteconomie ('The Wealth of Nations')."},
        {"id": "Karl Marx", "label": "Karl Marx", "category": "Sleutelfiguur", "period": "1818-1883", "ngram": "Karl Marx", "desc": "Duits filosoof en econoom, auteur van 'Das Kapital' en 'Het Communistisch Manifest'."},
        {"id": "Charles Darwin", "label": "Charles Darwin", "category": "Sleutelfiguur", "period": "1809-1882", "ngram": "Charles Darwin", "desc": "Bioloog en auteur van 'On the Origin of Species' (1859)."},
        {"id": "Sigmund Freud", "label": "Sigmund Freud", "category": "Sleutelfiguur", "period": "1856-1939", "ngram": "Sigmund Freud", "desc": "Oostenrijks neuroloog en grondlegger van de psychoanalyse."},
    ]

    for e in entities:
        G.add_node(
            e["id"],
            label=e["label"],
            category=e["category"],
            period=e["period"],
            ngram=e["ngram"],
            desc=e["desc"],
            color=CATEGORY_COLORS.get(e["category"], "#5f6368"),
        )

    # --------------------------------------------------------------------------
    # RELATIES (EDGES)
    # --------------------------------------------------------------------------
    relations = [
        # Industrie & Energie & Klimaat
        ("James Watt", "steam engine", "PERFECTED", 0.95, "Watt ontwikkelde de stoommachine met aparte condensor."),
        ("steam engine", "Industrial Revolution", "CATALYZED", 0.98, "Stoomkracht maakte massale mechanisatie en spoorwegen mogelijk."),
        ("Industrial Revolution", "fossil fuels", "INCREASED_DEMAND", 0.92, "Fabrieken en treinen leidden tot grootschalig kolenverbruik."),
        ("fossil fuels", "climate", "ALIGNED_WITH_GROWTH", 0.88, "CO2-uitstoot leidde tot wetenschappelijke documentatie van het broeikaseffect."),
        ("Alexander von Humboldt", "climate", "FOUNDED_STUDY", 0.85, "Humboldt beschreef in Zuid-Amerika als eerste hoe ontbossing het microklimaat aantast."),
        ("climate", "sustainability", "SPURRED", 0.90, "Klimaataanwijzingen leidden tot de opkomst van duurzaamheidsbeleid en milieubewegingen."),
        ("climate", "Paris Climate Accord 2015", "CULMINATED_IN", 0.95, "Wereldwijd juridisch en diplomatiek antwoord op de klimaatcrisis."),
        ("sustainability", "Paris Climate Accord 2015", "MANDATED_BY", 0.87, "Verdrag stelt emissiereducties en duurzame transitie verplicht."),

        # Communicatie, Informatie & AI
        ("telegraph", "Morse Telegraph 1844", "INAUGURATED_BY", 0.95, "Demonstratie van de telegraaf tussen Washington en Baltimore."),
        ("telegraph", "globalization", "ACCELERATED", 0.89, "Financiële markten en nieuwsvoorziening werden mondiaal verbonden."),
        ("telegraph", "electricity", "DEPENDED_ON", 0.85, "Eerste grootschalige civiele toepassing van elektrische stroom."),
        ("electricity", "radio", "ENABLED", 0.88, "Draadloze radiocommunicatie bouwt voort op elektromagnetisme."),
        ("electricity", "computer", "ENABLED", 0.90, "Elektronica en transistors vormden de basis voor digitale computers."),
        ("Ada Lovelace", "computer", "THEORIZED_FOR", 0.92, "Schreef het eerste algoritme voor de Mechanische Analytische Machine van Babbage."),
        ("Alan Turing", "computer", "MATHEMATIZED", 0.96, "Turing-machine formaliseerde het universele concept van berekenbaarheid."),
        ("Alan Turing", "artificial intelligence", "PIONEERED", 0.94, "Publiceerde 'Computing Machinery and Intelligence' (1950) met de Turingtest."),
        ("Dartmouth Workshop 1956", "artificial intelligence", "CHRISTENED", 0.99, "Geboorte van AI als formeel academisch onderzoeksveld."),
        ("computer", "internet", "INTERCONNECTED_BY", 0.93, "Netwerken van computers culmineerden in het wereldwijde internet."),
        ("Commercial Internet 1995", "internet", "EXPANDED", 0.95, "Het publieke World Wide Web verving gesloten academische netwerken."),
        ("internet", "smartphone", "CONVERGED_INTO", 0.94, "Smartphones brachten het volledige web direct in de broekzak van miljarden mensen."),
        ("internet", "generative ai", "SUPPLIED_DATA", 0.92, "Het wereldwijde web leverde de enorme tekstcorpora om LLMs te trainen."),
        ("artificial intelligence", "generative ai", "EVOLVED_INTO", 0.96, "Diepe transformermodellen luidden de generatieve AI-revolutie in."),
        ("computer", "Information Age", "DEFINED", 0.91, "De computerisering verschoof de maatschappij naar een kenniseconomie."),

        # Fysica & Ruimtevaart
        ("Isaac Newton", "gravity", "FORMULATED", 0.97, "Publiceerde de universele zwaartekrachtwet in 1687."),
        ("gravity", "relativity", "RECONCEPTUALIZED_BY", 0.95, "Einstein verving Newtons aantrekkingskracht door kromming van ruimtetijd."),
        ("Albert Einstein", "relativity", "DISCOVERED", 0.99, "Publiceerde de speciale (1905) en algemene relativiteitstheorie (1915)."),
        ("Albert Einstein", "quantum mechanics", "CO_FOUNDED", 0.91, "Ontving de Nobelprijs voor de verklaring van het foto-elektrisch effect."),
        ("relativity", "nuclear energy", "ENABLED_BY_E_MC2", 0.93, "E=mc² toonde de theoretische mogelijkheid van atoomenergie aan."),
        ("nuclear energy", "WW2", "DEPLOYED_IN", 0.90, "Het Manhattan-project leidde tot de beëindiging van de Tweede Wereldoorlog."),
        ("nuclear energy", "Cold War", "STALEMATED", 0.92, "Wederzijdse afschrikking (MAD) voorkwam een directe hete oorlog."),
        ("Cold War", "space exploration", "COMPETITIVELY_SPURRED", 0.94, "De 'Space Race' tussen VS en USSR dreef razendsnelle innovatie."),
        ("space exploration", "Apollo 11", "PEAKED_WITH", 0.98, "Eerste bemande maanlanding in 1969."),
        ("Apollo 11", "climate", "EARTHRISE_AWARENESS", 0.86, "Foto's van de aarde vanuit de ruimte inspireerden de moderne milieubeweging."),
        ("computer", "space exploration", "NAVIGATED", 0.92, "De Apollo Guidance Computer maakte geautomatiseerde ruimtenavigatie mogelijk."),

        # Literatuur, Verbeelding & Cultuur
        ("Mary Shelley", "Frankenstein", "AUTHORED", 0.99, "Schreef Frankenstein in het beroemde 'jaar zonder zomer' (1816-1818)."),
        ("electricity", "Frankenstein", "INSPIRED", 0.93, "Galvanische experimenten met elektriciteit inspireerden het wekken van het monster."),
        ("Frankenstein", "Romanticism", "CHAMPIONED_GOTHIC", 0.91, "Hoogtepunt van de romantische kritiek op kille wetenschappelijke hoogmoed."),
        ("Industrial Revolution", "Romanticism", "TRIGGERED_REACTION", 0.89, "Romantici verzetten zich tegen fabriekslawaai en ontzieling van de natuur."),
        ("Frankenstein", "artificial intelligence", "PREFIGURED_MYTH", 0.94, "Blijft de universele culturele metafoor voor de angst voor autonoom kunstmatig leven."),
        ("Arthur Conan Doyle", "Sherlock Holmes", "CREATED", 0.99, "Introduceerde Sherlock Holmes in 'A Study in Scarlet' (1887)."),
        ("Sherlock Holmes", "Victorian Era", "EMBODIED_ETHOS", 0.95, "Holmes weerspiegelde de Victoriaanse fascinatie voor wetenschap, orde en het Londense rijk."),
        ("evolution", "Sherlock Holmes", "INFLUENCED_METHOD", 0.82, "Darwins systematische observatie beïnvloedde Holmes' deductieve methode."),

        # Democratie, Rechten & Filosofie
        ("Enlightenment", "democracy", "INTELLECTUALLY_FOUNDED", 0.93, "Ideeën van Locke en Rousseau herdefinieerden het staatsbestel."),
        ("Enlightenment", "freedom of speech", "CHAMPIONED", 0.95, "Voltaire en verlichte denkers bepleitten ongehinderde meningsuiting."),
        ("Adam Smith", "capitalism", "FORMULATED", 0.94, "Beschreef vrije marktwerking en arbeidsdeling ('Wealth of Nations')."),
        ("capitalism", "Industrial Revolution", "FINANCED", 0.91, "Particulier risicokapitaal financierde fabrieken, mijnen en spoorlijnen."),
        ("Industrial Revolution", "socialism", "CATALYZED_CRITIQUE", 0.93, "Slechte fabrieksomstandigheden leidden tot arbeidersbewegingen."),
        ("Karl Marx", "socialism", "CO_FOUNDED", 0.98, "Formuleerde het historisch materialisme en de kritiek op het fabriekskapitalisme."),
        ("socialism", "Cold War", "IDEOLOGICALLY_DRIVEN", 0.92, "Socialistische staatsmodellen stonden lijnrecht tegenover westers marktkapitalisme."),
        ("democracy", "freedom of speech", "REQUIRES", 0.97, "Vrije meningsuiting is een absolute voorwaarde voor democratische verkiezingen."),
        ("American Civil War", "democracy", "TESTED_AND_PRESERVED", 0.89, "Lincoln definieerde de oorlog als proef of een democratie kan standhouden."),
        ("WW2", "human rights", "MANDATED", 0.96, "Nazi-gruwelen leidden direct tot de Universele Verklaring van de Rechten van de Mens."),
        ("human rights", "democracy", "ANCHORS", 0.95, "Mensenrechten vormen de juridische kern van de hedendaagse democratische rechtsstaat."),
        ("Alexander von Humboldt", "Charles Darwin", "INSPIRED", 0.91, "Humboldts reisverslagen vormden Darwins motivatie voor de Beagle-expeditie."),
        ("Charles Darwin", "evolution", "SYSTEMATIZED", 0.98, "Formuleerde natuurlijke selectie als sturend mechanisme van biologische variatie."),
        ("Sigmund Freud", "psychology", "FOUNDED_PSYCHOANALYSIS", 0.96, "Ontwikkelde de theorie van het onbewuste en droomduiding."),
        ("psychology", "artificial intelligence", "INSPIRED_COGNITION", 0.87, "Modellen van menselijk geheugen en perceptie inspireerden vroege AI-pioniers."),
    ]

    for src, dst, rel_type, weight, expl in relations:
        G.add_edge(src, dst, relation_type=rel_type, weight=weight, explanation=expl)

    # --------------------------------------------------------------------------
    # COMMUNITIES (THEMATISCHE CLUSTERS)
    # --------------------------------------------------------------------------
    communities = {
        "cluster_klimaat_energie": {
            "title": "Klimaat, Energie & Industriële Transformatie",
            "nodes": ["Industrial Revolution", "steam engine", "fossil fuels", "climate", "sustainability", "nuclear energy", "Paris Climate Accord 2015", "James Watt", "Alexander von Humboldt"],
            "summary": (
                "Deze community beschrijft de wisselwerking tussen mechanisatie (stoomkracht), het exponentieel "
                "stijgende verbruik van fossiele brandstoffen en de uiteindelijke wetenschappelijke en diplomatieke "
                "erkenning van de klimaatverandering vanaf de 18e eeuw tot het Parijs-akkoord in 2015."
            ),
            "dominant_era": "1760 – Heden",
            "key_drivers": ["Mechanisatie", "Kolen en Olie", "Ecologische Wetenschap", "Mondiale Verdragen"],
        },
        "cluster_informatie_ai": {
            "title": "Informatie, Communicatie & AI Revolutie",
            "nodes": ["telegraph", "Morse Telegraph 1844", "electricity", "radio", "computer", "internet", "Commercial Internet 1995", "smartphone", "artificial intelligence", "generative ai", "Dartmouth Workshop 1956", "Ada Lovelace", "Alan Turing", "Information Age"],
            "summary": (
                "Toont de evolutie van telecommunicatie en computerautomatie: van de telegraaf in 1844 "
                "tot de Turingtest, de opkomst van het commerciële web in 1995 en de recente doorbraak "
                "van generatieve taalmodellen."
            ),
            "dominant_era": "1844 – Heden",
            "key_drivers": ["Telecommunicatie", "Binaire Informatica", "Neurale Netwerken", "Massale Webdata"],
        },
        "cluster_literatuur_cultuur": {
            "title": "Literatuur, Verbeelding & Cultuurgeschiedenis",
            "nodes": ["Frankenstein", "Sherlock Holmes", "Mary Shelley", "Arthur Conan Doyle", "Romanticism", "Victorian Era", "printing press"],
            "summary": (
                "Onderzoekt hoe iconische romans en culturele stromingen reageerden op wetenschappelijke veranderingen. "
                "Van Mary Shelley's waarschuwing tegen wetenschappelijke hoogmoed in Frankenstein tot Sherlock Holmes' "
                "viering van Victoriaanse forensische rationaliteit."
            ),
            "dominant_era": "1818 – 1914",
            "key_drivers": ["Gotische Romantiek", "Victoriaanse Rationaliteit", "Sciencefiction", "Forensica"],
        },
        "cluster_fysica_ruimtevaart": {
            "title": "Fundamentele Fysica & Ruimtevaart",
            "nodes": ["gravity", "relativity", "quantum mechanics", "space exploration", "Apollo 11", "Isaac Newton", "Albert Einstein", "Cold War"],
            "summary": (
                "De ontrafeling van het universum: van Newtons gravitatiewetten en Einsteins ruimtetijd "
                "tot de kwantummechanica en de Apollo 11 maanlanding die de mensheid een blik terug op aarde gunde."
            ),
            "dominant_era": "1687 – Heden",
            "key_drivers": ["Klassieke Mechanica", "Relativiteit", "Kwantumtheorie", "Space Race"],
        },
        "cluster_democratie_rechten": {
            "title": "Democratie, Rechten & Staatsinrichting",
            "nodes": ["Enlightenment", "democracy", "freedom of speech", "human rights", "capitalism", "socialism", "Adam Smith", "Karl Marx", "American Civil War", "WW1", "WW2", "Cold War"],
            "summary": (
                "De wording van moderne politieke stelsels en burgerrechten: van Verlichtingsfilosofie "
                "en klassiek kapitalisme tot de strijd tussen ideologieën in de Wereldoorlogen en de Koude Oorlog."
            ),
            "dominant_era": "1700 – Heden",
            "key_drivers": ["Verlichtingsfilosofie", "Vrije Meningsuiting", "Mensenrechten", "Ideologieën"],
        },
        "cluster_wetenschap_geest": {
            "title": "Evolutieleer & Wetenschap van de Geest",
            "nodes": ["evolution", "psychology", "Charles Darwin", "Sigmund Freud", "Alexander von Humboldt"],
            "summary": (
                "De wetenschappelijke herdefinitie van de mens: Darwins biologische evolutie via natuurlijke selectie "
                "en Freuds theorieën over het onbewuste en de menselijke psyche."
            ),
            "dominant_era": "1850 – Heden",
            "key_drivers": ["Natuurlijke Selectie", "Psychoanalyse", "Empirische Observatie"],
        },
    }

    return G, communities


# ==============================================================================
# 2. HISTORISCHE BEWIJSSTUKKEN (CORPUS CHUNKS)
# ==============================================================================
CORPUS_EVIDENCE_CHUNKS: List[Dict[str, Any]] = [
    {
        "id": "chunk_01",
        "title": "James Watt en de Condensor (1769)",
        "source": "The Industrial and Commercial Revolutions in Great Britain (1921)",
        "year": 1769,
        "entities": ["James Watt", "steam engine", "Industrial Revolution"],
        "text": (
            "Watt's invention of the separate condenser transformed the steam engine from a clumsy mine-pumping machine "
            "into a universal prime mover, providing steady rotative power to the mills and foundries of Great Britain."
        ),
    },
    {
        "id": "chunk_02",
        "title": "Humboldts Ecologische Waarnemingen (1805)",
        "source": "Essai sur la géographie des plantes (Alexander von Humboldt, 1805)",
        "year": 1805,
        "entities": ["Alexander von Humboldt", "climate", "evolution"],
        "text": (
            "When forests are destroyed, the springs dry up completely or flow only with diminishing volume. "
            "The beds of the brooks are converted into torrents, and mankind alters the very atmosphere and local climate."
        ),
    },
    {
        "id": "chunk_03",
        "title": "Frankenstein en de Vonk van het Leven (1818)",
        "source": "Frankenstein; or, The Modern Prometheus (Mary Shelley, 1818)",
        "year": 1818,
        "entities": ["Frankenstein", "Mary Shelley", "electricity", "Romanticism"],
        "text": (
            "With an anxiety that almost amounted to agony, I collected the instruments of life around me, "
            "that I might infuse a spark of being into the lifeless thing that lay at my feet. It was on a dreary night of November."
        ),
    },
    {
        "id": "chunk_04",
        "title": "De Eerste Telegraaflijn (1844)",
        "source": "Harper's New Monthly Magazine: The Lightning Telegraph (1850)",
        "year": 1844,
        "entities": ["telegraph", "Morse Telegraph 1844", "electricity", "globalization"],
        "text": (
            "The message transmitted along the wire between Washington and Baltimore annihilated time and space. "
            "For the first time in human history, thought traveled faster than any physical vehicle could carry it."
        ),
    },
    {
        "id": "chunk_05",
        "title": "On the Origin of Species (1859)",
        "source": "On the Origin of Species by Means of Natural Selection (Charles Darwin, 1859)",
        "year": 1859,
        "entities": ["Charles Darwin", "evolution", "Alexander von Humboldt"],
        "text": (
            "As many more individuals of each species are born than can possibly survive, there is a recurring "
            "struggle for existence. Any being, if it vary however slightly in a profitable manner, will have a better chance of surviving."
        ),
    },
    {
        "id": "chunk_06",
        "title": "Sherlock Holmes en Wetenschappelijke Deductie (1890)",
        "source": "The Sign of the Four (Arthur Conan Doyle, 1890)",
        "year": 1890,
        "entities": ["Sherlock Holmes", "Arthur Conan Doyle", "Victorian Era"],
        "text": (
            "Eliminate all other factors, and the one which remains must be the truth. How often have I said to you "
            "that when you have eliminated the impossible, whatever remains, however improbable, must be the truth?"
        ),
    },
    {
        "id": "chunk_07",
        "title": "Dartmouth AI Proposal (1955-1956)",
        "source": "A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence (1955)",
        "year": 1956,
        "entities": ["Dartmouth Workshop 1956", "artificial intelligence", "computer", "Alan Turing"],
        "text": (
            "We propose that a study of artificial intelligence be carried out during the summer of 1956. "
            "The study proceeds on the conjecture that every aspect of learning or feature of intelligence can be so precisely described that a machine can simulate it."
        ),
    },
    {
        "id": "chunk_08",
        "title": "Apollo 11 en het Aardeperspectief (1969)",
        "source": "NASA Mission Commentary: Apollo 11 Lunar Landing (1969)",
        "year": 1969,
        "entities": ["Apollo 11", "space exploration", "climate", "sustainability"],
        "text": (
            "That's one small step for man, one giant leap for mankind. Viewing the fragile, blue marble of Earth "
            "suspended in the infinite blackness of space awakened a profound global awareness of planetary limits."
        ),
    },
]


# ==============================================================================
# 3. GRAPHRAG ENGINE KLASSE
# ==============================================================================
class GraphRAGEngine:
    """GraphRAG Engine voor semantisch zoeken, ego-subgrafen, pathfinding en live Google Ngram data."""

    def __init__(self) -> None:
        self.graph, self.communities = build_historical_knowledge_graph()
        self.chunks = CORPUS_EVIDENCE_CHUNKS
        self._ngram_cache: Dict[str, Dict[str, Any]] = {}

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Geeft alle entiteiten gesorteerd op label terug."""
        nodes_data = []
        for n, attrs in self.graph.nodes(data=True):
            nodes_data.append({"id": n, **attrs})
        return sorted(nodes_data, key=lambda x: x.get("label", x["id"]))

    def get_categories(self) -> List[str]:
        """Geeft alle unieke entiteitcategorieën terug."""
        cats = set()
        for _, attrs in self.graph.nodes(data=True):
            cats.add(attrs.get("category", "Concept"))
        return sorted(list(cats))

    def find_entity_by_query(self, query: str) -> Optional[str]:
        """Zoekt de best passende node ID op basis van een tekstuele zoekterm."""
        q = query.lower().strip()
        if not q:
            return None

        if q in self.graph:
            return q

        for n, attrs in self.graph.nodes(data=True):
            if q == n.lower() or q == attrs.get("label", "").lower() or q == attrs.get("ngram", "").lower():
                return n

        for n, attrs in self.graph.nodes(data=True):
            if q in n.lower() or q in attrs.get("label", "").lower():
                return n

        return None

    def get_entity_live_ngram_metrics(self, entity_id: str) -> Dict[str, Any]:
        """Haalt live Google Ngram statistieken op voor een entiteit (gecached)."""
        if entity_id in self._ngram_cache:
            return self._ngram_cache[entity_id]

        ngram_term = self.graph.nodes[entity_id].get("ngram", entity_id) if entity_id in self.graph else entity_id
        metrics = fetch_live_ngram_metrics(ngram_term)
        self._ngram_cache[entity_id] = metrics
        return metrics

    def add_custom_live_term(self, term: str, category: str = "Live Ngram Concept", connect_to_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Voegt een willekeurige zoekterm uit de Google Books Ngram Viewer live toe aan de kennisgraaf,
        haalt de actuele tijdreeks op en verbindt deze met bestaande concepten.
        """
        clean_t = term.strip()
        if not clean_t:
            return False, "Geen term opgegeven."

        node_id = clean_t.lower()
        if node_id in self.graph:
            return True, f"'{clean_t}' bestaat al in de kennisgraaf."

        # Haal live Ngram data op
        metrics = fetch_live_ngram_metrics(clean_t)
        self._ngram_cache[node_id] = metrics

        peak_yr = metrics.get("peak_year", 2000)
        desc = f"Live toegevoegde Google Books Ngram term. Piekjaar: {peak_yr} ({metrics.get('peak_val_pct', 0):.5f}%)."

        # Voeg node toe
        self.graph.add_node(
            node_id,
            label=clean_t.title(),
            category=category,
            period=f"{peak_yr} (Piek)",
            ngram=clean_t,
            desc=desc,
            color=CATEGORY_COLORS.get(category, "#d81b60"),
            is_live_ingested=True,
            peak_year=peak_yr,
        )

        # Bepaal verbinding: expliciet door gebruiker of op basis van strikte hele-woord match
        connected_node = None
        connect_reason = ""
        if connect_to_id and connect_to_id in self.graph:
            connected_node = connect_to_id
            connect_reason = f"Door gebruiker handmatig gekoppeld aan '{self.graph.nodes[connect_to_id].get('label', connect_to_id)}'."
        else:
            # Alleen koppelen als er een ECHTE hele-woord overeenkomst is met een bestaande entiteit
            t_words = set(clean_t.lower().split())
            for n, attrs in self.graph.nodes(data=True):
                if n != node_id:
                    n_words = set(n.lower().split())
                    label_words = set(attrs.get("label", "").lower().split())
                    common = t_words.intersection(n_words.union(label_words))
                    if common and any(len(w) >= 4 for w in common) and (n_words.issubset(t_words) or t_words.issubset(n_words)):
                        connected_node = n
                        connect_reason = f"Conceptuele en lexicale overlap: deelt trefwoord '{list(common)[0]}' met '{attrs.get('label', n)}'."
                        break

        if connected_node:
            self.graph.add_edge(
                connected_node,
                node_id,
                relation_type="RELATED_TO",
                weight=0.85,
                explanation=connect_reason,
            )
            return True, f"'{clean_t.title()}' succesvol toegevoegd en logisch gekoppeld aan '{self.graph.nodes[connected_node].get('label', connected_node)}' ({connect_reason})!"
        else:
            return True, f"'{clean_t.title()}' succesvol toegevoegd als zelfstandig kenniscluster (geen willekeurige verbindingen geforceerd)."

    def mine_and_ingest_google_books_database(
        self,
        term: str,
        corpus: str = "en-US-2019",
        max_items: int = 5,
        connect_to_existing_id: Optional[str] = None,
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Bevraagt rechtstreeks de officiële Google Books Ngram Viewer database met wildcard expansies,
        haalt de authentieke historische co-occurrenties/collocaties op en integreert ze dynamisch
        in de Kennisgraaf met getypeerde relaties en gewichten.
        Er worden GEEN willekeurige 'tijdsverbindingen' gemaakt: relaties worden uitsluitend
        gebaseerd op authentieke Google Books co-occurrenties of expliciete lexicale overlap.
        """
        clean_t = term.strip()
        if not clean_t:
            return False, "Voer een geldige zoekterm in om de Google Books database te bevragen.", []

        root_id = clean_t.lower()

        # Zorg dat het hoofdconcept bestaat in de graaf
        if root_id not in self.graph:
            metrics = self.get_entity_live_ngram_metrics(clean_t)
            peak_yr = metrics.get("peak_year", 2000)
            pk_pct = metrics.get("peak_val_pct", 0.0)

            # Historische toelichting / disambiguatie bij zeldzame of moderne termen
            if pk_pct < 0.00005:
                hist_note = f"Zeldzame of moderne term in historische boeken (frequentie < 0.00005%). Piek in boeken rond {peak_yr}."
            else:
                hist_note = f"Historisch concept in Google Books ({corpus}). Piekjaar: {peak_yr} ({pk_pct:.5f}% van alle gedrukte boeken)."

            self.graph.add_node(
                root_id,
                label=clean_t.title(),
                category="Concept",
                period=f"{peak_yr} (Piek)" if pk_pct >= 0.00005 else "Recent / Modern",
                ngram=clean_t,
                desc=hist_note,
                color=CATEGORY_COLORS.get("Concept", "#1a73e8"),
                is_database_root=True,
                peak_year=peak_yr,
                peak_val_pct=pk_pct,
            )

            # Optioneel koppelen: expliciet gekozen of op basis van strikte hele-woord match
            if connect_to_existing_id and connect_to_existing_id in self.graph:
                self.graph.add_edge(
                    connect_to_existing_id,
                    root_id,
                    relation_type="USER_ANCHORED_TO",
                    weight=0.85,
                    explanation=f"Door gebruiker handmatig verankerd aan '{self.graph.nodes[connect_to_existing_id].get('label', connect_to_existing_id)}'.",
                )
            else:
                # Controleer of er een echte hele-woord overlap is met een bestaande entiteit
                t_words = set(root_id.split())
                for ex_n, ex_attrs in list(self.graph.nodes(data=True)):
                    if ex_n != root_id:
                        ex_words = set(ex_n.split())
                        if ex_words and (ex_words.issubset(t_words) or t_words.issubset(ex_words)):
                            if any(len(w) >= 4 for w in t_words.intersection(ex_words)):
                                self.graph.add_edge(
                                    ex_n,
                                    root_id,
                                    relation_type="SUB_CONCEPT_OF",
                                    weight=0.88,
                                    explanation=f"Conceptuele deelrelatie: '{ex_attrs.get('label', ex_n)}' is direct verbonden met '{clean_t.title()}'.",
                                )
                                break

        # Mijn associaties uit de Google Books database
        associations = mine_google_ngram_associations(clean_t, corpus=corpus, max_results=max_items)
        if not associations:
            return (
                False,
                f"Geen betrouwbare historische co-locaties gevonden in Google Books voor '{clean_t}'. "
                f"Voor recente termen (zoals computerspellen of moderne fenomenen) heeft Google Books (1800-2019) geen significante historische boekassociaties.",
                [],
            )

        added_labels = []
        for assoc in associations:
            ng = assoc["ngram"]
            nid = ng.lower()
            peak_yr = assoc["peak_year"]
            pk_pct = assoc["peak_val_pct"]

            if nid not in self.graph:
                self.graph.add_node(
                    nid,
                    label=assoc["clean_label"],
                    category="Google Books Database",
                    period=f"{peak_yr} (Piek)",
                    ngram=ng,
                    desc=(
                        f"Geminete co-locatie uit Google Books Ngram Database ({corpus}). "
                        f"Historisch piekjaar: {peak_yr} met een boekfrequentie van {pk_pct:.5f}%."
                    ),
                    color=CATEGORY_COLORS.get("Google Books Database", "#0288d1"),
                    peak_year=peak_yr,
                    peak_val_pct=pk_pct,
                    is_database_ingested=True,
                )
                added_labels.append(assoc["clean_label"])

            # Bepaal richting en gewicht
            edge_weight = min(0.95, max(0.60, pk_pct * 15000))
            if assoc["direction"] == "prefix":
                u, v = nid, root_id
                expl = (
                    f"'{assoc['clean_label']}' trad in Google Books frequent op als vaste combinatie met '{clean_t.title()}' "
                    f"(piekjaar {peak_yr}, boekfrequentie {pk_pct:.5f}%)."
                )
            else:
                u, v = root_id, nid
                expl = (
                    f"'{clean_t.title()}' vormt in Google Books een vaste historische combinatie met '{assoc['clean_label']}' "
                    f"(piekjaar {peak_yr}, boekfrequentie {pk_pct:.5f}%)."
                )

            self.graph.add_edge(
                u,
                v,
                relation_type="CO_OCCURS_IN_GOOGLE_BOOKS",
                weight=round(edge_weight, 3),
                explanation=expl,
            )

            # Automatische kruisverbindingen uitsluitend bij strikte hele-woord overeenkomst (minimaal 4 letters)
            nid_words = set(nid.split())
            for existing_n, attrs in list(self.graph.nodes(data=True)):
                if existing_n not in (nid, root_id):
                    ex_words = set(existing_n.split())
                    common = nid_words.intersection(ex_words)
                    if common and any(len(w) >= 4 for w in common) and (ex_words.issubset(nid_words) or nid_words.issubset(ex_words)):
                        self.graph.add_edge(
                            existing_n,
                            nid,
                            relation_type="SUB_CONCEPT_OF",
                            weight=0.75,
                            explanation=f"Conceptuele verwantschap via trefwoord '{list(common)[0]}' met '{attrs.get('label', existing_n)}'.",
                        )

        msg = (
            f"Succesvol {len(associations)} historische associaties gemined uit Google Books Ngram Database "
            f"voor '{clean_t.title()}': {', '.join([a['clean_label'] for a in associations])}."
        )
        return True, msg, associations

    # --------------------------------------------------------------------------
    # MODUS A: LOCAL SEARCH (Ego-graaf & directe relaties)
    # --------------------------------------------------------------------------
    def local_search(self, target_entity: str, max_hops: int = 1) -> Dict[str, Any]:
        """Voert Local Search uit met verrijking door live Google Ngram statistieken."""
        node_id = self.find_entity_by_query(target_entity)
        if not node_id or node_id not in self.graph:
            return {
                "found": False,
                "query": target_entity,
                "message": f"Geen directe entiteit gevonden voor '{target_entity}'.",
            }

        node_attrs = self.graph.nodes[node_id]

        # Ego-subgraaf extraheren
        undirected = self.graph.to_undirected()
        ego_sub = nx.ego_graph(undirected, node_id, radius=max_hops)

        # Relaties
        out_edges = []
        for _, dst, data in self.graph.out_edges(node_id, data=True):
            dst_attrs = self.graph.nodes[dst]
            out_edges.append(
                {
                    "direction": "uitgaand",
                    "target_id": dst,
                    "target_label": dst_attrs.get("label", dst),
                    "target_category": dst_attrs.get("category", ""),
                    "relation_type": data.get("relation_type", "RELATED_TO"),
                    "weight": data.get("weight", 0.5),
                    "explanation": data.get("explanation", ""),
                }
            )

        in_edges = []
        for src, _, data in self.graph.in_edges(node_id, data=True):
            src_attrs = self.graph.nodes[src]
            in_edges.append(
                {
                    "direction": "inkomend",
                    "source_id": src,
                    "source_label": src_attrs.get("label", src),
                    "source_category": src_attrs.get("category", ""),
                    "relation_type": data.get("relation_type", "RELATED_TO"),
                    "weight": data.get("weight", 0.5),
                    "explanation": data.get("explanation", ""),
                }
            )

        # Tekstuele bewijzen
        related_chunks = []
        all_sub_nodes = set(ego_sub.nodes())
        for chk in self.chunks:
            overlap = set(chk["entities"]).intersection(all_sub_nodes)
            if node_id in chk["entities"] or overlap:
                relevance = 2.0 if node_id in chk["entities"] else 1.0 + (len(overlap) * 0.2)
                related_chunks.append({**chk, "relevance": relevance, "overlap_entities": list(overlap)})

        related_chunks.sort(key=lambda x: x["relevance"], reverse=True)

        # Community
        assigned_community = None
        for c_id, c_data in self.communities.items():
            if node_id in c_data["nodes"]:
                assigned_community = c_data
                break

        # Graaf-centraliteit
        deg_centrality = nx.degree_centrality(self.graph).get(node_id, 0.0)
        pagerank = nx.pagerank(self.graph).get(node_id, 0.0)

        # Live Google Ngram Metrics ophalen
        ngram_metrics = self.get_entity_live_ngram_metrics(node_id)

        # Gesynthetiseerd RAG-antwoord
        synthesis = self._synthesize_local_answer(
            node_attrs, out_edges, in_edges, assigned_community, related_chunks, ngram_metrics
        )

        return {
            "found": True,
            "entity_id": node_id,
            "label": node_attrs.get("label", node_id),
            "category": node_attrs.get("category", ""),
            "period": node_attrs.get("period", ""),
            "ngram": node_attrs.get("ngram", node_id),
            "description": node_attrs.get("desc", ""),
            "color": node_attrs.get("color", "#1a73e8"),
            "community": assigned_community,
            "ngram_metrics": ngram_metrics,
            "metrics": {
                "degree": self.graph.degree(node_id),
                "degree_centrality": round(deg_centrality, 3),
                "pagerank": round(pagerank, 4),
                "neighbors_count": len(ego_sub.nodes()) - 1,
            },
            "outgoing_relations": out_edges,
            "incoming_relations": in_edges,
            "evidence_chunks": related_chunks[:3],
            "synthesis": synthesis,
            "subgraph_nodes": list(ego_sub.nodes()),
        }

    # --------------------------------------------------------------------------
    # MODUS B: GLOBAL SEARCH (Thematische communities)
    # --------------------------------------------------------------------------
    def global_search(self, user_query: str) -> Dict[str, Any]:
        """Voert Global Search uit over thematische gemeenschappen."""
        q_tokens = set(user_query.lower().split())

        scored_communities = []
        for c_id, c_data in self.communities.items():
            match_score = 0
            text_blob = (c_data["title"] + " " + c_data["summary"] + " " + " ".join(c_data["key_drivers"])).lower()

            for tok in q_tokens:
                if len(tok) > 2 and tok in text_blob:
                    match_score += 1

            for node_name in c_data["nodes"]:
                if node_name.lower() in user_query.lower():
                    match_score += 3

            scored_communities.append({"id": c_id, **c_data, "match_score": match_score})

        scored_communities.sort(key=lambda x: x["match_score"], reverse=True)
        top_community = scored_communities[0] if scored_communities else None

        if top_community and top_community["match_score"] > 0:
            summary_text = (
                f"### {top_community['title']}\n\n"
                f"**Primaire historische periode:** `{top_community['dominant_era']}`  \n"
                f"**Drijvende krachten:** {', '.join(top_community['key_drivers'])}\n\n"
                f"{top_community['summary']}"
            )
        else:
            summary_text = (
                "### Overkoepelende Kennisgraaf Synthese\n\n"
                "De kennisgraaf verbindt wetenschappelijke revoluties (stoomkracht, relativiteit, kwantumfysica), "
                "communicatietechnologieën (telegraaf, computer, AI), literatuur (Frankenstein, Sherlock Holmes) "
                "en politieke maatschappijvormen (Verlichting, democratie, mensenrechten) in één samenhangend netwerk."
            )

        return {
            "query": user_query,
            "top_community": top_community,
            "all_communities": list(self.communities.values()),
            "synthesis": summary_text,
        }

    # --------------------------------------------------------------------------
    # MODUS C: MULTI-HOP PATHFINDING (Conceptoverbrugging)
    # --------------------------------------------------------------------------
    def find_relational_path(self, source_term: str, target_term: str) -> Dict[str, Any]:
        """Vindt het kortste en causale historische pad tussen twee concepten."""
        src_id = self.find_entity_by_query(source_term)
        dst_id = self.find_entity_by_query(target_term)

        if not src_id or not dst_id:
            return {"success": False, "message": f"Kon één of beide concepten niet vinden."}

        undirected = self.graph.to_undirected()

        try:
            shortest_path = nx.shortest_path(undirected, source=src_id, target=dst_id)
        except nx.NetworkXNoPath:
            src_lbl = self.graph.nodes[src_id].get("label", src_id) if src_id in self.graph else src_id
            dst_lbl = self.graph.nodes[dst_id].get("label", dst_id) if dst_id in self.graph else dst_id
            return {
                "success": False,
                "message": (
                    f"Geen historisch of causaal pad gevonden tussen '{src_lbl}' en '{dst_lbl}'. "
                    f"Deze concepten behoren tot gescheiden historische domeinen zonder geverifieerde co-occurrentie."
                ),
            }

        steps = []
        for i in range(len(shortest_path) - 1):
            u, v = shortest_path[i], shortest_path[i + 1]
            if self.graph.has_edge(u, v):
                edata = self.graph[u][v]
                direction = "->"
            elif self.graph.has_edge(v, u):
                edata = self.graph[v][u]
                direction = "<-"
            else:
                edata = {"relation_type": "CONNECTED_WITH", "explanation": "Historisch verbonden"}
                direction = "--"

            u_lbl = self.graph.nodes[u].get("label", u)
            v_lbl = self.graph.nodes[v].get("label", v)
            steps.append(
                {
                    "step": i + 1,
                    "from_label": u_lbl,
                    "to_label": v_lbl,
                    "direction": direction,
                    "relation_type": edata.get("relation_type", "RELATED_TO"),
                    "explanation": edata.get("explanation", ""),
                }
            )

        labels = [self.graph.nodes[n].get("label", n) for n in shortest_path]
        chain_visual = " -> ".join([f"`{lbl}`" for lbl in labels])

        return {
            "success": True,
            "source_id": src_id,
            "target_id": dst_id,
            "path_nodes": shortest_path,
            "path_length": len(shortest_path) - 1,
            "steps": steps,
            "chain_visual": chain_visual,
        }

    # --------------------------------------------------------------------------
    # VISUALISATIE DATA EXPORT (VIS.JS)
    # --------------------------------------------------------------------------
    def get_visualization_json(
        self,
        highlighted_nodes: Optional[Set[str]] = None,
        filter_category: Optional[str] = None,
    ) -> str:
        """Exporteert de kennisgraaf als JSON met optionele categoriefiltering."""
        vis_nodes = []
        visible_node_ids = set()

        for n, data in self.graph.nodes(data=True):
            cat = data.get("category", "Concept")
            if filter_category and filter_category != "Alles" and cat != filter_category:
                continue

            visible_node_ids.add(n)
            is_hl = highlighted_nodes is not None and n in highlighted_nodes
            base_color = data.get("color", "#1a73e8")

            vis_nodes.append(
                {
                    "id": n,
                    "label": data.get("label", n),
                    "title": f"<b>{data.get('label', n)}</b><br>Categorie: {cat}<br>Tijdvak: {data.get('period')}<br>{data.get('desc')}",
                    "color": {
                        "background": base_color if is_hl or not highlighted_nodes else "#f1f3f4",
                        "border": base_color,
                        "highlight": {"background": base_color, "border": "#202124"},
                    },
                    "borderWidth": 3 if is_hl else 1.5,
                    "size": 24 if is_hl else 16,
                    "font": {"size": 12, "color": "#202124", "face": "Google Sans, Roboto, sans-serif"},
                    "shape": "dot",
                    "category": cat,
                }
            )

        vis_edges = []
        for u, v, data in self.graph.edges(data=True):
            if u not in visible_node_ids or v not in visible_node_ids:
                continue

            is_edge_hl = highlighted_nodes is not None and u in highlighted_nodes and v in highlighted_nodes
            vis_edges.append(
                {
                    "from": u,
                    "to": v,
                    "label": data.get("relation_type", "").replace("_", " ").lower(),
                    "title": data.get("explanation", ""),
                    "arrows": "to",
                    "color": {"color": "#1a73e8" if is_edge_hl else "#dadce0", "highlight": "#1a73e8"},
                    "width": 2.5 if is_edge_hl else 1.0,
                    "font": {"size": 9, "color": "#5f6368", "align": "middle"},
                }
            )

        return json.dumps({"nodes": vis_nodes, "edges": vis_edges})

    # --------------------------------------------------------------------------
    # RAG SYNTHESE FORMATTER
    # --------------------------------------------------------------------------
    def _synthesize_local_answer(
        self,
        node_attrs: Dict[str, Any],
        out_edges: List[Dict[str, Any]],
        in_edges: List[Dict[str, Any]],
        community: Optional[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        ngram_metrics: Dict[str, Any],
    ) -> str:
        name = node_attrs.get("label", "")
        desc = node_attrs.get("desc", "")
        category = node_attrs.get("category", "")
        period = node_attrs.get("period", "")

        out_summary = ", ".join([f"`{e['target_label']}` ({e['relation_type'].replace('_', ' ').lower()})" for e in out_edges[:3]])
        in_summary = ", ".join([f"`{e['source_label']}` ({e['relation_type'].replace('_', ' ').lower()})" for e in in_edges[:3]])

        peak_val = ngram_metrics.get("peak_val_pct", 0.0)
        peak_str = f"**{ngram_metrics.get('peak_year')}** ({peak_val:.5f}%)" if ngram_metrics.get("success") else "Rond 1990"
        trend_str = ngram_metrics.get("trend", "➡️ Stabiel")

        # Contextuele waarschuwing / verduidelijking bij anachronistische of zeldzame termen
        disambiguation_note = ""
        if peak_val < 0.00005:
            disambiguation_note = (
                "\n\n> [Historische Context] Deze term kent een zeer lage frequentie in de historische "
                "Google Books index (1800-2019). Eventuele vroege historische vermeldingen kunnen homoniemen "
                "of vaktermen zijn (zoals 'mine craft' = marinevaartuigen/mijnenvegers), niet het moderne fenomeen.\n"
            )

        return (
            f"#### {name}\n"
            f"*Categorie:* `{category}` • *Historisch Tijdvak:* `{period}` • *Google Ngram Piek:* {peak_str}  \n"
            f"*Live Trend:* {trend_str}\n\n"
            f"**Context:**  \n{desc}{disambiguation_note}\n\n"
            f"**Graafrelaties:**  \n"
            f"* **Beïnvloedt direct:** {out_summary or 'Geen uitgaande paden geregistreerd.'}\n"
            f"* **Gedreven door:** {in_summary or 'Geen inkomende paden geregistreerd.'}"
        )


# ==============================================================================
# 4. VIS.JS INTERACTIVE CANVAS GENERATOR (CLEAN & MINIMALIST)
# ==============================================================================
def generate_visjs_html(graph_json: str, height_px: int = 460) -> str:
    """Genereert een opgeruimd, minimalistisch Vis.js netwerkcanvas."""
    return f"""
    <!DOCTYPE html>
    <html lang="nl">
    <head>
        <meta charset="utf-8">
        <title>GraphRAG Visualizer</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: 'Google Sans', Roboto, sans-serif;
                background-color: #ffffff;
                color: #202124;
                overflow: hidden;
            }}
            #network-container {{
                width: 100%;
                height: {height_px}px;
                border: 1px solid #dadce0;
                border-radius: 8px;
                background: #fdfdfd;
                position: relative;
            }}
            .node-tooltip-card {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                max-width: 360px;
                background: rgba(255, 255, 255, 0.97);
                border: 1px solid #dadce0;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                box-shadow: 0 1px 4px rgba(60,64,67,.18);
                z-index: 10;
                display: none;
                pointer-events: none;
            }}
            .node-title {{
                font-weight: 600;
                color: #1a73e8;
                margin-bottom: 2px;
            }}
        </style>
    </head>
    <body>
        <div id="network-container">
            <div id="node-tooltip-card" class="node-tooltip-card">
                <div id="card-title" class="node-title"></div>
                <div id="card-desc" style="color: #5f6368; line-height: 1.35;"></div>
            </div>
        </div>

        <script type="text/javascript">
            const graphData = {graph_json};
            const container = document.getElementById('network-container');
            const tooltipCard = document.getElementById('node-tooltip-card');
            const cardTitle = document.getElementById('card-title');
            const cardDesc = document.getElementById('card-desc');

            const data = {{
                nodes: new vis.DataSet(graphData.nodes),
                edges: new vis.DataSet(graphData.edges)
            }};

            const options = {{
                nodes: {{
                    shape: 'dot',
                    scaling: {{ min: 14, max: 28 }},
                    shadow: {{ enabled: true, color: 'rgba(0,0,0,0.05)', size: 4, x: 1, y: 1 }}
                }},
                edges: {{
                    arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
                    smooth: {{ type: 'continuous', roundness: 0.18 }}
                }},
                physics: {{
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {{
                        gravitationalConstant: -50,
                        centralGravity: 0.012,
                        springLength: 95,
                        springConstant: 0.08,
                        damping: 0.45
                    }},
                    maxVelocity: 40,
                    minVelocity: 0.1,
                    stabilization: {{ iterations: 90 }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 120,
                    zoomView: true,
                    dragView: true
                }}
            }};

            const network = new vis.Network(container, data, options);

            network.on("hoverNode", function(params) {{
                const nodeObj = graphData.nodes.find(n => n.id === params.node);
                if (nodeObj) {{
                    cardTitle.innerText = nodeObj.label + ' (' + (nodeObj.category || 'Entiteit') + ')';
                    cardDesc.innerHTML = nodeObj.title || '';
                    tooltipCard.style.display = 'block';
                }}
            }});

            network.on("blurNode", function() {{
                tooltipCard.style.display = 'none';
            }});
        </script>
    </body>
    </html>
    """
