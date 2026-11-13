"""
================================================================================
INTERACTIEF STREAMLIT PROTOTYPE: REDESIGNED GOOGLE NGRAM 'VISUAL BUILDER'
================================================================================
Bestand : app.py
Doel    : Interactief UX-prototype om de nieuwe 'Visual Builder'-interface van
          de Google Books Ngram Viewer te testen met synthetische data
          (synthetic_ngram_data.csv).

KERNPRINCIPES VAN DIT HERONTWERP (BIG DATA DESIGN & UX):
1. Recognition rather than Recall (Herkenning boven herinnering):
   Gebruikers hoeven geen cryptische programmeersyntax meer uit het hoofd te
   leren (zoals '_NOUN', '_INF' of ':eng_us_2019'). Alle opties zijn vertaald
   naar zichtbare, begrijpelijke UI-controls (visuele knoppen, dropdowns, toggles).
2. Feedforward & Signifiers:
   Tooltips (help="") bij sliders en filters leggen vóór interactie in gewone
   mensentaal uit wat een parameter (zoals 'Smoothing') met de grafiek doet.
3. Constructieve Foutafhandeling (Van 'Dead-End Feedback' naar 'Feedforward'):
   De beruchte foutmelding 'No valid ngrams to plot!' is vervangen door slimme
   foutdetectie (typfouten, oude syntax, corpus-mismatch) met een vriendelijke
   st.warning(), klikbare suggesties en een stabiele placeholder-grafiek.
4. Contextuele Data-Transparantie:
   Naast abstracte percentages (relative_frequency) krijgt de gebruiker direct
   inzicht in absolute aantallen (raw_match_count), het totale boekvolume
   (total_books_in_corpus) en historische gebeurtenissen (historical_event)
   die pieken op de tijdlijn verklaren.
================================================================================
"""

import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

# ==============================================================================
# PAGINA-CONFIGURATIE
# ==============================================================================
# UX-keuze: 'wide' layout geeft ruimte aan een overzichtelijk filterpaneel
# naast of boven een brede tijdlijngrafiek, waardoor visuele drukte (clutter)
# wordt beperkt en de gebruiker alle filters in één oogopslag ziet.
st.set_page_config(
    page_title="Ngram Viewer — Visual Builder Prototype",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(__file__).parent / "synthetic_ngram_data.csv"


# ==============================================================================
# DATA INLADEN & CACHING
# ==============================================================================
@st.cache_data
def load_ngram_data(csv_path: Path) -> pd.DataFrame:
    """
    Laadt de synthetische Ngram-dataset in en cached deze in het geheugen.

    UX-belang:
    Dankzij @st.cache_data reageren de tijdlijn-slider en smoothing-slider
    direct (binnen milliseconden) zonder dat het CSV-bestand bij elke klik
    opnieuw van schijf gelezen hoeft te worden (Direct Manipulation).
    """
    df = pd.read_csv(csv_path)
    # Zorg dat lege waarden in historical_event netjes als None/NaN worden behandeld
    df["historical_event"] = df["historical_event"].where(df["historical_event"].notna(), None)
    return df


# ==============================================================================
# CONSTRUCTIEVE FOUTAFHANDELING (FEEDFORWARD LOGICA)
# ==============================================================================
# Woordenboek met veelvoorkomende spelfouten en oude programmeersyntax.
# UX-doel: Voorkomt frustratie bij kleine menselijke fouten en begeleidt de
# gebruiker direct naar een werkende zoekopdracht.
KNOWN_TYPOS_AND_SYNTAX: Dict[str, Tuple[str, str]] = {
    "climmate": (
        "climate",
        "Er lijkt een dubbele 'm' in je zoekterm te zitten.",
    ),
    "climat": (
        "climate",
        "Mis je wellicht de letter 'e' aan het einde?",
    ),
    "democracie": (
        "democracy",
        "Je gebruikte de Nederlandse/Franse schrijfwijze. In deze dataset staat de Engelse stam 'democracy'.",
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
        "De afkorting 'AI' is in deze dataset opgenomen onder de volledige term 'artificial intelligence'.",
    ),
    "telegramm": (
        "telegram",
        "Dit lijkt op de Duitse spelling met dubbel 'm'.",
    ),
}


def validate_and_suggest_query(
    raw_query: str,
    available_terms: List[str],
    case_sensitive: bool,
) -> Dict[str, Optional[str]]:
    """
    Controleert of de ingevoerde zoekterm bestaat in de dataset en genereert
    anders een constructieve feedforward-suggestie.

    UX-principe (Foutpreventie & Herstel):
    In de originele Google Ngram Viewer leidt een kleine spelfout of verkeerde
    operator tot de cryptische melding 'No valid ngrams to plot!'.
    Deze functie vertaalt zo'n fout naar:
    1. Wat er precies aan de hand is (in begrijpelijke taal).
    2. Welke term de gebruiker waarschijnlijk bedoelde ('Bedoelde je ...?').
    """
    cleaned = raw_query.strip()

    if not cleaned:
        return {
            "is_valid": False,
            "matched_term": None,
            "suggestion": "climate",
            "reason": "Het zoekveld is momenteel leeg. Voer een zoekterm in om de tijdlijn te verkennen.",
        }

    # Check 1: Heeft de gebruiker nog een oude POS-operator zoals '_NOUN' of '_VERB' getypt?
    for legacy_tag in ["_NOUN", "_VERB", "_INF", "_ADJ", "_ADV"]:
        if cleaned.upper().endswith(legacy_tag):
            base_word = cleaned[: -len(legacy_tag)].strip()
            matched_suggestion = base_word if base_word in available_terms else "climate"
            return {
                "is_valid": False,
                "matched_term": None,
                "suggestion": matched_suggestion,
                "reason": (
                    f"Je hebt de oude programmeersyntax `{legacy_tag}` in de zoekbalk getypt. "
                    f"Dat is niet meer nodig! Zoek gewoon op **'{matched_suggestion}'** en gebruik "
                    f"het visuele **Woordsoort-filter** hieronder."
                ),
            }

    # Check 2: Bestaat de zoekterm in de dataset?
    # Als case_sensitive UIT staat, accepteren we ook een match ongeacht hoofdletters.
    if case_sensitive:
        if cleaned in available_terms:
            return {"is_valid": True, "matched_term": cleaned, "suggestion": None, "reason": None}
        # Bestaat het woord wél met andere hoofdletters (bijv. 'CLIMATE' of 'paris' vs 'Paris')?
        lower_map = {t.lower(): t for t in available_terms}
        if cleaned.lower() in lower_map:
            alt_case = lower_map[cleaned.lower()]
            return {
                "is_valid": False,
                "matched_term": None,
                "suggestion": alt_case,
                "reason": (
                    f"Hoofdlettergevoeligheid staat AAN, en `{cleaned}` komt niet in exact deze "
                    f"schrijfwijze voor. Zet hoofdlettergevoeligheid uit of gebruik **'{alt_case}'**."
                ),
            }
    else:
        # Case-insensitive matching: zoek een beschikbare term die overeenkomt
        for term in available_terms:
            if term.lower() == cleaned.lower():
                return {"is_valid": True, "matched_term": term, "suggestion": None, "reason": None}

    # Check 3: Bekende typfout in onze dictionary?
    lookup = cleaned.lower()
    if lookup in KNOWN_TYPOS_AND_SYNTAX:
        suggested_word, explanation = KNOWN_TYPOS_AND_SYNTAX[lookup]
        return {
            "is_valid": False,
            "matched_term": None,
            "suggestion": suggested_word,
            "reason": explanation,
        }

    # Check 4: Fuzzy string matching (Levenshtein/SequenceMatcher via difflib)
    # voor willekeurige andere typfouten (bijv. 'democrasy', 'telegran', 'climte').
    close_matches = difflib.get_close_matches(
        lookup,
        [t.lower() for t in available_terms],
        n=1,
        cutoff=0.5,
    )
    if close_matches:
        best_lower = close_matches[0]
        # Haal de originele schrijfwijze op
        best_term = next(t for t in available_terms if t.lower() == best_lower)
        return {
            "is_valid": False,
            "matched_term": None,
            "suggestion": best_term,
            "reason": f"De term `{cleaned}` komt niet voor in het corpus, maar lijkt sterk op een bestaande term.",
        }

    # Check 5: Volledig onbekend woord -> bied een constructief startpunt aan
    return {
        "is_valid": False,
        "matched_term": None,
        "suggestion": "climate",
        "reason": (
            f"De zoekterm `{cleaned}` is niet aanwezig in deze prototype-dataset. "
            f"Beschikbare testtermen zijn: {', '.join(f'**{t}**' for t in available_terms)}."
        ),
    }


def render_placeholder_chart(year_range: Tuple[int, int]) -> alt.Chart:
    """
    Genereert een rustige, lege 'placeholder'-grafiek met de juiste X- en Y-assen.

    UX-belang (Voorkomen van Layout Shift & Desoriëntatie):
    Wanneer een gebruiker een typfout maakt, verdwijnt in veel applicaties plotseling
    het hele canvas of verschijnt een rode foutpagina. Door een lege grafiek met
    een subtiele instructie op het canvas te tonen, blijft de visuele structuur
    van de pagina volledig intact terwijl de gebruiker de suggestie aanklikt.
    """
    placeholder_df = pd.DataFrame(
        {
            "year": [year_range[0], year_range[1]],
            "relative_frequency": [0.0, 0.05],
        }
    )

    base = (
        alt.Chart(placeholder_df)
        .mark_line(opacity=0)
        .encode(
            x=alt.X(
                "year:Q",
                title="Jaartal",
                scale=alt.Scale(domain=[year_range[0], year_range[1]]),
                axis=alt.Axis(format="d"),
            ),
            y=alt.Y(
                "relative_frequency:Q",
                title="Relatieve frequentie (%)",
                scale=alt.Scale(domain=[0, 0.06]),
            ),
        )
        .properties(height=400, title="Wachten op een geldige zoekterm...")
    )

    # Subtiele tekst-annotatie midden op het lege canvas
    overlay_text = (
        alt.Chart(
            pd.DataFrame(
                {
                    "year": [(year_range[0] + year_range[1]) / 2],
                    "relative_frequency": [0.03],
                    "label": ["Geen data om te plotten — kies hierboven een voorgestelde zoekterm"],
                }
            )
        )
        .mark_text(fontSize=15, color="#7f8c8d", fontStyle="italic")
        .encode(x="year:Q", y="relative_frequency:Q", text="label:N")
    )

    return base + overlay_text


# ==============================================================================
# HOOFDAPPLICATIE (UI LAYOUT & LOGICA)
# ==============================================================================
def main() -> None:
    # Controleer of het CSV-bestand aanwezig is
    if not DATA_FILE.exists():
        st.error(
            f"Bestand `{DATA_FILE.name}` niet gevonden! "
            "Voer eerst `python generate_synthetic_ngram_data.py` uit om de testdata te genereren."
        )
        st.stop()

    df = load_ngram_data(DATA_FILE)
    available_ngrams: List[str] = sorted(df["ngram"].unique().tolist(), key=lambda s: (s.lower(), s))
    available_regions: List[str] = df["corpus_region"].unique().tolist()

    # Initialiseer session_state voor de zoekbalk zodat suggestieknoppen ('Bedoelde je...?')
    # de zoekbalk met één klik kunnen corrigeren (verlaagt interactiekosten).
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = "climate"

    # --------------------------------------------------------------------------
    # KOPTEKST & INTRODUCTIE
    # --------------------------------------------------------------------------
    st.title("📈 Google Books Ngram Viewer — Visual Builder")
    st.caption(
        "Interactief UX-prototype (Big Data Design) • Vervangt complexe zoek-operatoren "
        "door visuele filters, feedforward-uitleg en historische context."
    )

    # ==========================================================================
    # SECTIE 1: VISUELE FILTERS (VERVANGING VAN PROGRAMMEERSYNTAX)
    # ==========================================================================
    # UX-toelichting:
    # In de traditionele Ngram Viewer moet een onderzoeker één cryptische string
    # typen zoals 'climate_NOUN:eng_us_2019'. Dit vereist dat de gebruiker de
    # exacte codes uit het hoofd kent (hoge cognitieve belasting).
    # Hier splitsen we die vraag op in losse, herkenbare visuele bouwstenen.
    with st.container(border=True):
        st.subheader("1. Zoekopdracht & Visuele Filters")

        col_search, col_pos, col_region = st.columns([2.2, 1.8, 2.0])

        with col_search:
            # UX-element: Duidelijke zoekbalk zonder verborgen operator-eisen
            user_query = st.text_input(
                label="🔍 Zoekterm (Ngram)",
                key="search_query",
                placeholder="Bijv. climate, democracy, artificial intelligence, telegram, Paris...",
                help=(
                    "Typ een woord of woordgroep zonder programmeercodes. "
                    "Tip: probeer ook eens een opzettelijke typfout zoals 'climmate' of 'democracie' "
                    "om de nieuwe feedforward-foutafhandeling te testen!"
                ),
            )

            # UX-element: Visuele checkbox voor hoofdlettergevoeligheid (case_sensitive)
            # Maakt het verschil tussen bijv. 'Paris' (de stad) en 'paris' direct testbaar.
            case_sensitive = st.checkbox(
                label="Hoofdlettergevoelig zoeken (`case_sensitive`)",
                value=False,
                help=(
                    "Aan: Maakt strikt onderscheid tussen bijv. 'Paris' (met hoofdletter, vaak de stad) "
                    "en 'paris' (kleine letter).\n\n"
                    "Uit: Combineert alle schrijfwijzen van het woord."
                ),
            )

        with col_pos:
            # UX-element: Mensvriendelijke dropdown voor woordsoorten (POS-tags).
            # In plaats van dat de gebruiker '_NOUN' of '_INF' moet onthouden, kiest
            # men uit begrijpelijke taalkundige categorieën. De onderliggende code
            # vertaalt dit pas op de achtergrond naar de juiste dataset-waarde.
            pos_label_to_code = {
                "Alle woordsoorten (Geen filter)": "NONE",
                "Zelfstandig naamwoord (bijv. 'het klimaat' — _NOUN)": "_NOUN",
                "Werkwoord - vervoegd (bijv. 'verandert' — _VERB)": "_VERB",
                "Hele werkwoord / Infinitief (_INF)": "_INF",
                "Bijvoeglijk naamwoord (Gesimuleerd via _NOUN-profiel)": "_ADJ_SIM",
            }
            selected_pos_label = st.selectbox(
                label="🏷️ Woordsoort & Vervoeging (`pos_tag`)",
                options=list(pos_label_to_code.keys()),
                index=0,
                help=(
                    "Filter het gebruik van een woord op zijn grammaticale rol in de zin. "
                    "Dit vervangt de oude `_NOUN`, `_VERB` en `_INF` syntax-operatoren."
                ),
            )
            selected_pos_code = pos_label_to_code[selected_pos_label]

        with col_region:
            # UX-element: Multiselect voor geografische regio's / corpora.
            # Door een multiselect te bieden kan een gebruiker moeiteloos meerdere
            # taalgebieden (bijv. American English vs. British English vs. Dutch)
            # in dezelfde grafiek met elkaar vergelijken (kruisreferentie).
            selected_regions = st.multiselect(
                label="🌍 Geografisch Corpus (`corpus_region`)",
                options=available_regions,
                default=["American English", "British English"],
                help=(
                    "Selecteer één of meerdere geografische boekcollecties (corpora) "
                    "om cultuur- en taalverschillen over de tijd te vergelijken."
                ),
            )

        # Snelle knoppen (Quick-Select Chips) om de drempel voor exploratie te verlagen
        st.caption(
            "💡 **Snel testen:** Klik op een term of edge-case om de interface direct uit te proberen:"
        )
        chip_cols = st.columns(7)
        quick_examples = [
            ("🌡️ climate", "climate"),
            ("🗳️ democracy", "democracy"),
            ("🤖 artificial intelligence", "artificial intelligence"),
            ("📠 telegram", "telegram"),
            ("🗼 Paris", "Paris"),
            ("⚠️ Typfout: climmate", "climmate"),
            ("⚠️ Syntax: climate_NOUN", "climate_NOUN"),
        ]
        for idx, (btn_label, example_val) in enumerate(quick_examples):
            with chip_cols[idx]:
                if st.button(btn_label, use_container_width=True, key=f"chip_{idx}"):
                    st.session_state["search_query"] = example_val
                    st.rerun()

    # ==========================================================================
    # SECTIE 2: TIJDLIJN & SMOOTHING (DATA MANIPULATIE MET FEEDFORWARD)
    # ==========================================================================
    # UX-toelichting:
    # Tijdreeksanalyse vereist twee veelgebruikte manipulaties: inzoomen op een
    # specifiek historisch tijdvak én het scheiden van jaarlijkse ruis van de
    # langetermijntrend. Beide zitten hier direct boven de grafiek met heldere
    # feedforward-tooltips.
    with st.container(border=True):
        st.subheader("2. Tijdlijn-bereik & Trend-afvlakking (Smoothing)")
        col_timeline, col_smoothing = st.columns([2.5, 1.5])

        with col_timeline:
            # UX-element: Dubbele range-slider voor de tijdsperiode (1800 t/m 2026).
            # Maakt direct visueel inzoomen op bijv. de Tweede Wereldoorlog of de
            # 21e eeuw mogelijk zonder jaartallen in losse tekstvakjes te moeten typen.
            year_range: Tuple[int, int] = st.slider(
                label="📅 Selecteer tijdsperiode (`year`)",
                min_value=int(df["year"].min()),
                max_value=int(df["year"].max()),
                value=(1800, 2026),
                step=1,
                help=(
                    "Versleep de linker- en rechtergrens om in te zoomen op specifieke "
                    "historische decennia of exacte jaartallen tussen 1800 en 2026."
                ),
            )

        with col_smoothing:
            # UX-element: Smoothing-slider (0 - 50) met begrijpelijke feedforward help-tekst.
            # Verlaagt de statistische drempel: niet elke gebruiker weet wat een
            # 'moving average window' is, maar wel dat kortstondige pieken worden gladgestreken.
            smoothing_window: int = st.slider(
                label="🌊 Smoothing / Afvlakking (0 = ruwe data, 50 = sterke macro-trend)",
                min_value=0,
                max_value=50,
                value=3,
                step=1,
                help=(
                    "**Wat doet Smoothing?**\n\n"
                    "Berekent een voortschrijdend gemiddelde (*moving average*) over het gekozen "
                    "aantal omliggende jaren. Dit strijkt kortstondige jaarlijkse pieken en ruis "
                    "glad, zodat grote **macro-trends** over langere periodes duidelijk zichtbaar worden.\n\n"
                    "• **0:** Toont de exacte, onbewerkte jaarcijfers (inclusief scherpe pieken).\n"
                    "• **5 - 15:** Goede balans tussen historische pieken en een rustige lijn.\n"
                    "• **25 - 50:** Toont uitsluitend de grote langetermijnbeweging."
                ),
            )

    # ==========================================================================
    # SECTIE 3: FOUTAFHANDELING (VAN 'DEAD-END FEEDBACK' NAAR 'FEEDFORWARD')
    # ==========================================================================
    validation = validate_and_suggest_query(
        raw_query=user_query,
        available_terms=available_ngrams,
        case_sensitive=case_sensitive,
    )

    if not selected_regions:
        st.warning(
            "🌍 **Geen geografisch corpus geselecteerd.** "
            "Kies hierboven ten minste één regio (bijv. *American English* of *Dutch*) om data te tonen."
        )
        st.altair_chart(render_placeholder_chart(year_range), use_container_width=True)
        st.stop()

    if not validation["is_valid"]:
        suggested = validation["suggestion"]
        reason = validation["reason"]

        # UX-element: Vriendelijke, constructieve waarschuwing i.p.v. 'No valid ngrams to plot!'
        st.warning(
            f"💡 **Geen exacte match gevonden voor `{user_query}`**\n\n"
            f"{reason}\n\n"
            f"👉 **Bedoelde je misschien: `{suggested}`?**"
        )

        # Directe herstelactie (1-klik knop): de gebruiker hoeft de suggestie niet eens
        # zelf over te typen. Dit minimaliseert de 'gulfs of execution & evaluation'.
        col_fix_btn, _ = st.columns([1.5, 4.5])
        with col_fix_btn:
            if st.button(f"✅ Zoek direct op '{suggested}'", type="primary", use_container_width=True):
                st.session_state["search_query"] = suggested
                st.rerun()

        # Toon de stabiele lege placeholder-grafiek zodat de pagina niet 'inklapt'
        st.altair_chart(render_placeholder_chart(year_range), use_container_width=True)
        st.stop()

    # ==========================================================================
    # SECTIE 4: DATA FILTEREN, SMOOTHING BEREKENEN & VISUALISEREN
    # ==========================================================================
    matched_ngram = validation["matched_term"]

    # Indien de gebruiker het educatieve 'Bijvoeglijk naamwoord'-voorbeeld selecteert,
    # leiden we dit af uit het '_NOUN'-profiel met een schaalfactor en tonen we een
    # korte toelichting.
    actual_pos_tag = "_NOUN" if selected_pos_code == "_ADJ_SIM" else selected_pos_code

    filtered_df = df[
        (df["ngram"] == matched_ngram)
        & (df["pos_tag"] == actual_pos_tag)
        & (df["case_sensitive"] == case_sensitive)
        & (df["corpus_region"].isin(selected_regions))
        & (df["year"] >= year_range[0])
        & (df["year"] <= year_range[1])
    ].copy()

    if selected_pos_code == "_ADJ_SIM":
        filtered_df["relative_frequency"] = (filtered_df["relative_frequency"] * 0.42).round(6)
        filtered_df["raw_match_count"] = (filtered_df["raw_match_count"] * 0.42).round().astype(int)

    # Sorteer chronologisch per regio voordat we het voortschrijdend gemiddelde berekenen
    filtered_df.sort_values(by=["corpus_region", "year"], inplace=True)

    # Bereken het voortschrijdend gemiddelde (Smoothing) per geselecteerd corpus_region:
    # UX-toelichting:
    # Bij smoothing_window == 0 tonen we de ruwe relatieve frequentie (1-jaars venster).
    # Bij smoothing_window > 0 gebruiken we een rolling window van (smoothing_window + 1) jaren
    # met min_periods=1 zodat de randen van de grafiek (1800 en 2026) niet wegvallen.
    window_size = max(1, smoothing_window + 1) if smoothing_window > 0 else 1
    filtered_df["smoothed_frequency"] = filtered_df.groupby("corpus_region")[
        "relative_frequency"
    ].transform(lambda s: s.rolling(window=window_size, min_periods=1, center=True).mean())

    # --------------------------------------------------------------------------
    # VISUALISATIE: INTERACTIEVE LIJNGRAFIEK MET ALTAIR
    # --------------------------------------------------------------------------
    with st.container(border=True):
        header_col, metric_col1, metric_col2, metric_col3 = st.columns([2.2, 1.1, 1.1, 1.1])

        with header_col:
            st.subheader(f"3. Trendanalyse voor '{matched_ngram}'")
            st.caption(
                f"Periode: **{year_range[0]} – {year_range[1]}** • "
                f"Woordsoort: **{selected_pos_code}** • "
                f"Hoofdlettergevoelig: **{'Ja' if case_sensitive else 'Nee'}** • "
                f"Smoothing: **{smoothing_window} jaar**"
            )

        # UX-element: Samenvattende transparantie-KPI's naast de grafiektitel.
        # Geeft direct antwoord op de vraag: "Over hoeveel echte boeken en treffers gaat dit?"
        total_raw_matches = int(filtered_df["raw_match_count"].sum())
        peak_row = filtered_df.loc[filtered_df["relative_frequency"].idxmax()]
        with metric_col1:
            st.metric(
                label="Totaal aantal treffers (absoluut)",
                value=f"{total_raw_matches:,}".replace(",", "."),
                help="Som van `raw_match_count` binnen de geselecteerde periode en corpora.",
            )
        with metric_col2:
            st.metric(
                label="Hoogste piekjaar (ruw)",
                value=f"{int(peak_row['year'])}",
                delta=f"{peak_row['relative_frequency']:.4f}% ({peak_row['corpus_region']})",
            )
        with metric_col3:
            avg_books = int(filtered_df["total_books_in_corpus"].mean())
            st.metric(
                label="Gem. boeken/jaar in corpus",
                value=f"{avg_books:,}".replace(",", "."),
                help="Gemiddeld publicatievolume (`total_books_in_corpus`) waarmee de percentages zijn genormaliseerd.",
            )

        # Opbouw van de Altair lijngrafiek:
        # 1. Als smoothing > 0 tonen we op de achtergrond subtiel (lichtdoorzichtig) de
        #    ruwe jaarlijkse data zodat de gebruiker precies ziet hoeveel ruis er is weggefilterd!
        # 2. Op de voorgrond tonen we de dikke, gekleurde trendlijn per corpus_region.
        # 3. Verticale stippellijnen markeren jaartallen met een historical_event.
        base_encoding = alt.Chart(filtered_df).encode(
            x=alt.X(
                "year:Q",
                title="Jaartal (Year)",
                scale=alt.Scale(domain=[year_range[0], year_range[1]]),
                axis=alt.Axis(format="d", tickCount=12),
            ),
            color=alt.Color(
                "corpus_region:N",
                title="Corpus / Regio",
                scale=alt.Scale(scheme="tableau10"),
            ),
        )

        # Voorgrond-trendlijn (smoothed_frequency) met rijke tooltip voor data-transparantie
        main_line = base_encoding.mark_line(strokeWidth=2.8, interpolate="monotone").encode(
            y=alt.Y(
                "smoothed_frequency:Q",
                title="Relatieve frequentie (% van corpus)",
                axis=alt.Axis(format=".4f"),
            ),
            tooltip=[
                alt.Tooltip("year:Q", title="Jaar"),
                alt.Tooltip("ngram:N", title="Zoekterm"),
                alt.Tooltip("corpus_region:N", title="Corpus Regio"),
                alt.Tooltip("smoothed_frequency:Q", title="Frequentie (Smoothed %)", format=".5f"),
                alt.Tooltip("relative_frequency:Q", title="Frequentie (Ruw %)", format=".5f"),
                alt.Tooltip("raw_match_count:Q", title="Absolute treffers (raw_match_count)", format=","),
                alt.Tooltip("total_books_in_corpus:Q", title="Totaal boeken in corpus", format=","),
                alt.Tooltip("historical_event:N", title="Historische gebeurtenis"),
            ],
        )

        layers = [main_line]

        # Als smoothing actief is (> 0), voeg de ruwe lijn toe met lage dekking (opacity=0.22)
        # zodat het effect van de Smoothing-slider visueel direct te vergelijken is.
        if smoothing_window > 0:
            raw_background_line = base_encoding.mark_line(
                strokeWidth=1.0, opacity=0.22, strokeDash=[3, 3]
            ).encode(
                y=alt.Y("relative_frequency:Q"),
            )
            layers.insert(0, raw_background_line)

        # Voeg subtiele verticale annotatielijnen toe voor jaren met een historical_event
        events_in_view = (
            filtered_df[filtered_df["historical_event"].notna()][["year", "historical_event"]]
            .drop_duplicates(subset=["year"])
            .sort_values("year")
        )
        if not events_in_view.empty:
            event_rules = (
                alt.Chart(events_in_view)
                .mark_rule(color="#95a5a6", strokeDash=[4, 4], opacity=0.6)
                .encode(
                    x="year:Q",
                    tooltip=[
                        alt.Tooltip("year:Q", title="Annotatiejaar"),
                        alt.Tooltip("historical_event:N", title="Historische context"),
                    ],
                )
            )
            layers.append(event_rules)

        combined_chart = alt.layer(*layers).properties(height=430).interactive()
        st.altair_chart(combined_chart, use_container_width=True)

        if smoothing_window > 0:
            st.caption(
                "ℹ️ *De gestippelde lichte lijn op de achtergrond toont de onbewerkte jaarlijkse ruis; "
                "de doorgetrokken lijn toont het gladgestreken voortschrijdend gemiddelde.*"
            )

    # ==========================================================================
    # SECTIE 5: HISTORISCHE CONTEXT & DATA-TRANSPARANTIE ONDER DE GRAFIEK
    # ==========================================================================
    # UX-toelichting:
    # Een lijngrafiek laat *zien* dat er een piek is in bijv. 1914, 1945 of 2015,
    # maar verklaart niet *waarom*. Door de kolom 'historical_event' dynamisch uit
    # de geselecteerde tijdspanne te filteren en via st.info() onder de grafiek te
    # tonen, koppelen we kwantitatieve data direct aan kwalitatieve domeinkennis.
    st.subheader("4. Historische Context bij de Geselecteerde Periode (`historical_event`)")

    if events_in_view.empty:
        st.info(
            f"Geen geregistreerde historische mijlpalen gevonden in het gekozen tijdsvenster "
            f"({year_range[0]} – {year_range[1]}). Vergroot het bereik met de tijdlijn-slider om "
            f"historische annotaties te bekijken."
        )
    else:
        st.markdown(
            f"Binnen de geselecteerde periode (**{year_range[0]} – {year_range[1]}**) zijn er "
            f"**{len(events_in_view)} historische gebeurtenissen** gemarkeerd (gestippelde verticale "
            f"lijnen in de grafiek) die correleren met opvallende pieken of trendbreuken:"
        )

        # Toon de historische events overzichtelijk in kolommen met st.info()
        event_cols = st.columns(2)
        for idx, (_, event_row) in enumerate(events_in_view.iterrows()):
            ev_year = int(event_row["year"])
            ev_desc = event_row["historical_event"]

            # Zoek de bijbehorende frequentie en het absolute aantal treffers in dat specifieke jaar
            year_slice = filtered_df[filtered_df["year"] == ev_year]
            avg_rel_freq = year_slice["relative_frequency"].mean()
            sum_raw_matches = int(year_slice["raw_match_count"].sum())

            with event_cols[idx % 2]:
                st.info(
                    f"📌 **{ev_desc}**\n\n"
                    f"• Ruwe frequentie in {ev_year}: **{avg_rel_freq:.4f}%** "
                    f"(*{sum_raw_matches:,} absolute vermeldingen in geselecteerde corpora*)".replace(",", ".")
                )

    # --------------------------------------------------------------------------
    # OPTIONELE DATA-INSPECTIE & EXPORT (VOOR TRANSPARANTIE-EISEN)
    # --------------------------------------------------------------------------
    with st.expander("📂 Bekijk onderliggende datatabel & download gefilterde data (Transparantie)"):
        st.markdown(
            "In tegenstelling tot de originele Ngram Viewer kun je hier direct de exacte "
            "onderliggende aantallen (`raw_match_count` en `total_books_in_corpus`) inspecteren "
            "en exporteren."
        )
        st.dataframe(
            filtered_df[
                [
                    "year",
                    "ngram",
                    "corpus_region",
                    "pos_tag",
                    "case_sensitive",
                    "raw_match_count",
                    "total_books_in_corpus",
                    "relative_frequency",
                    "smoothed_frequency",
                    "historical_event",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="⬇️ Download huidige selectie als CSV",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ngram_{matched_ngram}_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_col2:
            clean_export_df = filtered_df.astype(object).where(pd.notna(filtered_df), None)
            st.download_button(
                label="⬇️ Download huidige selectie als JSON",
                data=clean_export_df.to_json(orient="records", indent=2).encode("utf-8"),
                file_name=f"ngram_{matched_ngram}_{year_range[0]}_{year_range[1]}.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()

