"""
================================================================================
GRAPHRAG ENGINE: KNOWLEDGE GRAPH & HYBRID RETRIEVAL-AUGMENTED GENERATION
================================================================================
Bestand : graph_rag_engine.py
Doel    : Biedt Local Search, Global Community Search, Multi-hop Pathfinding
          en interactieve graafvisualisatie bovenop de historische Ngram-kennisbasis.
================================================================================
"""

import json
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# Kleurenpalet voor entiteit-categorieën (Google Material look)
CATEGORY_COLORS: Dict[str, str] = {
    "Concept": "#1a73e8",          # Google Blauw
    "Technologie": "#e37400",      # Google Oranje
    "Historisch Event": "#d93025", # Google Rood
    "Sleutelfiguur": "#188038",    # Google Groen
    "Tijdperk / Stroming": "#9334e6", # Google Paars
}

# ==============================================================================
# 1. KENNISGRAAF INITIALISATIE & ONTOLOGIE
# ==============================================================================
def build_historical_knowledge_graph() -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Construeert de NetworkX kennisgraaf met historische entiteiten, getypeerde relaties,
    gewichten en community metadata.
    """
    G = nx.DiGraph()

    # --------------------------------------------------------------------------
    # ENTITEITEN (NODES)
    # --------------------------------------------------------------------------
    entities = [
        # Concepten
        {"id": "climate", "label": "Klimaat / Climate", "category": "Concept", "period": "1800-heden", "ngram": "climate", "desc": "Wetenschappelijk en maatschappelijk begrip van wereldwijde weersystemen en opwarming."},
        {"id": "fossil fuels", "label": "Fossiele Brandstoffen", "category": "Concept", "period": "1820-heden", "ngram": "fossil fuels", "desc": "Kolen, olie en gas als primaire energiebronnen van de moderne industriële wereld."},
        {"id": "sustainability", "label": "Duurzaamheid", "category": "Concept", "period": "1970-heden", "ngram": "sustainability", "desc": "Beleid en ecologische theorie gericht op het behoud van natuurlijke hulpbronnen."},
        {"id": "democracy", "label": "Democratie / Democracy", "category": "Concept", "period": "Klassiek-heden", "ngram": "democracy", "desc": "Politiek systeem gebaseerd op volkssoevereiniteit, gelijke rechten en vrije verkiezingen."},
        {"id": "freedom of speech", "label": "Vrijheid van Meningsuiting", "category": "Concept", "period": "1789-heden", "ngram": "freedom of speech", "desc": "Fundamenteel burgerrecht vastgelegd tijdens de Verlichting en moderne grondwetten."},
        {"id": "capitalism", "label": "Kapitalisme", "category": "Concept", "period": "1776-heden", "ngram": "capitalism", "desc": "Economisch stelsel gebaseerd op privaat bezit van productiemiddelen en marktwerking."},
        {"id": "artificial intelligence", "label": "Artificial Intelligence", "category": "Concept", "period": "1956-heden", "ngram": "artificial intelligence", "desc": "Het wetenschapsgebied gericht op het creëren van machines die menselijke cognitie nabootsen."},
        {"id": "generative ai", "label": "Generatieve AI & LLMs", "category": "Concept", "period": "2020-heden", "ngram": "artificial intelligence", "desc": "Geavanceerde neurale netwerken die autonoom taal, beelden en code synthetiseren."},
        {"id": "evolution", "label": "Evolutietheorie", "category": "Concept", "period": "1859-heden", "ngram": "evolution", "desc": "Natuurwetenschappelijke verklaring voor het ontstaan en aanpassen van biologische soorten."},
        {"id": "globalization", "label": "Globalisering", "category": "Concept", "period": "1870-heden", "ngram": "globalization", "desc": "Toenemende internationale verwevenheid van economieën, culturen en informatie."},

        # Technologieën
        {"id": "steam engine", "label": "Stoommachine", "category": "Technologie", "period": "1769-1920", "ngram": "steam engine", "desc": "Thermische machine van James Watt die de Industriële Revolutie in gang zette."},
        {"id": "telegraph", "label": "Elektrische Telegraaf", "category": "Technologie", "period": "1844-1970", "ngram": "telegraph", "desc": "Eerste technologie voor instantane langeafstandstelecommunicatie via morsecode."},
        {"id": "electricity", "label": "Elektriciteit", "category": "Technologie", "period": "1880-heden", "ngram": "electricity", "desc": "Universele energiedrager voor verlichting, fabrieken en elektronica."},
        {"id": "computer", "label": "Elektronische Computer", "category": "Technologie", "period": "1940-heden", "ngram": "computer", "desc": "Programmeerbare digitale rekenmachine voor dataverwerking en automatie."},
        {"id": "internet", "label": "Internet & World Wide Web", "category": "Technologie", "period": "1990-heden", "ngram": "internet", "desc": "Wereldwijd computernetwerk dat massacommunicatie en kennistoegang democratiseerde."},
        {"id": "printing press", "label": "Boekdrukkunst", "category": "Technologie", "period": "1450-heden", "ngram": "printing press", "desc": "Mechanische reproductie van tekst, fundament onder de verspreiding van boeken en Ngrams."},

        # Historische Gebeurtenissen
        {"id": "Industrial Revolution", "label": "Industriële Revolutie", "category": "Historisch Event", "period": "1760-1840", "ngram": "Industrial Revolution", "desc": "Overgang van handmatige productie naar machinale fabrieken en massale urbanisatie."},
        {"id": "Morse Telegraph 1844", "label": "Eerste Telegraaflijn (1844)", "category": "Historisch Event", "period": "1844", "ngram": "telegraph", "desc": "Samuel Morse verstuurt het eerste telegram: 'What hath God wrought'."},
        {"id": "American Civil War", "label": "Amerikaanse Burgeroorlog", "category": "Historisch Event", "period": "1861-1865", "ngram": "Civil War", "desc": "Strijd over eenheid en de afschaffing van slavernij in de Verenigde Staten."},
        {"id": "WW1", "label": "Eerste Wereldoorlog", "category": "Historisch Event", "period": "1914-1918", "ngram": "World War", "desc": "Mondiaal conflict dat imperia ontmantelde en industriële oorlogsvoering introduceerde."},
        {"id": "WW2", "label": "Tweede Wereldoorlog & VN", "category": "Historisch Event", "period": "1939-1945", "ngram": "World War", "desc": "Wereldoorlog tegen het fascisme, resulterend in de Universele Verklaring van de Rechten van de Mens."},
        {"id": "Dartmouth Workshop 1956", "label": "Dartmouth AI Workshop (1956)", "category": "Historisch Event", "period": "1956", "ngram": "artificial intelligence", "desc": "Academische bijeenkomst waar John McCarthy de term 'Artificial Intelligence' introduceerde."},
        {"id": "Commercial Internet 1995", "label": "Commercieel Internet (1995)", "category": "Historisch Event", "period": "1995", "ngram": "internet", "desc": "Opkomst van browsers, zoekmachines en de dotcom-economie."},
        {"id": "Paris Climate Accord 2015", "label": "Klimaatakkoord Parijs (2015)", "category": "Historisch Event", "period": "2015", "ngram": "climate", "desc": "Internationaal bindend verdrag om opwarming te beperken tot ruim onder 2 graden Celsius."},

        # Sleutelfiguren
        {"id": "James Watt", "label": "James Watt", "category": "Sleutelfiguur", "period": "1736-1819", "ngram": "James Watt", "desc": "Schotse uitvinder die de stoommachine perfectioneerde met een aparte condensator."},
        {"id": "Charles Darwin", "label": "Charles Darwin", "category": "Sleutelfiguur", "period": "1809-1882", "ngram": "Charles Darwin", "desc": "Bioloog en auteur van 'On the Origin of Species' (1859)."},
        {"id": "Ada Lovelace", "label": "Ada Lovelace", "category": "Sleutelfiguur", "period": "1815-1852", "ngram": "Ada Lovelace", "desc": "Eerste computerprogrammeur die het universele potentieel van rekenmachines voorzag."},
        {"id": "Alan Turing", "label": "Alan Turing", "category": "Sleutelfiguur", "period": "1912-1954", "ngram": "Alan Turing", "desc": "Wiskundige, grondlegger van de informatica en bedenker van de Turingtest voor AI."},
        {"id": "Adam Smith", "label": "Adam Smith", "category": "Sleutelfiguur", "period": "1723-1790", "ngram": "Adam Smith", "desc": "Filosoof en econoom, grondlegger van de klassieke markteconomie ('The Wealth of Nations')."},
        {"id": "Alexander von Humboldt", "label": "Alexander von Humboldt", "category": "Sleutelfiguur", "period": "1769-1859", "ngram": "Alexander von Humboldt", "desc": "Pionier in ecologie en biogeografie; waarschuwde als eerste voor menselijke klimaatinvloed."},

        # Tijdperken / Stromingen
        {"id": "Enlightenment", "label": "De Verlichting", "category": "Tijdperk / Stroming", "period": "1700-1800", "ngram": "Enlightenment", "desc": "Filosofische stroming waarin rede, wetenschappelijk scepticisme en burgerrechten centraal stonden."},
        {"id": "Information Age", "label": "Informatietijdperk", "category": "Tijdperk / Stroming", "period": "1970-heden", "ngram": "Information Age", "desc": "Historische periode gekenmerkt door digitale data, snelle netwerken en computerautomatisering."},
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
        ("electricity", "computer", "ENABLED", 0.90, "Elektronica en transistors vormden de basis voor digitale computers."),
        ("Ada Lovelace", "computer", "THEORIZED_FOR", 0.92, "Schreef het eerste algoritme voor de Mechanische Analytische Machine van Babbage."),
        ("Alan Turing", "computer", "MATHEMATIZED", 0.96, "Turing-machine formaliseerde het universele concept van berekenbaarheid."),
        ("Alan Turing", "artificial intelligence", "PIONEERED", 0.94, "Publiceerde 'Computing Machinery and Intelligence' (1950) met de Turingtest."),
        ("Dartmouth Workshop 1956", "artificial intelligence", "CHRISTENED", 0.99, "Geboorte van AI als formeel academisch onderzoeksveld."),
        ("computer", "internet", "INTERCONNECTED_BY", 0.93, "Netwerken van computers culmineerden in het wereldwijde internet."),
        ("Commercial Internet 1995", "internet", "EXPANDED", 0.95, "Het publieke World Wide Web verving gesloten academische netwerken."),
        ("internet", "generative ai", "SUPPLIED_DATA", 0.92, "Het wereldwijde web leverde de enorme tekstcorpora om LLMs te trainen."),
        ("artificial intelligence", "generative ai", "EVOLVED_INTO", 0.96, "Diepe transformermodellen luidden de generatieve AI-revolutie in."),
        ("computer", "Information Age", "DEFINED", 0.91, "De computerisering verschoof de maatschappij naar een kenniseconomie."),

        # Democratie, Verlichting & Maatschappij
        ("Enlightenment", "democracy", "INTELLECTUALLY_FOUNDED", 0.93, "Ideeën van Locke, Rousseau en Montesquieu herdefinieerden het staatsbestel."),
        ("Enlightenment", "freedom of speech", "CHAMPIONED", 0.95, "Voltaire en verlichte filosofen bepleitten ongehinderde meningsuiting."),
        ("printing press", "Enlightenment", "PROPAGATED", 0.88, "Gedrukte pamfletten en boeken maakten de Verlichtingsfilosofie breed toegankelijk."),
        ("Adam Smith", "capitalism", "FORMULATED", 0.94, "Beschreef de principes van vrije marktwerking en arbeidsdeling."),
        ("democracy", "freedom of speech", "REQUIRES", 0.97, "Vrije meningsuiting is een absolute voorwaarde voor democratische verkiezingen."),
        ("Industrial Revolution", "capitalism", "REINFORCED", 0.87, "Fabriekskapitalisme vereiste grootschalig investeringskapitaal en bankwezen."),
        ("American Civil War", "democracy", "TESTED_AND_PRESERVED", 0.89, "Lincoln definieerde de oorlog als proef of een democratie kan standhouden."),
        ("WW2", "democracy", "DEFENDED_AND_EXPANDED", 0.93, "Overwinning op autoritaire regimes leidde tot VN-mensenrechtenhandvest."),

        # Wetenschap & Evolutie
        ("Alexander von Humboldt", "Charles Darwin", "INSPIRED", 0.91, "Humboldts reisverslagen vormden Darwins motivatie voor de Beagle-expeditie."),
        ("Charles Darwin", "evolution", "SYSTEMATIZED", 0.98, "Formuleerde natuurlijke selectie als sturend mechanisme van biologische variatie."),
        ("Enlightenment", "evolution", "PREPARED_EMPIRICISM", 0.82, "Brak met dogmatische verklaringen ten gunste van empirische observatie."),
    ]

    for src, dst, rel_type, weight, expl in relations:
        G.add_edge(src, dst, relation_type=rel_type, weight=weight, explanation=expl)

    # --------------------------------------------------------------------------
    # COMMUNITIES (THEMATISCHE CLUSTERS MET HIERARCHISCHE SAMENVATTINGEN)
    # --------------------------------------------------------------------------
    communities = {
        "cluster_klimaat_energie": {
            "title": "Klimaat, Energie & Industriële Transformatie",
            "nodes": ["Industrial Revolution", "steam engine", "fossil fuels", "climate", "sustainability", "Paris Climate Accord 2015", "James Watt", "Alexander von Humboldt"],
            "summary": (
                "Deze community beschrijft de wisselwerking tussen mechanisatie (stoomkracht), het exponentieel "
                "stijgende verbruik van fossiele brandstoffen en de uiteindelijke wetenschappelijke en diplomatieke "
                "erkenning van de klimaatverandering vanaf de 19e eeuw tot aan het Parijs-akkoord in 2015."
            ),
            "dominant_era": "1760 – Heden",
            "key_drivers": ["Mechanisatie", "Kolen en Olie", "Ecologische Wetenschap", "Mondiale Verdragen"],
        },
        "cluster_informatie_ai": {
            "title": "Informatie, Communicatie & AI Revolutie",
            "nodes": ["telegraph", "Morse Telegraph 1844", "electricity", "computer", "internet", "Commercial Internet 1995", "artificial intelligence", "generative ai", "Dartmouth Workshop 1956", "Ada Lovelace", "Alan Turing", "Information Age"],
            "summary": (
                "Deze community toont de evolutie van snelle communicatie en rekenkracht: beginnend bij de elektrische "
                "telegraaf in 1844, via formele computertheorieën van Turing en Lovelace, de geboorte van AI in 1956 "
                "te Dartmouth, tot aan het openbare internet en de huidige generatieve taalmodellen."
            ),
            "dominant_era": "1844 – Heden",
            "key_drivers": ["Telecommunicatie", "Binaire Informatica", "Neurale Netwerken", "Massale Webdata"],
        },
        "cluster_democratie_rechten": {
            "title": "Democratie, Rechten & Staatsinrichting",
            "nodes": ["Enlightenment", "democracy", "freedom of speech", "capitalism", "Adam Smith", "American Civil War", "WW2", "printing press"],
            "summary": (
                "Deze gemeenschap weerspiegelt de opkomst van democratische waarden, persvrijheid en markteconomie "
                "vanaf de Verlichting. Het documenteert hoe burgerrechten werden bevochten en vastgelegd door "
                "cruciale historische crises zoals de Amerikaanse Burgeroorlog en de Tweede Wereldoorlog."
            ),
            "dominant_era": "1700 – Heden",
            "key_drivers": ["Verlichtingsfilosofie", "Vrije Meningsuiting", "Klassiek Liberalisme", "VN Rechten"],
        },
        "cluster_wetenschap_evolutie": {
            "title": "Wetenschappelijke Revolutie & Biologie",
            "nodes": ["evolution", "Charles Darwin", "Alexander von Humboldt", "Enlightenment"],
            "summary": (
                "De verschuiving naar zuiver empirische natuurwetenschap. Begint bij de holistische ecologische "
                "waarnemingen van Humboldt en culmineert in Darwins evolutieleer via natuurlijke selectie (1859), "
                "die het menselijke zelfbeeld voorgoed veranderde."
            ),
            "dominant_era": "1750 – 1900",
            "key_drivers": ["Empirisch Veldonderzoek", "Natuurlijke Selectie", "Biogeografie"],
        },
    }

    return G, communities


# ==============================================================================
# 2. BRONFRAGMENTEN & HISTORISCHE BEWIJSSTUKKEN (CORPUS CHUNKS)
# ==============================================================================
CORPUS_EVIDENCE_CHUNKS: List[Dict[str, Any]] = [
    {
        "id": "chunk_01",
        "title": "James Watt en de Condensor (1769)",
        "source": "The Industrial and Commercial Revolutions in Great Britain during the Nineteenth Century (1921)",
        "year": 1769,
        "entities": ["James Watt", "steam engine", "Industrial Revolution"],
        "text": (
            "Watt's invention of the separate condenser transformed the steam engine from a clumsy mine-pumping machine "
            "into a universal prime mover, providing steady rotative power to the textile mills of Lancashire and "
            "foundries across Great Britain."
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
            "The beds of the brooks are converted into torrents whenever heavy rain falls, and mankind alters the "
            "very atmosphere and local climate."
        ),
    },
    {
        "id": "chunk_03",
        "title": "De Eerste Telegraaflijn (1844)",
        "source": "Harper's New Monthly Magazine: The Lightning Telegraph (1850)",
        "year": 1844,
        "entities": ["telegraph", "Morse Telegraph 1844", "electricity", "globalization"],
        "text": (
            "The message transmitted along the wire between Washington and Baltimore annihilated time and space. "
            "For the first time in human history, thought traveled faster than any physical vehicle could carry it, "
            "binding distant cities into a single nervous system."
        ),
    },
    {
        "id": "chunk_04",
        "title": "On the Origin of Species (1859)",
        "source": "On the Origin of Species by Means of Natural Selection (Charles Darwin, 1859)",
        "year": 1859,
        "entities": ["Charles Darwin", "evolution", "Enlightenment"],
        "text": (
            "As many more individuals of each species are born than can possibly survive, there is a frequently recurring "
            "struggle for existence. It follows that any being, if it vary however slightly in any manner profitable to itself, "
            "will have a better chance of surviving, and thus be naturally selected."
        ),
    },
    {
        "id": "chunk_05",
        "title": "Dartmouth AI Proposal (1955-1956)",
        "source": "A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence (1955)",
        "year": 1956,
        "entities": ["Dartmouth Workshop 1956", "artificial intelligence", "computer", "Alan Turing"],
        "text": (
            "We propose that a 2 month, 10 man study of artificial intelligence be carried out during the summer of 1956. "
            "The study is to proceed on the basis of the conjecture that every aspect of learning or any other feature of "
            "intelligence can in principle be so precisely described that a machine can be made to simulate it."
        ),
    },
    {
        "id": "chunk_06",
        "title": "Klimaatakkoord van Parijs (2015)",
        "source": "United Nations Framework Convention on Climate Change (UNFCCC, 2015)",
        "year": 2015,
        "entities": ["climate", "sustainability", "Paris Climate Accord 2015", "fossil fuels"],
        "text": (
            "Holding the increase in the global average temperature to well below 2°C above pre-industrial levels and pursuing "
            "efforts to limit the temperature increase to 1.5°C, recognizing that this would significantly reduce the risks "
            "and impacts of climate change."
        ),
    },
    {
        "id": "chunk_07",
        "title": "De Verlichting en Vrijheid van Meningsuiting",
        "source": "Treatise on Tolerance (Voltaire, 1763) & Two Treatises of Government (Locke, 1689)",
        "year": 1789,
        "entities": ["Enlightenment", "democracy", "freedom of speech", "printing press"],
        "text": (
            "The free communication of thoughts and of opinions is one of the most precious rights of man: any citizen thus "
            "may speak, write, print freely, except to answer for the abuse of this liberty, in cases determined by the law."
        ),
    },
]


# ==============================================================================
# 3. GRAPHRAG QUERY ENGINE
# ==============================================================================
class GraphRAGEngine:
    """GraphRAG Engine voor semantisch zoeken, ego-subgrafen en community summaries."""

    def __init__(self) -> None:
        self.graph, self.communities = build_historical_knowledge_graph()
        self.chunks = CORPUS_EVIDENCE_CHUNKS

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Geeft alle entiteiten met hun eigenschappen terug."""
        nodes_data = []
        for n, attrs in self.graph.nodes(data=True):
            nodes_data.append({"id": n, **attrs})
        return sorted(nodes_data, key=lambda x: x["label"])

    def find_entity_by_query(self, query: str) -> Optional[str]:
        """Zoekt de best passende node ID op basis van een tekstuele zoekterm."""
        q = query.lower().strip()
        if not q:
            return None

        # Exact match op ID
        if q in self.graph:
            return q

        # Match op label of ngram
        for n, attrs in self.graph.nodes(data=True):
            if q == n.lower() or q == attrs.get("label", "").lower() or q == attrs.get("ngram", "").lower():
                return n

        # Deelmatch
        for n, attrs in self.graph.nodes(data=True):
            if q in n.lower() or q in attrs.get("label", "").lower():
                return n

        return None

    # --------------------------------------------------------------------------
    # MODUS A: LOCAL SEARCH (Ego-graaf, directe relaties & bewijzen)
    # --------------------------------------------------------------------------
    def local_search(self, target_entity: str, max_hops: int = 1) -> Dict[str, Any]:
        """
        Voert een Local Search uit rondom een specifieke entiteit.
        Haalt directe buren, getypeerde relaties, graafstatistieken en tekstuele bewijzen op.
        """
        node_id = self.find_entity_by_query(target_entity)
        if not node_id or node_id not in self.graph:
            return {
                "found": False,
                "query": target_entity,
                "message": f"Geen directe entiteit gevonden voor '{target_entity}'. Probeer bijv. 'climate', 'democracy', 'telegraph', of 'artificial intelligence'.",
            }

        node_attrs = self.graph.nodes[node_id]

        # Ego-subgraaf extraheren
        undirected = self.graph.to_undirected()
        ego_sub = nx.ego_graph(undirected, node_id, radius=max_hops)

        # Directe uitgaande en inkomende relaties
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

        # Relevante tekstchunks verzamelen
        related_chunks = []
        all_sub_nodes = set(ego_sub.nodes())
        for chk in self.chunks:
            overlap = set(chk["entities"]).intersection(all_sub_nodes)
            if node_id in chk["entities"] or overlap:
                relevance = 2.0 if node_id in chk["entities"] else 1.0 + (len(overlap) * 0.2)
                related_chunks.append({**chk, "relevance": relevance, "overlap_entities": list(overlap)})

        related_chunks.sort(key=lambda x: x["relevance"], reverse=True)

        # Community waarin de entiteit zich bevindt
        assigned_community = None
        for c_id, c_data in self.communities.items():
            if node_id in c_data["nodes"]:
                assigned_community = c_data
                break

        # Graaf-centraliteit binnen netwerk
        deg_centrality = nx.degree_centrality(self.graph).get(node_id, 0.0)
        pagerank = nx.pagerank(self.graph).get(node_id, 0.0)

        # Gesynthetiseerd RAG-antwoord formuleren
        synthesis = self._synthesize_local_answer(node_attrs, out_edges, in_edges, assigned_community, related_chunks)

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
            "metrics": {
                "degree": self.graph.degree(node_id),
                "degree_centrality": round(deg_centrality, 3),
                "pagerank": round(pagerank, 4),
                "neighbors_count": len(ego_sub.nodes()) - 1,
            },
            "outgoing_relations": out_edges,
            "incoming_relations": in_edges,
            "evidence_chunks": related_chunks[:4],
            "synthesis": synthesis,
            "subgraph_nodes": list(ego_sub.nodes()),
        }

    # --------------------------------------------------------------------------
    # MODUS B: GLOBAL SEARCH (Synthese over Community Summaries)
    # --------------------------------------------------------------------------
    def global_search(self, user_query: str) -> Dict[str, Any]:
        """
        Voert een Global Search uit door overkoepelende community summaries te raadplegen.
        Biedt holistisch inzicht in bredere historische thema's.
        """
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

        # Synthese genereren
        if top_community and top_community["match_score"] > 0:
            summary_text = (
                f"### Macro-Synthese: {top_community['title']}\n\n"
                f"**Primaire historische periode:** {top_community['dominant_era']}  \n"
                f"**Sleuteldrijvers:** {', '.join(top_community['key_drivers'])}\n\n"
                f"**Thematische Analyse:**  \n{top_community['summary']}\n\n"
                f"**Verbonden Kernconcepten in deze Community:**  \n"
                + " • ".join([f"`{n}`" for n in top_community["nodes"]])
            )
        else:
            summary_text = (
                "### Overkoepelende Kennisgraaf Synthese\n\n"
                "De historische dataset omspant vier fundamentele transformatieve golven tussen 1750 en heden:\n"
                "1. **Industrie & Energie:** Van de stoommachine van Watt naar massaal fossiel brandstofverbruik en modern milieubewustzijn.\n"
                "2. **Communicatie & AI:** Van de telegraaf in 1844 tot de Turingtest en de huidige generatieve AI-revolutie.\n"
                "3. **Democratie & Burgerrechten:** Verlichtingstheorieën verankerd in de Verklaring van de Rechten van de Mens.\n"
                "4. **Natuurwetenschappen:** De opkomst van empirische observatie en Darwins evolutieleer."
            )

        return {
            "query": user_query,
            "top_community": top_community,
            "all_communities": list(self.communities.values()),
            "synthesis": summary_text,
        }

    # --------------------------------------------------------------------------
    # MODUS C: MULTI-HOP PATHFINDING (Concept Bridging)
    # --------------------------------------------------------------------------
    def find_relational_path(self, source_term: str, target_term: str) -> Dict[str, Any]:
        """
        Vindt de kortste en semantisch rijkste routes tussen twee entiteiten.
        Toont de causale keten die Concept A historisch verbindt met Concept B.
        """
        src_id = self.find_entity_by_query(source_term)
        dst_id = self.find_entity_by_query(target_term)

        if not src_id or not dst_id:
            return {
                "success": False,
                "message": f"Kon één of beide entiteiten niet vinden: '{source_term}' of '{target_term}'.",
            }

        undirected = self.graph.to_undirected()

        try:
            shortest_path = nx.shortest_path(undirected, source=src_id, target=dst_id)
        except nx.NetworkXNoPath:
            return {
                "success": False,
                "message": f"Er bestaat geen direct of indirect pad tussen '{src_id}' en '{dst_id}' in de kennisgraaf.",
            }

        # Stappen in het pad analyseren
        steps = []
        for i in range(len(shortest_path) - 1):
            u, v = shortest_path[i], shortest_path[i + 1]
            # Zoek edge gegevens in originele gerichte graaf (in beide richtingen)
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
                    "step_number": i + 1,
                    "from_node": u,
                    "from_label": u_lbl,
                    "to_node": v,
                    "to_label": v_lbl,
                    "direction": direction,
                    "relation_type": edata.get("relation_type", "RELATED_TO"),
                    "explanation": edata.get("explanation", ""),
                }
            )

        # Tekstuele synthese van de concept-bridge
        synthesis = self._synthesize_path_explanation(shortest_path, steps)

        return {
            "success": True,
            "source_id": src_id,
            "target_id": dst_id,
            "path_nodes": shortest_path,
            "path_length": len(shortest_path) - 1,
            "steps": steps,
            "synthesis": synthesis,
        }

    # --------------------------------------------------------------------------
    # DATA VOOR INTERACTIEVE VIS.JS FORCE-DIRECTED GRAAF
    # --------------------------------------------------------------------------
    def get_visualization_json(self, highlighted_nodes: Optional[Set[str]] = None) -> str:
        """
        Exporteert de kennisgraaf als JSON voor de Vis.js interactive canvas visualisatie.
        """
        vis_nodes = []
        for n, data in self.graph.nodes(data=True):
            is_highlighted = highlighted_nodes is not None and n in highlighted_nodes
            base_color = data.get("color", "#1a73e8")

            vis_nodes.append(
                {
                    "id": n,
                    "label": data.get("label", n),
                    "title": f"<b>{data.get('label', n)}</b><br>Type: {data.get('category')}<br>Periode: {data.get('period')}<br>{data.get('desc')}",
                    "color": {
                        "background": "#ffffff" if not is_highlighted and highlighted_nodes else base_color,
                        "border": base_color,
                        "highlight": {"background": base_color, "border": "#202124"},
                    },
                    "borderWidth": 3 if is_highlighted else 1.5,
                    "size": 26 if is_highlighted else 18,
                    "font": {"size": 13, "color": "#202124", "face": "Google Sans, Roboto, sans-serif"},
                    "shape": "dot",
                    "category": data.get("category", "Concept"),
                }
            )

        vis_edges = []
        for u, v, data in self.graph.edges(data=True):
            is_edge_highlighted = (
                highlighted_nodes is not None and u in highlighted_nodes and v in highlighted_nodes
            )
            rel_type = data.get("relation_type", "")
            vis_edges.append(
                {
                    "from": u,
                    "to": v,
                    "label": rel_type.replace("_", " ").lower(),
                    "title": data.get("explanation", ""),
                    "arrows": "to",
                    "color": {"color": "#1a73e8" if is_edge_highlighted else "#dadce0", "highlight": "#1a73e8"},
                    "width": 3.0 if is_edge_highlighted else 1.2,
                    "font": {"size": 10, "color": "#5f6368", "align": "middle"},
                }
            )

        return json.dumps({"nodes": vis_nodes, "edges": vis_edges})

    # --------------------------------------------------------------------------
    # INTERNE HELPERS VOOR RAG-SYNTHESE
    # --------------------------------------------------------------------------
    def _synthesize_local_answer(
        self,
        node_attrs: Dict[str, Any],
        out_edges: List[Dict[str, Any]],
        in_edges: List[Dict[str, Any]],
        community: Optional[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> str:
        name = node_attrs.get("label", "")
        desc = node_attrs.get("desc", "")
        category = node_attrs.get("category", "")
        period = node_attrs.get("period", "")

        out_summary = ", ".join([f"`{e['target_label']}` ({e['relation_type']})" for e in out_edges[:3]])
        in_summary = ", ".join([f"`{e['source_label']}` ({e['relation_type']})" for e in in_edges[:3]])

        comm_str = f"Behoort tot de community **{community['title']}**." if community else ""

        chunk_quote = ""
        if chunks:
            top_chunk = chunks[0]
            chunk_quote = f"\n> *\"{top_chunk['text']}\"*\n> — **{top_chunk['source']} ({top_chunk['year']})**\n"

        return (
            f"### GraphRAG Analyse: {name}\n\n"
            f"**Entiteittype:** `{category}` | **Historische Actieradius:** `{period}`  \n"
            f"{comm_str}\n\n"
            f"**Omschrijving:**  \n{desc}\n\n"
            f"**Kennisgraaf Relaties:**  \n"
            f"* **Beïnvloedt direct:** {out_summary or 'Geen geregistreerde uitgaande paden.'}\n"
            f"* **Werd gedreven door:** {in_summary or 'Geen geregistreerde inkomende paden.'}\n\n"
            f"**Historisch Bewijsmateriaal (Corpus Chunk):**  {chunk_quote}"
        )

    def _synthesize_path_explanation(self, path: List[str], steps: List[Dict[str, Any]]) -> str:
        labels = [self.graph.nodes[n].get("label", n) for n in path]
        chain_visual = " ➔ ".join([f"**{lbl}**" for lbl in labels])

        details = []
        for s in steps:
            details.append(
                f"{s['step_number']}. **{s['from_label']}** {s['direction']} **{s['to_label']}**  \n"
                f"   *Relatietype:* `{s['relation_type']}` — *Context:* {s['explanation']}"
            )

        return (
            f"### Historische Concept-Bridge: {labels[0]} ➔ {labels[-1]}\n\n"
            f"**Verbindingsketen ({len(steps)} stappen):**  \n{chain_visual}\n\n"
            f"**Gedetailleerde Verantwoording:**  \n" + "\n\n".join(details)
        )


def generate_visjs_html(graph_json: str, height_px: int = 520) -> str:
    """
    Genereert een complete, zelfstandige HTML/JS-pagina met een interactieve Vis.js
    force-directed network visualisatie, inclusief physics, zoom/pan en legenda.
    """
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
                font-family: 'Google Sans', Roboto, -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                background-color: #ffffff;
                color: #202124;
                overflow: hidden;
            }}
            #network-container {{
                width: 100%;
                height: {height_px}px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background: #fafafa;
                position: relative;
            }}
            .graph-toolbar {{
                position: absolute;
                top: 10px;
                left: 12px;
                z-index: 10;
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #dadce0;
                border-radius: 6px;
                padding: 6px 10px;
                display: flex;
                gap: 12px;
                font-size: 11px;
                box-shadow: 0 1px 3px rgba(60,64,67,.15);
                align-items: center;
                user-select: none;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 4px;
            }}
            .legend-dot {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                display: inline-block;
            }}
            .node-info-card {{
                position: absolute;
                bottom: 12px;
                left: 12px;
                max-width: 380px;
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid #dadce0;
                border-radius: 6px;
                padding: 10px 14px;
                font-size: 12px;
                box-shadow: 0 2px 6px rgba(60,64,67,.2);
                z-index: 10;
                display: none;
            }}
            .node-info-title {{
                font-weight: 600;
                color: #1a73e8;
                margin-bottom: 4px;
                font-size: 13px;
            }}
        </style>
    </head>
    <body>
        <div id="network-container">
            <div class="graph-toolbar">
                <span style="font-weight: 500; color: #5f6368;">Legenda:</span>
                <span class="legend-item"><span class="legend-dot" style="background:#1a73e8;"></span> Concept</span>
                <span class="legend-item"><span class="legend-dot" style="background:#e37400;"></span> Technologie</span>
                <span class="legend-item"><span class="legend-dot" style="background:#d93025;"></span> Historisch Event</span>
                <span class="legend-item"><span class="legend-dot" style="background:#188038;"></span> Figuur</span>
                <span class="legend-item"><span class="legend-dot" style="background:#9334e6;"></span> Tijdperk</span>
            </div>
            <div id="node-info-card" class="node-info-card">
                <div id="card-title" class="node-info-title"></div>
                <div id="card-desc" style="color: #3c4043; line-height: 1.4;"></div>
            </div>
        </div>

        <script type="text/javascript">
            const graphData = {graph_json};
            const container = document.getElementById('network-container');
            const infoCard = document.getElementById('node-info-card');
            const cardTitle = document.getElementById('card-title');
            const cardDesc = document.getElementById('card-desc');

            const data = {{
                nodes: new vis.DataSet(graphData.nodes),
                edges: new vis.DataSet(graphData.edges)
            }};

            const options = {{
                nodes: {{
                    shape: 'dot',
                    scaling: {{ min: 14, max: 30 }},
                    shadow: {{ enabled: true, color: 'rgba(0,0,0,0.06)', size: 5, x: 2, y: 2 }}
                }},
                edges: {{
                    arrows: {{ to: {{ enabled: true, scaleFactor: 0.55 }} }},
                    smooth: {{ type: 'continuous', roundness: 0.2 }},
                    selectionWidth: 2.5
                }},
                physics: {{
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {{
                        gravitationalConstant: -55,
                        centralGravity: 0.015,
                        springLength: 100,
                        springConstant: 0.08,
                        damping: 0.4
                    }},
                    maxVelocity: 50,
                    minVelocity: 0.1,
                    stabilization: {{ iterations: 120 }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 150,
                    zoomView: true,
                    dragView: true,
                    navigationButtons: false
                }}
            }};

            const network = new vis.Network(container, data, options);

            network.on("click", function(params) {{
                if (params.nodes.length > 0) {{
                    const nodeId = params.nodes[0];
                    const nodeObj = graphData.nodes.find(n => n.id === nodeId);
                    if (nodeObj) {{
                        cardTitle.innerText = nodeObj.label + ' (' + (nodeObj.category || 'Entiteit') + ')';
                        cardDesc.innerHTML = nodeObj.title || '';
                        infoCard.style.display = 'block';
                    }}
                }} else {{
                    infoCard.style.display = 'none';
                }}
            }});
        </script>
    </body>
    </html>
    """

