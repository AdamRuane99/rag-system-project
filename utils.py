"""
utils.py — Helper functions for the RAG dashboard.

Handles:
  - Text chunking
  - Sample data generation
  - Document loading from uploaded files
"""

import re
import json
import logging
import urllib.parse
import urllib.request
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split a long string into overlapping word-level chunks.

    Args:
        text:       The full text to split.
        chunk_size: Target number of words per chunk.
        overlap:    Number of words shared between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap   # slide window with overlap

    return chunks


def clean_text(text: str) -> str:
    """Collapse whitespace and strip leading/trailing spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sample data generation — no external dataset required
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    {
        "title": "Q3 2024 Earnings Report — Apex Technologies",
        "content": (
            "Apex Technologies reported record revenues of $4.2 billion in Q3 2024, "
            "driven by a 38% year-over-year increase in cloud services. The company's "
            "AI division contributed $870 million, up from $510 million in Q3 2023. "
            "Operating margins improved to 22% as infrastructure costs declined following "
            "the migration to next-generation data centers. CEO Maria Chen credited the "
            "strong results to accelerated enterprise adoption of ApexAI Suite, which now "
            "serves over 12,000 business customers globally. The board approved a $500 million "
            "share buyback programme and raised the quarterly dividend by 8 cents to $0.42 per share. "
            "Guidance for Q4 2024 was set at $4.4–4.6 billion in revenue, reflecting continued "
            "momentum in cloud and AI, partially offset by seasonal slowdowns in hardware sales."
        ),
    },
    {
        "title": "Climate Change and Renewable Energy Transition",
        "content": (
            "Global renewable energy capacity surpassed 3,000 GW in 2024, with solar photovoltaics "
            "accounting for the largest share of new installations at 45%. Wind energy added 117 GW "
            "of capacity, while battery storage deployments more than doubled compared to 2023. "
            "The International Energy Agency noted that clean energy investment reached $1.8 trillion, "
            "outpacing fossil-fuel investment for the fourth consecutive year. Emerging markets in "
            "Southeast Asia and Africa drove a significant portion of growth, benefiting from rapidly "
            "falling solar panel costs — down 89% over the past decade. Policy frameworks such as the "
            "EU Green Deal and the US Inflation Reduction Act continued to accelerate deployment. "
            "Analysts warn, however, that grid infrastructure upgrades remain the key bottleneck "
            "to absorbing intermittent renewable generation at scale."
        ),
    },
    {
        "title": "Advances in Large Language Models — 2024 Review",
        "content": (
            "The past year saw remarkable progress in large language models (LLMs). Open-source models "
            "narrowed the capability gap with proprietary systems; the Llama and Mistral families "
            "demonstrated near-GPT-4-level performance on many benchmarks at a fraction of the inference "
            "cost. Multimodal models capable of processing text, images, audio, and video became "
            "mainstream, enabling applications from medical imaging analysis to real-time translation. "
            "Retrieval-Augmented Generation (RAG) emerged as the dominant pattern for enterprise AI, "
            "allowing organisations to ground model responses in proprietary knowledge bases without "
            "fine-tuning. Concerns around hallucination, bias, and data privacy prompted regulators in "
            "the EU and US to draft AI governance frameworks. The open-weights movement gained momentum "
            "with major academic institutions releasing state-of-the-art models freely under permissive licences."
        ),
    },
    {
        "title": "Global Supply Chain Disruptions — Semiconductor Industry",
        "content": (
            "The semiconductor industry continued to grapple with supply chain volatility in 2024. "
            "Geopolitical tensions between the US and China resulted in new export controls on advanced "
            "chips, prompting manufacturers to diversify sourcing. TSMC's new Arizona fab reached pilot "
            "production, while Samsung and Intel accelerated investment in European facilities under "
            "the EU Chips Act. Automotive-grade chip shortages eased as demand moderated and new capacity "
            "came online, but AI accelerators — particularly high-bandwidth memory — remained constrained. "
            "SK Hynix and Micron ramped HBM3e production to meet surging demand from hyperscale AI "
            "training clusters. Industry analysts projected global semiconductor revenues would reach "
            "$628 billion by end-2024, recovering from the 2023 downturn. Long-term, the push toward "
            "3 nm and 2 nm process nodes is expected to sustain performance gains through 2028."
        ),
    },
    {
        "title": "Health Innovations: GLP-1 Drugs and Obesity Treatment",
        "content": (
            "GLP-1 receptor agonists such as semaglutide (Ozempic / Wegovy) and tirzepatide (Mounjaro) "
            "reshaped the pharmaceutical landscape in 2024. Clinical trials demonstrated weight reductions "
            "of 15–22% in obese patients, with additional cardioprotective benefits reducing major "
            "cardiovascular events by up to 20%. Global demand far outpaced manufacturing capacity, "
            "leading to widespread shortages and affordability concerns. Novo Nordisk and Eli Lilly "
            "announced combined capital investment of over $25 billion to expand production. Researchers "
            "are investigating oral formulations to widen access; early Phase 2 data look promising. "
            "Beyond obesity, GLP-1 trials are exploring applications in Alzheimer's, addiction, and "
            "non-alcoholic fatty liver disease. Healthcare economists estimate that widespread GLP-1 "
            "adoption could reduce obesity-related healthcare costs by $100 billion annually in the US alone."
        ),
    },
    {
        "title": "Electric Vehicle Market Update — 2024",
        "content": (
            "Global electric vehicle sales reached 17.5 million units in 2024, representing 21% of all "
            "new car sales — up from 14% in 2023. China remained the largest market with 9.5 million "
            "EVs sold, followed by Europe at 3.8 million and the US at 2.1 million. Battery costs fell "
            "below $100/kWh at pack level for the first time, making some EV models price-competitive "
            "with internal combustion counterparts without subsidies. BYD overtook Tesla as the world's "
            "largest EV manufacturer by volume, selling 3.6 million vehicles. Charging infrastructure "
            "expanded rapidly: the US surpassed 200,000 public charging ports, though rural coverage "
            "gaps persist. Range anxiety continues to decline as mid-range EVs now routinely offer "
            "350–400 miles per charge. Analysts forecast EVs will account for 40% of global auto sales "
            "by 2030, contingent on continued battery innovation and grid decarbonisation."
        ),
    },
    {
        "title": "Space Exploration Milestones — Artemis and Beyond",
        "content": (
            "2024 was a landmark year for space exploration. NASA's Artemis programme progressed with "
            "crewed lunar orbital missions, laying the groundwork for a Moon landing targeted for 2026. "
            "SpaceX's Starship completed its fifth integrated flight test, successfully catching the "
            "booster with the 'Mechazilla' arms — a breakthrough for full reusability. The European "
            "Space Agency's Jupiter Icy Moons Explorer (JUICE) performed its first flyby of the Moon "
            "en route to the Jovian system. India's ISRO celebrated the continued success of Chandrayaan-3 "
            "data analysis, revealing new insights about lunar south pole water ice. Commercial space "
            "activity accelerated, with more than 2,600 satellites launched in 2024 alone, primarily "
            "for broadband constellations. Space tourism saw six crewed private missions, and asteroid "
            "mining startup AstroForge conducted the first prospecting flyby of a metallic near-Earth asteroid."
        ),
    },
    {
        "title": "Cybersecurity Threat Landscape — 2024 Annual Report",
        "content": (
            "Ransomware attacks increased by 45% in 2024, with critical infrastructure — healthcare, "
            "energy, and water utilities — the primary targets. AI-generated phishing emails became "
            "indistinguishable from authentic communications, driving a 67% increase in successful "
            "spear-phishing campaigns. State-sponsored threat actors, particularly APT groups attributed "
            "to North Korea and Russia, targeted financial institutions and defence contractors. Supply "
            "chain attacks via compromised software dependencies accounted for 18% of all breaches. "
            "Zero-day vulnerabilities fetched record prices on exploit markets — some exceeding $10 million "
            "for iOS and Windows flaws. On the defensive side, the adoption of zero-trust architectures "
            "and AI-powered security operations centres (SOCs) improved mean time to detect by 35%. "
            "Governments globally pushed mandatory incident reporting windows down to 72 hours. "
            "Cyber insurance premiums rose 30% on average as underwriters tightened exclusion clauses."
        ),
    },
    {
        "title": "Global Economy — IMF World Economic Outlook Summary",
        "content": (
            "The IMF's October 2024 World Economic Outlook projected global growth of 3.2% for 2024, "
            "unchanged from 2023. Advanced economies grew at 1.8%, led by the United States at 2.8% — "
            "surprising to the upside amid resilient consumer spending. The Euro Area expanded modestly "
            "at 0.9%, held back by weak German industrial output. Emerging market and developing economies "
            "grew at 4.5%, with India leading at 7.0% and China at 4.8%. Global inflation declined to "
            "5.8% from its 2022 peak of 9.4%, allowing major central banks to begin easing cycles. "
            "The US Federal Reserve cut rates by 25 bps in September and November. Downside risks include "
            "geopolitical fragmentation, trade policy uncertainty, and elevated public debt levels across "
            "both advanced and emerging economies. The IMF called for fiscal consolidation to rebuild "
            "budgetary space ahead of potential future shocks."
        ),
    },
    {
        "title": "Quantum Computing: Progress and Challenges",
        "content": (
            "Quantum computing made significant strides in 2024. Google's Willow chip demonstrated "
            "quantum error correction below the threshold required for fault-tolerant computation — a "
            "long-sought milestone. IBM's 1,000-qubit Condor processor ran benchmark circuits with "
            "improved coherence times. Microsoft advanced its topological qubit research, claiming "
            "a path to more stable logical qubits. Despite hardware progress, practical quantum advantage "
            "over classical supercomputers remains limited to narrow problem domains such as quantum "
            "chemistry simulation and optimisation. The quantum software ecosystem matured, with Qiskit "
            "and PennyLane attracting large developer communities. National security implications drove "
            "substantial government investment: the US allocated $3 billion through the National Quantum "
            "Initiative, while China reportedly invested comparable sums. Post-quantum cryptography "
            "standards published by NIST in 2024 are already being integrated into TLS and VPN protocols "
            "to prepare for the eventual cryptographically-relevant quantum computer."
        ),
    },
]


def get_sample_documents() -> List[dict]:
    """Return the built-in sample documents (list of dicts with 'title' and 'content')."""
    return SAMPLE_DOCUMENTS


# ---------------------------------------------------------------------------
# Wikipedia article fetching — no API key required
# ---------------------------------------------------------------------------

def fetch_wikipedia_articles(topics: List[str]) -> List[dict]:
    """
    Fetch article text from the Wikipedia REST API for each topic.

    Uses the /page/summary endpoint (no key, free, CC-BY-SA licensed).
    Silently skips articles that cannot be fetched or are too short.

    Args:
        topics: List of Wikipedia article titles, e.g. ["FAISS", "Transformer model"].

    Returns:
        List of {"title": str, "content": str} dicts.
    """
    docs: List[dict] = []
    for topic in topics:
        topic = topic.strip()
        if not topic:
            continue

        # Use the full article extract via the query API for richer content
        api_url = (
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "query",
                "prop": "extracts",
                "titles": topic,
                "format": "json",
                "explaintext": "true",
                "redirects": "true",
            })
        )
        try:
            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": "rag-dashboard/1.0 (educational project)"},
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id == "-1":
                    logger.warning("Wikipedia: article not found — '%s'", topic)
                    continue
                title   = page.get("title", topic)
                extract = clean_text(page.get("extract", ""))
                if len(extract) < 100:          # skip stubs / disambiguation pages
                    logger.warning("Wikipedia: article too short, skipping — '%s'", title)
                    continue
                docs.append({"title": f"Wikipedia: {title}", "content": extract})
                logger.info("Wikipedia: fetched '%s' (%d chars)", title, len(extract))

        except Exception as exc:
            logger.warning("Wikipedia: could not fetch '%s': %s", topic, exc)

    return docs


# ---------------------------------------------------------------------------
# Document loading from uploaded file content
# ---------------------------------------------------------------------------

def load_text_from_upload(file_bytes: bytes, filename: str) -> List[dict]:
    """
    Parse uploaded file bytes into a list of document dicts.

    Supports:
        .txt  — treated as a single document
        .csv  — first text column is used; each row becomes a document

    Args:
        file_bytes: Raw bytes from st.file_uploader.
        filename:   Original filename (used to detect format).

    Returns:
        List of dicts: [{"title": ..., "content": ...}, ...]
    """
    import io

    filename_lower = filename.lower()

    if filename_lower.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="replace")
        text = clean_text(text)
        return [{"title": filename, "content": text}]

    elif filename_lower.endswith(".csv"):
        import csv

        docs: List[dict] = []
        reader = csv.DictReader(io.StringIO(file_bytes.decode("utf-8", errors="replace")))
        fields = reader.fieldnames or []

        # Pick the longest-looking text field heuristically
        text_field = _pick_text_field(fields)
        title_field = _pick_title_field(fields)

        for i, row in enumerate(reader):
            content = clean_text(row.get(text_field, ""))
            if not content:
                continue
            title = row.get(title_field, f"Row {i + 1}") if title_field else f"Row {i + 1}"
            docs.append({"title": str(title), "content": content})

        return docs

    else:
        raise ValueError(f"Unsupported file type: '{filename}'. Please upload a .txt or .csv file.")


def _pick_text_field(fields: List[str]) -> str:
    """Heuristically select the field most likely to contain long text."""
    preferred = ["text", "content", "body", "description", "abstract", "article", "summary"]
    for pref in preferred:
        for f in fields:
            if pref in f.lower():
                return f
    return fields[-1] if fields else ""


def _pick_title_field(fields: List[str]) -> str:
    """Heuristically select a title/heading field."""
    preferred = ["title", "headline", "name", "subject", "heading"]
    for pref in preferred:
        for f in fields:
            if pref in f.lower():
                return f
    return ""
