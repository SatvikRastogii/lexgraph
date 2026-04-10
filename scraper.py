'''import requests
from bs4 import BeautifulSoup
import os
import time
from tqdm import tqdm

# Configuration
SEARCH_QUERY = "Article 21 right to privacy Supreme Court"
OUTPUT_DIR = "legal_corpus_art21"
MAX_DOCUMENTS = 200
DELAY = 2  # seconds between requests, be respectful

os.makedirs(OUTPUT_DIR, exist_ok=True)

def search_indian_kanoon(query, page=0):
    """Search Indian Kanoon and return list of document URLs"""
    url = f"https://indiankanoon.org/search/"
    params = {
        "formInput": query,
        "pagenum": page
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all judgment links
        links = []
        for result in soup.find_all("div", class_="result"):
            a_tag = result.find("a", href=True)
            if a_tag and "/doc/" in a_tag["href"]:
                links.append("https://indiankanoon.org" + a_tag["href"])
        return links
    except Exception as e:
        print(f"Search error on page {page}: {e}")
        return []

def extract_judgment_text(url):
    """Extract clean text from a judgment page"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Get judgment title
        title = soup.find("h2", class_="doc-title")
        title_text = title.get_text(strip=True) if title else "Unknown"
        
        # Get main judgment text
        judgment_div = soup.find("div", id="judgments")
        if not judgment_div:
            judgment_div = soup.find("div", class_="judgments")
        
        if judgment_div:
            # Remove unwanted elements
            for tag in judgment_div.find_all(["script", "style", "a"]):
                tag.decompose()
            text = judgment_div.get_text(separator="\n", strip=True)
        else:
            # Fallback to body text
            text = soup.get_text(separator="\n", strip=True)
        
        # Basic cleaning
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        clean_text = "\n".join(lines)
        
        return title_text, clean_text
    
    except Exception as e:
        print(f"Extraction error for {url}: {e}")
        return None, None

def is_supreme_court(text):
    """Filter only Supreme Court judgments"""
    indicators = [
        "Supreme Court of India",
        "SUPREME COURT OF INDIA", 
        "Hon'ble Supreme Court",
        "This Court"
    ]
    return any(indicator in text[:500] for indicator in indicators)

def already_downloaded(url, output_dir):
    """Check if URL was already downloaded in any previous run"""
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            first_lines = f.read(200)
            if url in first_lines:
                return True
    return False

def main():
    print("Starting Indian Kanoon scraper...")
    print(f"Target: {MAX_DOCUMENTS} Supreme Court judgments")
    print(f"Query: {SEARCH_QUERY}\n")
    
    all_urls = []
    page = 0
    
    # Collect URLs across multiple pages
    print("Collecting judgment URLs...")
    while len(all_urls) < MAX_DOCUMENTS * 2:  # collect extra to filter
        urls = search_indian_kanoon(SEARCH_QUERY, page)
        if not urls:
            break
        all_urls.extend(urls)
        print(f"Page {page}: found {len(urls)} results, total: {len(all_urls)}")
        page += 1
        time.sleep(DELAY)
        if page > 20:  # max 20 pages
            break
    
    print(f"\nTotal URLs collected: {len(all_urls)}")
    print("Downloading and extracting judgments...\n")
    
    downloaded = 0
    failed = 0
    
    for url in tqdm(all_urls):
        if downloaded >= MAX_DOCUMENTS:
            break
            
        title, text = extract_judgment_text(url)
        
        if not text or len(text) < 1000:  # skip very short documents
            failed += 1
            continue
            
        if not is_supreme_court(text):  # skip non-Supreme Court
            failed += 1
            continue

        if already_downloaded(url, OUTPUT_DIR):
            continue
        
        # Save to file
        filename = f"judgment_{downloaded+1:03d}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {title}\n")
            f.write(f"SOURCE: {url}\n")
            f.write(f"{'='*50}\n\n")
            f.write(text)
        
        downloaded += 1
        time.sleep(DELAY)  # respectful delay
    
    print(f"\nCompleted!")
    print(f"Successfully downloaded: {downloaded} judgments")
    print(f"Skipped/failed: {failed} documents")
    print(f"Saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main() '''


"""
Indian Supreme Court Judgment Scraper
Optimized for GraphRAG indexing and RAGAS evaluation
Collects judgments across Articles 14, 19, 21, 32 and constitutional bench cases
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import json
import re
import hashlib
import logging
from datetime import datetime
from tqdm import tqdm

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

OUTPUT_DIR = "legal_corpus"
METADATA_FILE = "corpus_metadata.json"
DOWNLOADED_URLS_FILE = "downloaded_urls.json"
LOG_FILE = "scraper.log"

MAX_DOCUMENTS = 500
DELAY = 2.5
MAX_PAGES_PER_QUERY = 50
MIN_TEXT_LENGTH = 3000
MAX_TEXT_LENGTH = 150000

SEARCH_QUERIES = [
    {
        "query": "Article 21 right to life personal liberty Supreme Court constitution",
        "article": "Article 21",
        "target": 100
    },
    {
        "query": "Article 14 equality before law equal protection Supreme Court",
        "article": "Article 14",
        "target": 100
    },
    {
        "query": "Article 19 freedom of speech expression Supreme Court fundamental rights",
        "article": "Article 19",
        "target": 100
    },
    {
        "query": "Article 32 constitutional remedies writ jurisdiction Supreme Court",
        "article": "Article 32",
        "target": 100
    },
    {
        "query": "constitutional bench nine judge Supreme Court fundamental rights landmark",
        "article": "Constitutional Bench",
        "target": 100
    }
]

QUALITY_PHRASES = [
    "therefore", "accordingly", "held that", "we are of the opinion",
    "it is clear that", "the question is", "in our view", "we find that",
    "it is well settled", "the law is", "we hold that", "it must be noted",
    "in our considered opinion", "the ratio of", "the principle",
    "constitutional validity", "ultra vires", "violation of",
    "fundamental right", "right to", "judicial review"
]

SKIP_PHRASES = [
    "list on", "adjourned", "notice issued", "tag with",
    "heard and dismissed", "special leave petition dismissed",
    "liberty to", "two weeks", "four weeks", "stand over",
    "call on", "office report"
]

LEGAL_ENTITIES = [
    "petitioner", "respondent", "appellant", "constitution",
    "article", "section", "act", "court", "judge", "justice",
    "bench", "judgment", "order", "writ", "petition"
]

# ─── LOGGING SETUP ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def load_json(filepath, default):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_url_hash(url):
    return hashlib.md5(url.encode()).hexdigest()[:12]

def clean_text(raw_text):
    """Deep clean judgment text for GraphRAG indexing"""
    text = re.sub(r'\n{3,}', '\n\n', raw_text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    text = re.sub(r'(REPORTABLE|NOT REPORTABLE|IN THE SUPREME COURT OF INDIA)', '', text)
    text = re.sub(r'\[(\d{4})\]\s*(\d+)\s*SCC\s*(\d+)', r'[\1] \2 SCC \3', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    lines = text.split('\n')
    lines = [line for line in lines if len(line.strip()) > 3 or line.strip() == '']
    return '\n'.join(lines).strip()

def extract_metadata_from_text(text, url):
    """Extract structured metadata for RAGAS ground truth generation"""
    metadata = {
        "url": url,
        "articles_cited": [],
        "judges": [],
        "year": None,
        "case_number": None,
        "has_dissent": False,
        "bench_size": None
    }

    articles = re.findall(r'Article\s+(\d+[A-Z]?)', text)
    metadata["articles_cited"] = list(set(articles))[:10]

    year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', text[:500])
    if year_match:
        metadata["year"] = int(year_match.group(1))

    case_match = re.search(
        r'(Writ Petition|Civil Appeal|Criminal Appeal|SLP).*?No\.?\s*(\d+)',
        text[:300], re.IGNORECASE
    )
    if case_match:
        metadata["case_number"] = case_match.group(0)[:100]

    metadata["has_dissent"] = bool(re.search(
        r'dissent|dissenting|I disagree|contrary view',
        text, re.IGNORECASE
    ))

    cji_mentions = len(re.findall(r'J\.\s*[A-Z]', text[:1000]))
    if cji_mentions >= 9:
        metadata["bench_size"] = "Constitutional Bench (9+)"
    elif cji_mentions >= 5:
        metadata["bench_size"] = "Constitution Bench (5)"
    elif cji_mentions >= 3:
        metadata["bench_size"] = "Division Bench (3)"
    else:
        metadata["bench_size"] = "Division Bench (2)"

    return metadata

def is_quality_judgment(text, title):
    """Multi-factor quality filter. Returns (bool, reason)"""
    if len(text) < MIN_TEXT_LENGTH:
        return False, f"Too short ({len(text)} chars)"

    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Too long ({len(text)} chars)"

    text_lower = text.lower()

    for phrase in SKIP_PHRASES:
        if phrase in text_lower[:300]:
            return False, f"Procedural document"

    quality_count = sum(1 for phrase in QUALITY_PHRASES if phrase in text_lower)
    if quality_count < 3:
        return False, f"Low quality score ({quality_count}/3)"

    entity_count = sum(1 for entity in LEGAL_ENTITIES if entity in text_lower)
    if entity_count < 5:
        return False, f"Low entity density ({entity_count}/5)"

    sc_indicators = [
        "supreme court of india",
        "hon'ble supreme court",
        "this court",
        "the apex court"
    ]
    if not any(ind in text_lower[:1000] for ind in sc_indicators):
        return False, "Not a Supreme Court judgment"

    return True, "Quality judgment"

def format_for_graphrag(title, date, article_focus, url, metadata, text):
    """
    Format judgment specifically for GraphRAG entity extraction.
    Structured header helps LLM identify entities and relationships clearly.
    """
    header = f"""CASE TITLE: {title}
DATE: {date}
PRIMARY ARTICLE: {article_focus}
SOURCE: {url}
YEAR: {metadata.get('year', 'Unknown')}
ARTICLES CITED: {', '.join(['Article ' + a for a in metadata.get('articles_cited', [])])}
BENCH TYPE: {metadata.get('bench_size', 'Unknown')}
HAS DISSENT: {'Yes' if metadata.get('has_dissent') else 'No'}
CASE NUMBER: {metadata.get('case_number', 'Unknown')}
{'=' * 70}

"""
    return header + text

# ─── SCRAPING FUNCTIONS ───────────────────────────────────────────────────────

def search_indian_kanoon(query, page=0):
    """Search Indian Kanoon and return judgment URLs"""
    url = "https://indiankanoon.org/search/"
    params = {"formInput": query, "pagenum": page}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all document links directly
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "/doc/" in href and href not in links:
                full_url = "https://indiankanoon.org" + href
                links.append(full_url)

        return links

    except requests.exceptions.RequestException as e:
        logger.warning(f"Search error page {page}: {e}")
        return []


def extract_judgment(url):
    """Extract and structure judgment from Indian Kanoon"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title_tag = soup.find("h2", class_="doc-title") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"

        date_tag = soup.find("div", class_="docsource_main")
        date = date_tag.get_text(strip=True) if date_tag else "Unknown Date"

        judgment_div = (
            soup.find("div", id="judgments") or
            soup.find("div", class_="judgments") or
            soup.find("div", id="doc_content") or
            soup.find("div", class_="doc_content")
        )

        if not judgment_div:
            return None, None, None

        for tag in judgment_div.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()

        for a_tag in judgment_div.find_all("a"):
            a_tag.replace_with(a_tag.get_text())

        raw_text = judgment_div.get_text(separator="\n", strip=True)
        return title, date, raw_text

    except Exception as e:
        logger.warning(f"Error extracting {url}: {e}")
        return None, None, None

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    downloaded_urls = set(load_json(DOWNLOADED_URLS_FILE, []))
    corpus_metadata = load_json(METADATA_FILE, {"documents": [], "stats": {}})
    existing_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')])

    logger.info("=" * 60)
    logger.info("Indian Supreme Court Judgment Scraper")
    logger.info(f"Target: {MAX_DOCUMENTS} quality judgments")
    logger.info(f"Previously downloaded: {existing_count} judgments")
    logger.info("=" * 60)

    total_downloaded = existing_count
    session_downloaded = 0
    session_failed = 0
    session_skipped = 0

    for query_config in SEARCH_QUERIES:
        query = query_config["query"]
        article_focus = query_config["article"]
        query_target = query_config["target"]

        logger.info(f"\nQuery: {article_focus} | Target: {query_target} judgments")

        query_downloaded = 0

        for page in range(MAX_PAGES_PER_QUERY):
            if total_downloaded >= MAX_DOCUMENTS:
                break
            if query_downloaded >= query_target:
                break

            urls = search_indian_kanoon(query, page)
            if not urls:
                break

            logger.info(f"  Page {page}: {len(urls)} URLs")

            for url in urls:
                if total_downloaded >= MAX_DOCUMENTS:
                    break
                if query_downloaded >= query_target:
                    break
                if url in downloaded_urls:
                    session_skipped += 1
                    continue

                title, date, raw_text = extract_judgment(url)
                downloaded_urls.add(url)

                if not raw_text:
                    session_failed += 1
                    time.sleep(DELAY)
                    continue

                clean = clean_text(raw_text)
                is_quality, reason = is_quality_judgment(clean, title)

                if not is_quality:
                    logger.debug(f"  Skipped: {reason}")
                    session_failed += 1
                    time.sleep(DELAY)
                    continue

                metadata = extract_metadata_from_text(clean, url)
                metadata["article_focus"] = article_focus
                metadata["title"] = title
                metadata["date"] = date
                metadata["filename"] = f"judgment_{total_downloaded+1:04d}.txt"

                formatted = format_for_graphrag(title, date, article_focus, url, metadata, clean)

                filepath = os.path.join(OUTPUT_DIR, metadata["filename"])
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(formatted)

                corpus_metadata["documents"].append(metadata)
                total_downloaded += 1
                query_downloaded += 1
                session_downloaded += 1

                logger.info(
                    f"  [{total_downloaded}/{MAX_DOCUMENTS}] "
                    f"{title[:55]} | "
                    f"Arts: {','.join(metadata['articles_cited'][:3])}"
                )

                time.sleep(DELAY)

            save_json(DOWNLOADED_URLS_FILE, list(downloaded_urls))
            save_json(METADATA_FILE, corpus_metadata)
            time.sleep(DELAY)

    # ─── FINAL STATS ──────────────────────────────────────────────────────────

    all_docs = corpus_metadata["documents"]
    article_counts = {}
    year_distribution = {}
    dissent_count = 0
    constitutional_bench_count = 0

    for doc in all_docs:
        focus = doc.get("article_focus", "Unknown")
        article_counts[focus] = article_counts.get(focus, 0) + 1
        year = doc.get("year")
        if year:
            decade = f"{(year // 10) * 10}s"
            year_distribution[decade] = year_distribution.get(decade, 0) + 1
        if doc.get("has_dissent"):
            dissent_count += 1
        if doc.get("bench_size", "").startswith("Constitutional"):
            constitutional_bench_count += 1

    corpus_metadata["stats"] = {
        "total_documents": total_downloaded,
        "session_downloaded": session_downloaded,
        "session_failed": session_failed,
        "session_skipped": session_skipped,
        "article_distribution": article_counts,
        "decade_distribution": year_distribution,
        "judgments_with_dissent": dissent_count,
        "constitutional_bench_judgments": constitutional_bench_count,
        "timestamp": datetime.now().isoformat()
    }

    save_json(METADATA_FILE, corpus_metadata)
    save_json(DOWNLOADED_URLS_FILE, list(downloaded_urls))

    logger.info("\n" + "=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info(f"Total: {total_downloaded} | New: {session_downloaded} | Failed: {session_failed} | Duplicates: {session_skipped}")
    logger.info("\nArticle Distribution:")
    for article, count in article_counts.items():
        logger.info(f"  {article}: {count}")
    logger.info("\nDecade Distribution:")
    for decade, count in sorted(year_distribution.items()):
        logger.info(f"  {decade}: {count}")
    logger.info(f"\nDissenting opinions: {dissent_count}")
    logger.info(f"Constitutional bench: {constitutional_bench_count}")

    # Generate benchmark questions
    generate_benchmark_questions(corpus_metadata)

def generate_benchmark_questions(corpus_metadata):
    """
    Auto-generate benchmark questions for RAGAS evaluation.
    Covers all query types to demonstrate GraphRAG vs naive RAG differences.
    """
    has_dissent = corpus_metadata["stats"].get("judgments_with_dissent", 0) > 0
    has_constitutional = corpus_metadata["stats"].get("constitutional_bench_judgments", 0) > 0

    questions = {
        "single_hop_factual": [
            "What does Article 21 of the Indian Constitution guarantee?",
            "What is the right to equality under Article 14?",
            "What freedoms are protected under Article 19?",
            "What remedies does Article 32 provide?"
        ],
        "multi_hop_relational": [
            "How are Articles 14, 19, and 21 interconnected in Supreme Court judgments?",
            "Which legal principles from early Article 21 cases were expanded in privacy judgments?",
            "How has the golden triangle of Articles 14, 19, and 21 evolved across constitutional bench decisions?",
            "Which judges have consistently interpreted fundamental rights expansively across multiple articles?"
        ],
        "global_thematic": [
            "What are the dominant themes across Supreme Court fundamental rights jurisprudence?",
            "How has the Supreme Court balanced individual rights against state power?",
            "What constitutional principles appear most frequently across these judgments?",
            "What patterns exist in how the court interprets reasonable restrictions?"
        ],
        "cross_document_reasoning": [
            "How has the interpretation of Article 21 evolved from the 1950s to 2020s?",
            "Which landmark cases form the foundational lineage of privacy rights in India?",
            "How do dissenting opinions reflect evolving constitutional philosophy?",
            "What is the relationship between Article 32 petitions and expansion of fundamental rights?"
        ],
        "entity_relationship": [
            "Which Supreme Court judges authored the most significant fundamental rights judgments?",
            "How are the Puttaswamy, Maneka Gandhi, and Kesavananda Bharati cases connected?",
            "Which petitioners had the most landmark victories in fundamental rights cases?",
            "What is the relationship between bench size and landmark judgment significance?"
        ]
    }

    if has_dissent:
        questions["multi_hop_relational"].append(
            "In which fundamental rights cases did dissenting opinions later become majority views?"
        )
    if has_constitutional:
        questions["global_thematic"].append(
            "What distinguishes constitutional bench judgments in fundamental rights cases?"
        )

    save_json("benchmark_questions.json", questions)
    total = sum(len(q) for q in questions.values())
    logger.info(f"\nGenerated {total} benchmark questions saved to benchmark_questions.json")
    logger.info("Use these for GraphRAG vs Naive RAG RAGAS evaluation")

if __name__ == "__main__":
    main()
