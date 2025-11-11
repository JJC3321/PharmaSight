"""
Custom tools for biotech research and analysis
"""
import os
import re
import tempfile
from typing import Dict, Any, List, Optional, Iterable, Union
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF


NO_COMPANY_DOCS_MSG = "No company process documentation provided."

DEFAULT_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
}

POW_CHALLENGE_RE = re.compile(r'POW_CHALLENGE\s*=\s*"([^"]+)"')
POW_DIFFICULTY_RE = re.compile(r'POW_DIFFICULTY\s*=\s*"([^"]+)"')
POW_COOKIE_NAME_RE = re.compile(r'POW_COOKIE_NAME\s*=\s*"([^"]+)"')
POW_COOKIE_PATH_RE = re.compile(r'POW_COOKIE_PATH\s*=\s*"([^"]+)"')


def is_pdf_response(resp: requests.Response) -> bool:
    content_type = (resp.headers.get("Content-Type") or "").lower()
    return "pdf" in content_type or resp.content.startswith(b"%PDF")


def solve_pow(challenge: str, difficulty: int) -> int:
    """
    Solve the simple proof-of-work challenge required by PMC downloads.
    Finds the smallest nonce such that sha256(f\"{challenge}{nonce}\") starts with
    the required number of zeroes (hex).
    """
    import hashlib

    prefix = "0" * max(difficulty, 1)
    nonce = 0
    while True:
        digest = hashlib.sha256(f"{challenge}{nonce}".encode("utf-8")).hexdigest()
        if digest.startswith(prefix):
            return nonce
        nonce += 1


def download_pmc_pdf(pdf_url: str, headers: Dict[str, str]) -> bytes:
    """
    Download a PDF from PMC, solving the proof-of-work challenge when required.
    """
    session = requests.Session()
    resp = session.get(pdf_url, headers=headers, timeout=45)
    if is_pdf_response(resp):
        return resp.content

    text = resp.text
    challenge_match = POW_CHALLENGE_RE.search(text)
    if not challenge_match:
        raise ValueError(f"PMC returned non-PDF content and no POW challenge for URL: {pdf_url}")

    difficulty_match = POW_DIFFICULTY_RE.search(text)
    cookie_name_match = POW_COOKIE_NAME_RE.search(text)
    cookie_path_match = POW_COOKIE_PATH_RE.search(text)

    challenge = challenge_match.group(1)
    difficulty = int(difficulty_match.group(1)) if difficulty_match else 4
    cookie_name = cookie_name_match.group(1) if cookie_name_match else "cloudpmc-viewer-pow"
    cookie_path = cookie_path_match.group(1) if cookie_path_match else "/"

    nonce = solve_pow(challenge, difficulty)
    cookie_value = f"{challenge},{nonce}"

    domain = urlparse(pdf_url).hostname
    session.cookies.set(cookie_name, cookie_value, domain=domain, path=cookie_path)

    # Retry with the new cookie (first the original URL, then append download=1 if needed)
    resp = session.get(pdf_url, headers=headers, timeout=45)
    if not is_pdf_response(resp):
        alt_url = pdf_url + ("&download=1" if "?" in pdf_url else "?download=1")
        resp = session.get(alt_url, headers=headers, timeout=45)

    if not is_pdf_response(resp):
        raise ValueError(f"PMC returned non-PDF content after solving POW challenge (content-type: {resp.headers.get('Content-Type')})")

    return resp.content


def summarize_drug_mentions(
    text: str,
    drug_name: str,
    source_label: str,
    *,
    max_snippets: int = 3,
    context_chars: int = 120,
) -> str:
    """
    Extract contextual snippets where the drug name appears in the provided text.

    Args:
        text: Body of text to inspect.
        drug_name: Name of the drug to highlight.
        source_label: Human-readable description of the source.
        max_snippets: Maximum number of snippets to include.
        context_chars: Number of characters to show on each side of the match.
    """
    clean_drug = (drug_name or "").strip()
    if not clean_drug or not text:
        return ""

    pattern = re.compile(re.escape(clean_drug), re.IGNORECASE)
    snippets: List[str] = []

    for match in pattern.finditer(text):
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)
        snippet = text[start:end].strip()
        snippet = re.sub(r"\s+", " ", snippet)
        snippets.append(snippet)
        if len(snippets) >= max_snippets:
            break

    if not snippets:
        return f"[No direct mentions of '{clean_drug}' detected in {source_label}.]"

    lines = [f"[Direct mentions of '{clean_drug}' detected in {source_label}:]"]
    for snippet in snippets:
        lines.append(f"- ...{snippet}...")

    return "\n".join(lines)


def _pdf_parse_failed(content: Optional[str]) -> bool:
    """
    Determine if a PDF parsing attempt failed based on the returned content.
    """
    if not content:
        return True

    normalized = content.strip().lower()
    if not normalized:
        return True

    error_markers = [
        "error parsing pdf with landingai ade",
        "error extracting pdf text",
        "landingai ade sdk not installed",
        "landingai api key not found",
    ]

    return any(marker in normalized for marker in error_markers)


def parse_pdf_with_landingai_http(
    pdf_source: str | Path,
    api_key: str,
    model_name: str,
    base_url: str,
) -> Optional[Dict[str, Any]]:
    """
    Call the ADE parse endpoint directly via HTTP as a fallback when the SDK fails.
    """
    endpoint = base_url.rstrip("/") + "/v1/ade/parse"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    files = None
    data = {"model": model_name}

    if isinstance(pdf_source, Path):
        try:
            files = {"document": open(pdf_source, "rb")}
        except OSError as exc:
            print(f"  LandingAI HTTP fallback error opening file: {exc}")
            return None
    else:
        data["document_url"] = pdf_source
    
    document_repr = data.get("document_url")
    if not document_repr:
        document_repr = str(pdf_source)

    print(
        "  [LandingAI HTTP fallback] Prepared request:",
        f"endpoint={endpoint}, model={model_name}, document={document_repr}",
    )
    
    try:
        response = requests.post(endpoint, headers=headers, data=data, files=files, timeout=90)
        if response.status_code >= 400:
            print(f"  LandingAI HTTP fallback failed: {response.status_code} - {response.text}")
            return None
        return response.json()
    except Exception as exc:
        print(f"  LandingAI HTTP fallback error: {exc}")
        return None
    finally:
        if isinstance(pdf_source, Path) and files and files.get("document"):
            try:
                files["document"].close()
            except Exception:
                pass


def search_pubmed_papers(
    query: str,
    max_results: int = 5,
    return_pmids: bool = False,
    drug_name: str | None = None,
    indication: str | None = None,
) -> str | tuple[str, List[str]]:
    """
    Search PubMed for biotech and clinical trial research papers.
    
    Args:
        query: Search query (e.g., "pembrolizumab phase 3 clinical trial")
        max_results: Maximum number of results to return
        return_pmids: If True, return (results_string, pmid_list) tuple
    
    Returns:
        String containing paper titles, abstracts, and URLs
        Or tuple of (string, list) if return_pmids=True
    """
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        def _sanitize_term(value: str | None) -> str | None:
            if not value:
                return None
            cleaned = value.replace('"', "").strip()
            return cleaned or None

        search_terms: List[str] = []
        safe_drug = _sanitize_term(drug_name)
        safe_indication = _sanitize_term(indication)

        if safe_drug:
            search_terms.append(f'("{safe_drug}"[Title/Abstract])')
        if safe_indication:
            search_terms.append(f'("{safe_indication}"[Title/Abstract])')

        if query and query.strip():
            search_terms.append(f"({query})")

        search_terms.append(
            '("clinical trial"[Title/Abstract] OR "clinical trials as topic"[MeSH Terms])'
        )

        term_query = " AND ".join(search_terms)

        # Search for papers
        search_url = f"{base_url}esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": term_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        
        search_response = requests.get(search_url, params=search_params)
        search_data = search_response.json()
        
        if "esearchresult" not in search_data or "idlist" not in search_data["esearchresult"]:
            message = "No papers found for the given query."
            if return_pmids:
                return message, []
            return message
        
        paper_ids = search_data["esearchresult"]["idlist"]
        
        if not paper_ids:
            message = "No papers found for the given query."
            if return_pmids:
                return message, []
            return message
        
        # Fetch paper details
        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(paper_ids),
            "retmode": "xml"
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params)
        soup = BeautifulSoup(fetch_response.content, "xml")
        
        papers = []
        filtered_pmids: List[str] = []
        skipped_papers = []
        articles = soup.find_all("PubmedArticle")
        
        for article in articles:
            try:
                title = article.find("ArticleTitle").text if article.find("ArticleTitle") else "No title"
                abstract_sections = article.find_all("AbstractText")
                if abstract_sections:
                    abstract_text = "\n".join(
                        section.text.strip()
                        for section in abstract_sections
                        if section and section.text
                    ).strip()
                else:
                    abstract = article.find("AbstractText")
                    abstract_text = abstract.text.strip() if abstract and abstract.text else "No abstract available"
                pmid = article.find("PMID").text if article.find("PMID") else "No ID"
                mention_summary = ""
                if drug_name:
                    mention_summary = summarize_drug_mentions(
                        abstract_text,
                        drug_name,
                        f"PubMed abstract for PMID {pmid}",
                    )
                has_direct_mention = bool(
                    mention_summary and mention_summary.startswith("[Direct mentions")
                )
                
                excerpt_limit = 1500
                abstract_excerpt = abstract_text[:excerpt_limit]
                if len(abstract_text) > excerpt_limit:
                    abstract_excerpt += "..."
                
                mention_block = f"{mention_summary}\n" if mention_summary else ""

                paper_info = f"""
Title: {title}
PMID: {pmid}
URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/
Abstract:
{mention_block}{abstract_excerpt}

---
"""
                if has_direct_mention or not drug_name:
                    papers.append(paper_info)
                    filtered_pmids.append(pmid)
                else:
                    skipped_papers.append(paper_info)
            except Exception as e:
                continue
        
        result = "\n".join(papers)

        if not papers:
            if skipped_papers:
                result = (
                    f"No PubMed abstracts explicitly mentioning '{drug_name}' were found. "
                    "Showing closely related results (may require manual validation):\n"
                    + "\n".join(skipped_papers)
                )
            else:
                result = "Error parsing paper data."
        
        if return_pmids:
            return result, filtered_pmids
        return result
        
    except Exception as e:
        error_msg = f"Error searching PubMed: {str(e)}"
        if return_pmids:
            return error_msg, []
        return error_msg


def analyze_clinical_trial_phases(drug_name: str) -> str:
    """
    Search ClinicalTrials.gov for information about a specific drug's clinical trial phases.
    
    Args:
        drug_name: Name of the drug or compound
    
    Returns:
        Information about clinical trial phases, status, and results
    """
    try:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.term": drug_name,
            "pageSize": 10,
            "format": "json"
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "studies" not in data or len(data["studies"]) == 0:
            return f"No clinical trials found for {drug_name}"
        
        trials_info = []
        for study in data["studies"][:5]:
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            
            trial_info = f"""
Trial: {identification.get("officialTitle", "No title")}
NCT ID: {identification.get("nctId", "No ID")}
Phase: {design.get("phases", ["Unknown"])}
Status: {status.get("overallStatus", "Unknown")}
Start Date: {status.get("startDateStruct", {}).get("date", "Unknown")}
Completion Date: {status.get("completionDateStruct", {}).get("date", "Unknown")}
---
"""
            trials_info.append(trial_info)
        
        return "\n".join(trials_info)
        
    except Exception as e:
        return f"Error fetching clinical trial data: {str(e)}"

def analyze_drug_mechanism(drug_name: str, target: str) -> str:
    """
    Provide basic information about drug mechanism (simplified version).
    
    Args:
        drug_name: Name of the drug
        target: Therapeutic target or indication
    
    Returns:
        Basic mechanism information template
    """
    return f"""
Drug Mechanism Analysis for {drug_name}
Target/Indication: {target}

Key Considerations for Clinical Trial Success:
1. Mechanism of Action - Novel vs established drug class
2. Precedent success/failure in this therapeutic area
3. Key risk factors - Safety profile, efficacy endpoints
4. Competitive landscape - Existing treatments
5. Probability factors - Patient selection, biomarkers

Note: For detailed mechanism analysis, consult scientific literature and clinical trial protocols.
"""


def search_fda_approvals(keyword: str) -> str:
    """
    Search for FDA approval information related to a drug or therapeutic area.
    
    Args:
        keyword: Drug name or therapeutic area
    
    Returns:
        FDA approval information template
    """
    return f"""
Searching FDA records for: {keyword}

For detailed FDA approval information, check:
- https://www.fda.gov/drugs/drug-approvals-and-databases
- https://www.accessdata.fda.gov/scripts/cder/daf/

Key FDA Approval Considerations:
1. Phase 3 trial success with statistical significance
2. Safety profile and adverse event monitoring
3. Unmet medical need and competitive alternatives
4. Risk-benefit analysis
5. Advisory committee recommendations (if applicable)
6. Manufacturing and quality control processes
7. Post-marketing surveillance requirements
"""


def get_pubmed_pdf_urls(pmid_list: List[str]) -> Dict[str, Optional[str]]:
    """
    Try to get PDF URLs from PubMed Central for open access papers.
    
    Args:
        pmid_list: List of PubMed IDs
    
    Returns:
        Dictionary mapping PMID to PDF URL (or None if not available)
    """
    pdf_urls = {}
    
    try:
        # Check if papers are in PMC (open access)
        pmids_str = ",".join(pmid_list)
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmids_str}&format=json"
        response = requests.get(pmc_url, headers=DEFAULT_REQUEST_HEADERS)
        data = response.json()
        
        if "records" in data:
            for record in data["records"]:
                pmid = record.get("pmid")
                pmcid = record.get("pmcid")
                
                if pmid and pmcid:
                    # Resolve the most specific PDF URL available for the PMC article
                    pdf_url = resolve_pmc_pdf_url(pmcid)
                    pdf_urls[pmid] = pdf_url
                elif pmid:
                    pdf_urls[pmid] = None
        
        return pdf_urls
        
    except Exception as e:
        print(f"Error fetching PDF URLs: {str(e)}")
        return {pmid: None for pmid in pmid_list}


def resolve_pmc_pdf_url(pmcid: str) -> Optional[str]:
    """
    Resolve the full PDF URL for a PMC article, including filenames when required.
    
    Some PMC entries expose PDFs at URLs like `/pdf/<filename>.pdf`. This helper
    inspects the article page to capture the exact link when present.
    """
    article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    try:
        response = requests.get(article_url, timeout=20, headers=DEFAULT_REQUEST_HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # First try the citation meta tag which usually contains the full PDF URL
        meta_pdf = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if meta_pdf and meta_pdf.get("content"):
            href = meta_pdf["content"]
            return href if href.startswith("http") else urljoin(article_url, href)
        
        # Prefer links that explicitly end with .pdf (most accurate)
        pdf_link = soup.find("a", href=re.compile(r"/pmc/articles/[^/]+/pdf/[^/]+\.pdf"))
        if pdf_link and pdf_link.get("href"):
            href = pdf_link["href"]
            return href if href.startswith("http") else urljoin("https://www.ncbi.nlm.nih.gov", href)
        
        # Fallback for PDFs served at the directory endpoint
        pdf_link = soup.find("a", href=re.compile(r"/pmc/articles/[^/]+/pdf/?$"))
        if pdf_link and pdf_link.get("href"):
            href = pdf_link["href"]
            return href if href.startswith("http") else urljoin("https://www.ncbi.nlm.nih.gov", href + "/?download=1")
        
        # Final fallback: generic PDF directory with download hint
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/?download=1"
    except Exception as exc:
        print(f"  Warning: unable to resolve PDF filename for {pmcid}: {exc}")
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/?download=1"


def parse_pdf_with_landingai(pdf_url: str, drug_name: str, indication: str) -> str:
    """
    Parse a research paper PDF using the LandingAI ADE document parser.
    
    Args:
        pdf_url: URL to the PDF paper
        drug_name: Name of the drug to focus analysis on
        indication: Medical indication/disease
    
    Returns:
        Structured data extracted from the paper
    """
    try:
        from landingai_ade import LandingAIADE
    except ImportError:
        return "\nLandingAI ADE SDK not installed. Install with: pip install landingai-ade\n"
    
    api_key = os.getenv("VISION_AGENT_API_KEY")
    if not api_key:
        return "LandingAI API key not found. Skipping advanced PDF parsing."
    model_name = os.getenv("LANDINGAI_ADE_MODEL", "dpt-2-latest")
    base_url = os.getenv("LANDINGAI_ADE_BASE_URL", "https://api.va.landing.ai")

    temp_pdf_path: Optional[Path] = None
    
    try:
        headers = dict(DEFAULT_REQUEST_HEADERS)
        if "/pdf/" in pdf_url:
            headers["Referer"] = pdf_url.split("/pdf/")[0] + "/"
        
        print(f"  Downloading PDF from {pdf_url[:80]}...")
        pdf_bytes = download_pmc_pdf(pdf_url, headers=headers)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            temp_pdf_path = Path(tmp_file.name)
        
        ade_client = LandingAIADE(apikey=api_key, base_url=base_url)
        print(f"  Parsing PDF with LandingAI ADE (model: {model_name}) via local file...")
        ade_response = ade_client.parse(document=temp_pdf_path, model=model_name)
        
        markdown_content = getattr(ade_response, "markdown", None)
        chunks = getattr(ade_response, "chunks", None)
        
        if markdown_content:
            print(f"  LandingAI ADE parse succeeded (markdown length: {len(markdown_content)})")
            parsed_body = markdown_content.strip()
        elif chunks:
            chunk_count = len(chunks)
            print(f"  LandingAI ADE parse returned {chunk_count} chunk(s) but no markdown.")
            chunk_markdown = [
                getattr(chunk, "markdown", "").strip()
                for chunk in chunks
                if getattr(chunk, "markdown", "").strip()
            ]
            parsed_body = "\n".join(chunk_markdown) if chunk_markdown else "[No structured content returned by LandingAI ADE]"
        else:
            print("  LandingAI ADE parse returned no content.")
            parsed_body = "[No structured content returned by LandingAI ADE]"
        
        mention_summary = summarize_drug_mentions(
            parsed_body,
            drug_name,
            "LandingAI ADE parsed paper",
        )
        if mention_summary:
            parsed_body = f"{mention_summary}\n\n{parsed_body}"

        header = (
            "=== DETAILED PAPER ANALYSIS (via LandingAI ADE) ===\n"
            f"Source URL: {pdf_url}\n"
            f"Drug: {drug_name}\n"
            f"Indication: {indication}\n\n"
        )
        return f"\n{header}{parsed_body}\n"
    
    except Exception as e:
        print(f"  LandingAI ADE parse failed: {e}")
        
        error_text = str(e)
        if any(code in error_text for code in ("Internal Server Error", "500", "422")) and temp_pdf_path and temp_pdf_path.exists():
            print("  Attempting LandingAI HTTP fallback with file upload...")
            
            http_result = parse_pdf_with_landingai_http(temp_pdf_path, api_key, model_name, base_url)
            if http_result:
                markdown_content = http_result.get("markdown")
                chunks = http_result.get("chunks")
                
                if markdown_content:
                    parsed_body = markdown_content.strip()
                elif chunks:
                    chunk_markdown = [
                        chunk.get("markdown", "").strip()
                        for chunk in chunks
                        if isinstance(chunk, dict) and chunk.get("markdown")
                    ]
                    parsed_body = "\n".join(chunk_markdown) if chunk_markdown else "[No structured content returned by LandingAI ADE HTTP fallback]"
                else:
                    parsed_body = "[No structured content returned by LandingAI ADE HTTP fallback]"
                
                mention_summary = summarize_drug_mentions(
                    parsed_body,
                    drug_name,
                    "LandingAI ADE HTTP fallback",
                )
                if mention_summary:
                    parsed_body = f"{mention_summary}\n\n{parsed_body}"

                header = (
                    "=== DETAILED PAPER ANALYSIS (via LandingAI ADE) ===\n"
                    f"Source URL: {pdf_url}\n"
                    f"Drug: {drug_name}\n"
                    f"Indication: {indication}\n\n"
                )
                return f"\n{header}{parsed_body}\n"
        
        return f"\nError parsing PDF with LandingAI ADE: {str(e)}\n"
    
    finally:
        if temp_pdf_path and temp_pdf_path.exists():
            try:
                temp_pdf_path.unlink()
            except OSError:
                pass
    


def search_and_parse_pubmed_papers(query: str, drug_name: str, indication: str, max_results: int = 3, max_pdfs: int = 2) -> str:
    """
    Enhanced PubMed search that also downloads and parses available PDFs.
    
    Args:
        query: Search query
        drug_name: Drug name for focused analysis
        indication: Medical indication
        max_results: Maximum number of papers to search
        max_pdfs: Maximum number of PDFs to parse (to control time/API usage)
    
    Returns:
        Combined results from abstracts and parsed PDFs
    """
    print("  Searching PubMed...")
    
    # Get PubMed results with PMIDs
    pubmed_results, pmid_list = search_pubmed_papers(
        query,
        max_results=max_results,
        return_pmids=True,
        drug_name=drug_name,
        indication=indication,
    )
    
    combined_results = f"=== PUBMED ABSTRACTS ===\n{pubmed_results}\n"
    
    if not pmid_list:
        return combined_results
    
    # Try to get PDF URLs
    print(f"  Checking for available PDFs from {len(pmid_list)} papers...")
    pdf_urls = get_pubmed_pdf_urls(pmid_list)
    
    # Parse available PDFs
    pdf_count = 0
    for pmid, pdf_url in pdf_urls.items():
        if pdf_url and pdf_count < max_pdfs:
            print(f"  Found PDF for PMID {pmid}, parsing...")
            
            # Try LandingAI first
            parsed_content = parse_pdf_with_landingai(pdf_url, drug_name, indication)

            if _pdf_parse_failed(parsed_content):
                print("  LandingAI parse failed or returned no usable content; trying PyMuPDF fallback.")
                parsed_content = parse_pdf_with_pymupdf(pdf_url, drug_name)

            if _pdf_parse_failed(parsed_content):
                print("  Skipping this PDF because parsing was unsuccessful.")
                continue

            combined_results += f"\n=== FULL PAPER ANALYSIS - PMID {pmid} ===\n{parsed_content}\n"
            pdf_count += 1
    
    if pdf_count == 0:
        combined_results += "\n(Note: No open-access PDFs were available for detailed parsing. Analysis based on abstracts only.)\n"
    else:
        combined_results += f"\n(Successfully parsed {pdf_count} full-text paper(s))\n"
    
    return combined_results


def parse_pdf_with_pymupdf(pdf_url: str, drug_name: str) -> str:
    """
    Download and parse a research paper PDF using PyMuPDF (fallback method).
    
    Args:
        pdf_url: URL to the PDF paper
        drug_name: Name of the drug
    
    Returns:
        Extracted text from the PDF
    """
    try:
        # Download PDF
        print(f"  Downloading PDF from {pdf_url[:50]}...")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            pdf_path = tmp_file.name
        
        print(f"  Extracting text from PDF...")
        
        # Extract text using PyMuPDF
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num, page in enumerate(doc):
            full_text += f"\n--- Page {page_num + 1} ---\n"
            full_text += page.get_text()
        
        doc.close()
        
        # Clean up temp file
        try:
            os.remove(pdf_path)
        except:
            pass
        
        # Truncate if too long (keep first 20,000 chars for context limits)
        if len(full_text) > 20000:
            full_text = full_text[:20000] + "\n\n... [Truncated for length] ..."

        mention_summary = summarize_drug_mentions(
            full_text,
            drug_name,
            "PyMuPDF extracted PDF text",
        )
        summary_block = f"{mention_summary}\n\n" if mention_summary else ""
        
        return f"\n=== PAPER FULL TEXT ===\n{summary_block}{full_text}\n"
        
    except Exception as e:
        # Clean up temp file on error
        try:
            if 'pdf_path' in locals():
                os.remove(pdf_path)
        except:
            pass
        return f"\nError extracting PDF text: {str(e)}\n"


def _read_text_file(path: Path) -> str:
    """
    Internal helper to read a plain text-like file with sensible defaults.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def _extract_text_from_local_pdf(path: Path) -> str:
    """
    Extract text from a local PDF file using PyMuPDF.
    """
    try:
        doc = fitz.open(path)
    except Exception as exc:
        return f"[Error opening PDF {path.name}: {exc}]"

    pages: List[str] = []
    for page_num, page in enumerate(doc):
        header = f"\n--- Page {page_num + 1} ({path.name}) ---\n"
        pages.append(header + page.get_text())
    doc.close()

    extracted = "".join(pages).strip()
    if not extracted:
        return f"[No extractable text content found in {path.name}]"

    if len(extracted) > 20000:
        extracted = extracted[:20000] + "\n\n... [Truncated for length] ..."
    return extracted


def _gather_candidate_files(paths: Iterable[Union[str, Path]]) -> List[Path]:
    """
    Resolve incoming paths to a flat list of files to ingest.
    """
    files: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file():
                    files.append(child)
        elif path.is_file():
            files.append(path)
        else:
            print(f"Warning: company report path not found or unsupported: {raw_path}")
    return files


def load_company_process_reports(paths: Iterable[Union[str, Path]]) -> str:
    """
    Load company-produced process documentation from arbitrary files.

    Supports:
    - Plain text / Markdown / CSV (.txt, .md, .markdown, .csv)
    - PDF files (.pdf) via PyMuPDF extraction

    Args:
        paths: Iterable of files or directories containing documentation.

    Returns:
        Aggregated string ready to be fed into LLM prompts.
    """
    candidate_files = _gather_candidate_files(paths)
    if not candidate_files:
        return NO_COMPANY_DOCS_MSG

    supported_text_ext = {".txt", ".md", ".markdown", ".csv"}
    aggregated_sections: List[str] = ["=== COMPANY PROCESS DOCUMENTATION ==="]

    for file_path in candidate_files:
        suffix = file_path.suffix.lower()
        if suffix in supported_text_ext:
            content = _read_text_file(file_path)
        elif suffix == ".pdf":
            content = _extract_text_from_local_pdf(file_path)
        else:
            print(f"Skipping unsupported company report format: {file_path.name}")
            continue

        if not content.strip():
            continue

        if len(content) > 20000:
            content = content[:20000] + "\n\n... [Truncated for length] ..."

        section = (
            f"\n---\nDocument: {file_path.name}\nLocation: {file_path}\n"
            f"Content:\n{content.strip()}\n"
        )
        aggregated_sections.append(section)

    if len(aggregated_sections) == 1:
        return "Company process documentation files were found but contained no readable text."

    return "\n".join(aggregated_sections) + "\n"
