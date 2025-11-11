"""
Biotech Clinical Trial Analysis System using Google Gemini
Analyzes biotech research and company process documentation to predict clinical trial success
"""
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from .tools import (
    search_and_parse_pubmed_papers,
    analyze_clinical_trial_phases,
    analyze_drug_mechanism,
    load_company_process_reports,
    NO_COMPANY_DOCS_MSG,
)

load_dotenv()


class BiotechAnalysisSystem:
    """
    Analyzes biotech drugs and predicts clinical trial success using Google Gemini.
    """
    
    def __init__(self):
        """Initialize the system with required environment variables"""
        self.validate_env_vars()
        self.setup_gemini()
    
    def validate_env_vars(self):
        """Validate that required API keys are set"""
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY is required. Get your free API key at: "
                "https://aistudio.google.com/app/apikey"
            )
        
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        print(f"Using Google Gemini API (Model: {self.model_name})")
        
        # Check for LandingAI API key (optional)
        if os.getenv("VISION_AGENT_API_KEY"):
            print("LandingAI API key found - will use advanced PDF parsing")
        else:
            print("LandingAI API key not found - will use basic text extraction for PDFs")
    
    def setup_gemini(self):
        """Configure Google Gemini"""
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(self.model_name)
    
    def infer_metadata_from_gemini(
        self,
        drug_name: str,
        company: str,
        existing_indication: str | None = None,
        existing_phase: str | None = None,
        existing_ticker: str | None = None,
    ) -> dict[str, str | None]:
        """
        Use Gemini to infer missing metadata such as indication, current phase, ticker, compound aliases, and company use.
        """
        prompt = (
            "You are a biotech data assistant. Infer any missing metadata for the given drug.\n"
            "Return ONLY a JSON object with keys 'ticker', 'indication', 'current_phase', 'compound_name', and 'company_use'.\n"
            "The 'company_use' field should describe what the sponsoring company is developing the drug for (disease area/indication) in plain text.\n"
            "If you cannot determine a value, set it to null.\n"
            "Prefer the most recent or widely accepted information.\n"
            f"Company: {company}\n"
            f"Drug: {drug_name}\n"
            f"Known indication: {existing_indication or 'None provided'}\n"
            f"Known current phase: {existing_phase or 'None provided'}\n"
            f"Known ticker: {existing_ticker or 'None provided'}\n"
            "If there is a development code, research code, or alternative compound identifier (e.g., MK-6070), include it in 'compound_name'."
        )

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text or ""
        except Exception as exc:
            print(f"  Warning: Gemini metadata inference failed: {exc}")
            return {}

        json_start = raw_text.find("{")
        json_end = raw_text.rfind("}")
        if json_start == -1 or json_end == -1 or json_end <= json_start:
            print("  Warning: Gemini metadata response did not contain JSON.")
            return {}

        json_candidate = raw_text[json_start : json_end + 1]
        try:
            parsed = json.loads(json_candidate)
        except json.JSONDecodeError as exc:
            print(f"  Warning: Failed to parse Gemini metadata JSON: {exc}")
            return {}

        if not isinstance(parsed, dict):
            print("  Warning: Gemini metadata JSON was not an object.")
            return {}

        def normalize(value: object) -> str | None:
            if value is None:
                return None
            cleaned = str(value).strip()
            if not cleaned:
                return None
            if cleaned.lower() in {"unknown", "not specified", "n/a", "na"}:
                return None
            return cleaned

        return {
            "ticker": normalize(parsed.get("ticker")),
            "indication": normalize(parsed.get("indication")),
            "current_phase": normalize(parsed.get("current_phase")),
            "compound_name": normalize(parsed.get("compound_name")),
            "company_use": normalize(parsed.get("company_use")),
        }
    
    def answer_question_about_report(
        self,
        report: dict,
        question: str,
    ) -> str:
        """
        Use Gemini to answer a user question about a previously generated report.
        """
        if not question.strip():
            return "Please provide a question to analyze."

        context_sections = [
            f"Drug: {report.get('drug_name', 'Unknown')}",
            f"Company: {report.get('company', 'Unknown')}",
            f"Ticker: {report.get('ticker') or 'Not provided'}",
            f"Compound Code: {report.get('compound_name') or 'Not provided'}",
            f"Company Use: {report.get('company_use') or 'Not specified'}",
            f"Indication: {report.get('indication') or 'Not specified'}",
            f"Current Phase: {report.get('phase') or 'Not specified'}",
            "\n=== EXECUTIVE SUMMARY ===",
            report.get("prediction", "").strip(),
            "\n=== DETAILED ANALYSIS ===",
            report.get("analysis", "").strip(),
            "\n=== RESEARCH NOTES ===",
            report.get("research", "").strip(),
        ]

        if report.get("company_docs_available"):
            context_sections.append("\n=== COMPANY DOCUMENTATION ===")
            context_sections.append(report.get("company_reports", "").strip())

        context = "\n".join(section for section in context_sections if section)

        prompt = (
            "You are an expert biotech analyst assistant. Answer the user's question "
            "using ONLY the context provided. If the answer cannot be found in the context, "
            "state that it is not available.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question.strip()}\n\n"
            "Answer with clear, concise sentences, citing relevant numbers from the context when available."
        )

        try:
            response = self.model.generate_content(prompt)
            answer = (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: Gemini report query failed: {exc}")
            return "Unable to generate an answer at this time."

        return answer or "No answer was generated from the provided report."
    
    def _strip_markdown(self, text: str) -> str:
        """Remove common Markdown syntax while preserving readable content."""
        if not text:
            return ""
        
        cleaned = text.strip()
        cleaned = cleaned.replace("\r\n", "\n")
        
        # Replace list markers first to avoid conflicts with emphasis removal
        cleaned = re.sub(r"^\s*[-*+]\s+", "- ", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", cleaned)
        cleaned = re.sub(r"`{1,3}([^`]+)`{1,3}", r"\1", cleaned)
        cleaned = re.sub(r"(\*\*|__)(.*?)\1", r"\2", cleaned)
        cleaned = re.sub(r"(\*|_)(.*?)\1", r"\2", cleaned)
        cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^\s{0,3}>\s?", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        
        return cleaned.strip()
    
    def _prepare_paragraph(self, text: str) -> str:
        """Clean Markdown and escape HTML-sensitive characters for PDF output."""
        stripped = self._strip_markdown(text)
        if not stripped:
            return ""
        
        escaped = (
            stripped
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return escaped.replace("\n", "<br/>")
    
    def research_analysis(
        self,
        drug_name: str,
        company: str,
        indication: str,
        search_term: str,
        compound_name: str | None = None,
        company_use: str | None = None,
        company_reports_text: str | None = None
    ) -> str:
        """
        Conduct research analysis on the drug using PubMed and ClinicalTrials.gov.
        Now includes full PDF parsing using LandingAI when available.
        """
        print("\n[1/2] Research Agent: Analyzing scientific literature and company documentation...")

        indication_value = (indication or "").strip()
        primary_term = (search_term or drug_name).strip() or drug_name
        indication_for_prompt = indication_value or "Not specified"
        indication_query_fragment = f" {indication_value}" if indication_value else ""
        
        # Gather data from tools - now with PDF parsing
        pubmed_results = search_and_parse_pubmed_papers(
            query=f"{primary_term}{indication_query_fragment} clinical trial",
            drug_name=primary_term,
            indication=indication_for_prompt,
            max_results=5,
            max_pdfs=2  # Limit to 2 PDFs to control processing time
        )
        trial_data = analyze_clinical_trial_phases(drug_name)
        mechanism_info = analyze_drug_mechanism(drug_name, indication_for_prompt)

        company_reports_section = company_reports_text or NO_COMPANY_DOCS_MSG
        
        # Create prompt for Gemini
        alias_text = (
            compound_name.strip() if compound_name and compound_name.strip() else "Not supplied"
        )

        company_use_text = (
            company_use.strip() if company_use and company_use.strip() else "Not specified"
        )

        prompt = f"""You are a biotech research analyst. Analyze the following information about {drug_name} (Compound code: {alias_text}) for {indication_for_prompt}.
Company use focus: {company_use_text}

PubMed Research (includes full-text papers when available):
{pubmed_results}

Clinical Trials Data:
{trial_data}

Drug Mechanism:
{mechanism_info}

Company Process Documentation:
{company_reports_section}

Provide a comprehensive research analysis covering:
1. Summary of preclinical and clinical data from the full papers
2. Trial design and methodology assessment with specific details from the studies
3. Key efficacy and safety findings with exact statistical values (p-values, confidence intervals, response rates)
4. Patient demographics and enrollment details
5. Competitive landscape comparison
6. Identified risk factors or concerns from safety data
7. Quality assessment of the studies
8. Comparison of company-reported processes versus independent trial evidence, highlighting consistencies, discrepancies, and data gaps

Be specific and evidence-based. Cite exact numbers and statistics from the research papers."""
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def predict_trial_success(
        self,
        drug_name: str,
        company: str,
        indication: str,
        current_phase: str,
        research_findings: str,
        company_reports_text: str
    ) -> str:
        """
        Synthesize all information and predict clinical trial success
        """
        print("\n[2/2] Clinical Trials Analyst: Predicting trial success...")
        
        prompt = f"""You are a clinical trials success predictor with 20 years of experience in drug development.

Based on the following comprehensive analysis, predict whether {drug_name} will successfully pass its current clinical trial phase ({current_phase}) for {indication}.

RESEARCH FINDINGS:
{research_findings}

COMPANY PROCESS DOCUMENTATION:
{company_reports_text}

Provide a comprehensive prediction report with:
1. EXECUTIVE SUMMARY with probability estimate (e.g., 65% chance of success)
2. Detailed analysis of success factors
3. Risk factor identification and assessment
4. Scenario analysis (best case, base case, worst case)
5. Regulatory pathway assessment
6. Timeline and milestone predictions
7. Cross-check of company-reported processes versus observed trial outcomes
8. Key assumptions, confidence level in prediction, and any recommended remediation steps for process gaps

Use specific percentages and evidence-based reasoning. Be honest about uncertainties."""
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def analyze_drug(
        self,
        drug_name: str,
        company: str,
        ticker: str | None,
        indication: str | None = None,
        current_phase: str | None = None,
        company_reports: list[str] | None = None,
        compound_name: str | None = None,
        company_use: str | None = None,
    ) -> dict:
        """
        Analyze a biotech drug and predict clinical trial success.
        
        Args:
            drug_name: Name of the drug being analyzed
            company: Company developing the drug
            ticker: Optional stock ticker symbol
            indication: Medical indication/disease target
            current_phase: Current clinical trial phase
            company_reports: Optional list of file or directory paths containing company process documentation
        
        Returns:
            Dictionary containing analysis results
        """
        indication_value = (indication or "").strip() or "Not specified"
        ticker_value = (ticker or "").strip()

        print(f"\n{'='*80}")
        search_name = (compound_name or drug_name).strip()

        print(f"Starting Analysis for {drug_name} ({company})")
        if compound_name and compound_name.strip():
            print(f"Compound Code / Alias: {compound_name.strip()}")
        if company_use and company_use.strip():
            print(f"Company is pursuing use case: {company_use.strip()}")
        phase_value = (current_phase or "").strip() or "Not specified"

        print(f"Indication: {indication_value}")
        print(f"Current Phase: {phase_value}")
        print(f"{'='*80}")

        company_reports_data = load_company_process_reports(company_reports or [])
        company_reports_clean = company_reports_data.strip() if company_reports_data else ""
        has_company_docs = bool(
            company_reports_clean
            and company_reports_clean != NO_COMPANY_DOCS_MSG
        )
        company_reports_for_report = company_reports_data if has_company_docs else ""
        
        # Run analyses sequentially
        research_findings = self.research_analysis(
            drug_name,
            company,
            indication_value,
            search_name,
            compound_name=compound_name,
            company_use=company_use,
            company_reports_text=company_reports_data
        )
        prediction = self.predict_trial_success(
            drug_name,
            company,
            indication_value,
            phase_value,
            research_findings,
            company_reports_data
        )
        
        company_section_text = ""
        if has_company_docs:
            company_section_text = f"""
{'='*80}
COMPANY PROCESS DOCUMENTATION
{'='*80}

{company_reports_data}

"""
        
        # Combine all results
        company_line = (
            f"Company: {company} ({ticker_value})"
            if ticker_value
            else f"Company: {company}"
        )

        compound_line = (
            f"Compound Code: {compound_name.strip()}"
            if compound_name and compound_name.strip()
            else "Compound Code: Not provided"
        )
        company_use_line = (
            f"Company Use Focus: {company_use.strip()}"
            if company_use and company_use.strip()
            else "Company Use Focus: Not specified"
        )

        full_analysis = f"""
{'='*80}
BIOTECH CLINICAL TRIAL ANALYSIS REPORT
{'='*80}

Drug: {drug_name}
{company_line}
{compound_line}
{company_use_line}
Indication: {indication_value}
Current Phase: {phase_value}

{'='*80}
RESEARCH ANALYSIS
{'='*80}

{research_findings}

{company_section_text}{'='*80}
CLINICAL TRIAL SUCCESS PREDICTION
{'='*80}

{prediction}

{'='*80}
END OF REPORT
{'='*80}
"""
        
        print(f"\n{'='*80}")
        print("Analysis Complete!")
        print(f"{'='*80}\n")
        
        return {
            "drug_name": drug_name,
            "company": company,
            "ticker": ticker_value,
            "indication": indication_value,
            "phase": phase_value,
            "analysis": full_analysis,
            "research": research_findings,
            "company_reports": company_reports_for_report,
            "company_docs_available": has_company_docs,
            "prediction": prediction,
            "compound_name": compound_name or "",
            "company_use": company_use or "",
        }
    
    def save_report(self, results: dict, output_path: str = "reports"):
        """
        Save analysis results to a PDF file.
        
        Args:
            results: Analysis results dictionary
            output_path: Directory to save report (default: reports/)
        """
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        filename = f"{results['drug_name'].replace(' ', '_')}_{results['company'].replace(' ', '_')}_analysis.pdf"
        filepath = output_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='darkblue',
            spaceAfter=30,
            alignment=TA_CENTER,
            bold=True
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor='darkblue',
            spaceAfter=12,
            spaceBefore=12,
            bold=True
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor='navy',
            spaceAfter=8,
            spaceBefore=8,
            bold=True
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            spaceAfter=12,
            alignment=TA_LEFT
        )
        
        # Add title
        elements.append(Paragraph("BIOTECH CLINICAL TRIAL ANALYSIS REPORT", title_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Add drug information
        ticker_value = (results.get('ticker') or "").strip()
        compound_name = (results.get('compound_name') or "").strip()
        company_display = (
            f"{results['company']} ({ticker_value})"
            if ticker_value
            else results['company']
        )
        indication_display = (results.get('indication') or "").strip() or "Not specified"
        phase_display = (results.get('phase') or "").strip() or "Not specified"
        compound_display = compound_name or "Not provided"
        company_use_display = (results.get('company_use') or "").strip() or "Not specified"

        info_text = f"""
        <b>Drug:</b> {results['drug_name']}<br/>
        <b>Company:</b> {company_display}<br/>
        <b>Compound Code:</b> {compound_display}<br/>
        <b>Company Use:</b> {company_use_display}<br/>
        <b>Indication:</b> {indication_display}<br/>
        <b>Current Phase:</b> {phase_display}
        """
        elements.append(Paragraph(info_text, body_style))
        elements.append(Spacer(1, 0.5*inch))
        
        # Add research analysis section
        elements.append(Paragraph("RESEARCH ANALYSIS", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Split research text into paragraphs
        research_paragraphs = results['research'].split('\n\n')
        for para in research_paragraphs:
            if para.strip():
                clean_para = self._prepare_paragraph(para)
                if not clean_para:
                    continue
                elements.append(Paragraph(clean_para, body_style))
        
        elements.append(PageBreak())
        
        company_reports_content = results.get('company_reports', "")
        if company_reports_content and company_reports_content.strip():
            elements.append(Paragraph("COMPANY PROCESS DOCUMENTATION", heading_style))
            elements.append(Spacer(1, 0.2*inch))
            
            company_paragraphs = company_reports_content.split('\n\n')
            for para in company_paragraphs:
                if para.strip():
                    clean_para = self._prepare_paragraph(para)
                    if not clean_para:
                        continue
                    elements.append(Paragraph(clean_para, body_style))
            
            elements.append(PageBreak())
        
        # Add prediction section
        elements.append(Paragraph("CLINICAL TRIAL SUCCESS PREDICTION", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        prediction_paragraphs = results['prediction'].split('\n\n')
        for para in prediction_paragraphs:
            if para.strip():
                clean_para = self._prepare_paragraph(para)
                if not clean_para:
                    continue
                elements.append(Paragraph(clean_para, body_style))
        
        # Build PDF
        doc.build(elements)
        
        print(f"\nReport saved to: {filepath}")
        return filepath


def main():
    """
    Main function - Example usage of the Biotech Analysis System
    """
    system = BiotechAnalysisSystem()
    
    print("=" * 80)
    print("BIOTECH CLINICAL TRIAL ANALYSIS SYSTEM")
    print("Powered by Google Gemini")
    print("=" * 80)
    
    # Example analysis - customize these parameters for any biotech company/drug
    results = system.analyze_drug(
        drug_name="Pembrolizumab",
        company="Merck",
        ticker="MRK",
        indication="Non-small cell lung cancer",
        current_phase="Phase 3"
    )
    
    # Save the report
    system.save_report(results)
    
    print("\nAnalysis complete! Check the reports/ directory for detailed results.\n")
    
    # Print summary
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    print(f"Drug: {results['drug_name']}")
    print(f"Company: {results['company']}")
    print(f"Status: Analysis completed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
