#!/usr/bin/env python3
"""
WebIntel - Competitive Website Audit CLI Tool
Performs automated competitive analysis of websites using AI-powered content summarization.
Now supports OpenAI, Hugging Face, and Ollama backends.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urljoin, urlparse

import requests
import tldextract
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify as md

# Optional imports for different AI providers
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai")

try:
    from transformers import pipeline
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Hugging Face not available. Install with: pip install transformers torch")

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file if present."""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value

# Load .env file at module level
load_env_file()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class WebIntelError(Exception):
    """Base exception for WebIntel operations."""
    pass


class WebIntelScraper:
    """Handles website scraping and content extraction."""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with appropriate headers."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        return session
    
    def fetch_page(self, url: str) -> Tuple[str, str]:
        """
        Fetch HTML content from URL with retries and error handling.
        
        Args:
            url: Target URL to scrape
            
        Returns:
            Tuple of (final_url, html_content)
            
        Raises:
            WebIntelError: If page cannot be fetched after retries
        """
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching {url} (attempt {attempt + 1}/{self.max_retries})")
                response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    raise WebIntelError(f"URL does not return HTML content: {content_type}")
                
                logger.info(f"Successfully fetched {response.url} ({len(response.content)} bytes)")
                return response.url, response.text
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1}")
            except requests.exceptions.HTTPError as e:
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        raise WebIntelError(f"Failed to fetch {url} after {self.max_retries} attempts")


class ContentProcessor:
    """Handles HTML parsing and content cleaning."""
    
    # Tags to remove completely
    REMOVE_TAGS = [
        'script', 'style', 'nav', 'footer', 'header', 'aside', 
        'iframe', 'noscript', 'meta', 'link', 'form', 'input', 
        'button', 'select', 'textarea', 'img', 'svg', 'canvas'
    ]
    
    # Tags to keep but extract text from
    CONTENT_TAGS = [
        'main', 'article', 'section', 'div', 'p', 'h1', 'h2', 'h3', 
        'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote', 'span'
    ]
    
    def __init__(self, char_limit: int = 6000):  # Reduced for Ollama
        self.char_limit = char_limit
    
    def clean_html(self, html: str) -> str:
        """
        Clean HTML content by removing unwanted elements and extracting text.
        
        Args:
            html: Raw HTML content
            
        Returns:
            Cleaned text content
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Remove unwanted tags
            for tag_name in self.REMOVE_TAGS:
                for tag in soup.find_all(tag_name):
                    tag.decompose()
            
            # Try to find main content area first
            main_content = self._extract_main_content(soup)
            
            if main_content:
                # Convert to markdown for better structure
                markdown_text = md(str(main_content), heading_style='ATX')
            else:
                # Fallback to body content
                body = soup.find('body')
                if body:
                    markdown_text = md(str(body), heading_style='ATX')
                else:
                    markdown_text = md(str(soup), heading_style='ATX')
            
            # Clean up the text
            cleaned_text = self._clean_text(markdown_text)
            
            # Truncate if necessary
            if len(cleaned_text) > self.char_limit:
                cleaned_text = cleaned_text[:self.char_limit] + "..."
                logger.warning(f"Content truncated to {self.char_limit} characters")
            
            logger.info(f"Extracted {len(cleaned_text)} characters of content")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            raise WebIntelError(f"Failed to clean HTML content: {e}")
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Extract main content using common patterns."""
        # Try common main content selectors
        selectors = [
            'main',
            '[role="main"]',
            '#main',
            '.main',
            '#content',
            '.content',
            'article',
            '.container .row',
            '.page-content'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element and len(element.get_text(strip=True)) > 200:
                logger.info(f"Found main content using selector: {selector}")
                return element
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove empty markdown elements
        text = re.sub(r'\n#+\s*\n', '\n', text)
        text = re.sub(r'\n\*\s*\n', '\n', text)
        
        # Clean up common artifacts
        text = re.sub(r'Skip to (?:main )?content', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Toggle navigation', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Menu', '', text, flags=re.IGNORECASE)
        
        return text.strip()


class BaseAIAnalyzer:
    """Base class for AI analyzers."""
    
    def analyze_content(self, content: str, url: str) -> str:
        """Analyze website content for competitive intelligence."""
        raise NotImplementedError


class OpenAIAnalyzer(BaseAIAnalyzer):
    """OpenAI-powered content analysis."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        if not OPENAI_AVAILABLE:
            raise WebIntelError("OpenAI not available. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model
        
        if not (api_key or os.getenv('OPENAI_API_KEY')):
            raise WebIntelError("OpenAI API key required")
    
    def analyze_content(self, content: str, url: str) -> str:
        prompt = self._create_analysis_prompt(content, url)
        
        try:
            logger.info("Sending content to OpenAI for analysis...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a competitive intelligence analyst specializing in website analysis. Provide detailed, actionable insights in clean Markdown format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content.strip()
            logger.info(f"Received analysis ({len(analysis)} characters)")
            return analysis
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise WebIntelError(f"Failed to analyze content: {e}")
    
    def _create_analysis_prompt(self, content: str, url: str) -> str:
        """Create a detailed prompt for competitive analysis."""
        return f"""
Perform a comprehensive competitive analysis of the website: {url}

Website Content:
{content}

Please analyze and provide insights in the following areas:

## Company Overview
- Brief description of the company and its primary business
- Target audience and market positioning

## Product/Service Analysis
- Core products or services offered
- Key features and capabilities
- Unique selling propositions

## Pricing Strategy
- Pricing models mentioned
- Value propositions tied to pricing
- Competitive positioning on price

## Design & User Experience
- Overall design approach and aesthetic
- User experience highlights
- Navigation and information architecture

## Marketing & Messaging
- Key marketing messages and value propositions
- Brand positioning and tone
- Content strategy insights

## Competitive Advantages
- Stated or implied competitive advantages
- Innovation areas or unique approaches
- Market differentiation factors

## Areas for Analysis
- Potential weaknesses or gaps
- Opportunities for competitive response
- Questions for further investigation

Format your response in clean Markdown with clear headings and bullet points where appropriate. Focus on actionable insights that would be valuable for competitive intelligence.
"""


class HuggingFaceAnalyzer(BaseAIAnalyzer):
    """Hugging Face transformers-powered content analysis."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-large"):
        if not HUGGINGFACE_AVAILABLE:
            raise WebIntelError("Hugging Face transformers not available. Install with: pip install transformers torch")
        
        self.model_name = model_name
        logger.info(f"Loading Hugging Face model: {model_name}")
        
        # Use text-generation pipeline for analysis
        try:
            self.generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Hugging Face model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load GPU model, falling back to CPU: {e}")
            self.generator = pipeline(
                "text-generation",
                model="gpt2",
                device=-1
            )
    
    def analyze_content(self, content: str, url: str) -> str:
        try:
            logger.info("Analyzing content with Hugging Face model...")
            
            # Create analysis prompt
            prompt = f"Analyze this website content for competitive intelligence: {content[:1000]}"  # Truncate for model limits
            
            # Generate analysis
            result = self.generator(
                prompt,
                max_length=1000,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            analysis = result[0]['generated_text']
            
            # Post-process to create structured output
            structured_analysis = self._create_structured_analysis(content, url, analysis)
            
            logger.info(f"Generated analysis ({len(structured_analysis)} characters)")
            return structured_analysis
            
        except Exception as e:
            logger.error(f"Hugging Face analysis error: {e}")
            raise WebIntelError(f"Failed to analyze content with Hugging Face: {e}")
    
    def _create_structured_analysis(self, content: str, url: str, raw_analysis: str) -> str:
        """Create a structured analysis from the model output."""
        # This is a simplified analysis since local models may not be as sophisticated
        lines = content.split('\n')[:20]  # First 20 lines
        
        analysis = f"""# Competitive Analysis: {url}

## Content Overview
The website contains the following key sections and information:

"""
        
        # Extract headings and key content
        for line in lines:
            if line.strip().startswith('#') or len(line.strip()) > 30:
                analysis += f"- {line.strip()}\n"
        
        analysis += f"""

## AI-Generated Insights
{raw_analysis}

## Key Observations
- Website appears to focus on the primary domain activities
- Content structure suggests organized information architecture
- Further detailed analysis would require domain-specific expertise

*Note: This analysis was generated using a local Hugging Face model and may be less detailed than cloud-based alternatives.*
"""
        
        return analysis


class OllamaAnalyzer(BaseAIAnalyzer):
    """Ollama-powered content analysis."""
    
    def __init__(self, model_name: str = "llama2", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host.rstrip('/')
        
        # Test Ollama connection
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logger.info(f"Connected to Ollama. Available models: {model_names}")
                
                if model_name not in model_names:
                    logger.warning(f"Model {model_name} not found. Available: {model_names}")
                    if model_names:
                        self.model_name = model_names[0]
                        logger.info(f"Using {self.model_name} instead")
            else:
                raise WebIntelError(f"Ollama API returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise WebIntelError(f"Cannot connect to Ollama at {self.host}. Is Ollama running? Error: {e}")
    
    def analyze_content(self, content: str, url: str) -> str:
        prompt = self._create_analysis_prompt(content, url)
        
        try:
            logger.info(f"Sending content to Ollama ({self.model_name}) for analysis...")
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=300  # Increased to 5 minutes
            )
            response.raise_for_status()
            
            result = response.json()
            analysis = result.get('response', '').strip()
            
            if not analysis:
                raise WebIntelError("Empty response from Ollama")
            
            logger.info(f"Received analysis ({len(analysis)} characters)")
            return analysis
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise WebIntelError(f"Failed to analyze content with Ollama: {e}")
        except Exception as e:
            logger.error(f"Ollama analysis error: {e}")
            raise WebIntelError(f"Failed to analyze content: {e}")
    
    def _create_analysis_prompt(self, content: str, url: str) -> str:
        """Create a detailed prompt for competitive analysis."""
        return f"""You are a competitive intelligence analyst. Analyze the following website content and provide detailed insights.

Website URL: {url}

Content:
{content[:4000]}  # Reduced content limit for Ollama performance

Provide a concise analysis covering:

1. Company Overview: What does this company do?
2. Products/Services: What do they offer?
3. Target Market: Who are their customers?
4. Value Proposition: What makes them unique?
5. Competitive Advantages: What are their strengths?

Keep your response focused and under 1000 words.

Analysis:"""


class WebIntel:
    """Main WebIntel application orchestrator."""
    
    def __init__(self, provider: str = "openai", **kwargs):
        self.scraper = WebIntelScraper()
        self.processor = ContentProcessor()
        
        # Initialize the appropriate AI analyzer
        if provider.lower() == "openai":
            self.analyzer = OpenAIAnalyzer(**kwargs)
        elif provider.lower() == "huggingface":
            self.analyzer = HuggingFaceAnalyzer(**kwargs)
        elif provider.lower() == "ollama":
            self.analyzer = OllamaAnalyzer(**kwargs)
        else:
            raise WebIntelError(f"Unsupported AI provider: {provider}")
        
        self.provider = provider.lower()
        logger.info(f"Initialized WebIntel with {self.provider} provider")
    
    def audit_website(self, url: str, output_dir: str = ".") -> str:
        """
        Perform complete website audit and generate report.
        
        Args:
            url: Target URL to audit
            output_dir: Directory to save the audit report
            
        Returns:
            Path to the generated audit report
        """
        try:
            # Extract domain for filename
            domain = self._extract_domain(url)
            logger.info(f"Starting audit for domain: {domain}")
            
            # Fetch website content
            final_url, html_content = self.scraper.fetch_page(url)
            
            # Process and clean content
            cleaned_content = self.processor.clean_html(html_content)
            
            if not cleaned_content.strip():
                raise WebIntelError("No meaningful content could be extracted from the page")
            
            # Generate AI analysis
            analysis = self.analyzer.analyze_content(cleaned_content, final_url)
            
            # Create comprehensive report
            report = self._create_report(domain, final_url, analysis)
            
            # Save report
            output_path = self._save_report(report, domain, output_dir)
            
            logger.info(f"Audit completed successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            raise WebIntelError(f"Website audit failed: {e}")
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain name for filename."""
        try:
            extracted = tldextract.extract(url)
            domain = extracted.domain
            if not domain:
                # Fallback to parsing
                parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'https://{url}')
                domain = parsed.netloc.replace('www.', '').split('.')[0]
            return domain.lower()
        except Exception:
            return "unknown_site"
    
    def _create_report(self, domain: str, url: str, analysis: str) -> str:
        """Create formatted audit report."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        
        report = f"""# WebIntel Competitive Audit: {domain.title()}

**Generated:** {timestamp}  
**Target URL:** {url}  
**AI Provider:** {self.provider.title()}  
**Tool:** WebIntel v2.0

---

{analysis}

---

*This report was generated automatically by WebIntel using {self.provider.title()} AI analysis. For questions or additional analysis, please review the methodology and consider manual verification of key findings.*
"""
        return report
    
    def _save_report(self, report: str, domain: str, output_dir: str) -> str:
        """Save report to markdown file."""
        output_path = Path(output_dir) / f"{domain}_audit_{self.provider}.md"
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            return str(output_path)
        except Exception as e:
            raise WebIntelError(f"Failed to save report: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WebIntel - Competitive Website Audit Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
AI Provider Options:
  openai        OpenAI GPT models (requires API key)
  huggingface   Local Hugging Face models (no API key needed)
  ollama        Local Ollama models (requires Ollama running)

Examples:
  python webintel.py --url https://stripe.com --provider ollama
  python webintel.py --url shopify.com --provider huggingface
  python webintel.py --url https://openai.com --provider openai --api-key your-key

Environment Variables:
  OPENAI_API_KEY    OpenAI API key (for OpenAI provider)
        """
    )
    
    parser.add_argument(
        '--url', 
        required=True, 
        help='Target website URL to audit'
    )
    
    parser.add_argument(
        '--provider',
        choices=['openai', 'huggingface', 'ollama'],
        default='ollama',
        help='AI provider to use (default: ollama)'
    )
    
    parser.add_argument(
        '--output', 
        default='.', 
        help='Output directory for audit reports (default: current directory)'
    )
    
    parser.add_argument(
        '--api-key', 
        help='OpenAI API key (only needed for OpenAI provider)'
    )
    
    parser.add_argument(
        '--model', 
        help='Model name (e.g., gpt-4o-mini for OpenAI, llama2 for Ollama)'
    )
    
    parser.add_argument(
        '--ollama-host',
        default='http://localhost:11434',
        help='Ollama host URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='WebIntel v2.0'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Prepare provider arguments
    provider_kwargs = {}
    
    if args.provider == 'openai':
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OpenAI API key required for OpenAI provider. Set OPENAI_API_KEY environment variable or use --api-key")
            sys.exit(1)
        provider_kwargs['api_key'] = api_key
        if args.model:
            provider_kwargs['model'] = args.model
    
    elif args.provider == 'huggingface':
        if args.model:
            provider_kwargs['model_name'] = args.model
    
    elif args.provider == 'ollama':
        if args.model:
            provider_kwargs['model_name'] = args.model
        provider_kwargs['host'] = args.ollama_host
    
    try:
        # Initialize WebIntel
        webintel = WebIntel(provider=args.provider, **provider_kwargs)
        
        # Perform audit
        report_path = webintel.audit_website(args.url, args.output)
        
        print(f"\n✅ Audit completed successfully!")
        print(f"🤖 AI Provider: {args.provider.title()}")
        print(f"📄 Report saved to: {report_path}")
        print(f"\nTo view the report:")
        print(f"  cat {report_path}")
        
    except WebIntelError as e:
        logger.error(f"WebIntel error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Audit cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()