import requests
import re
import os

try:
    import fitz  # PyMuPDF for PDF processing
except ImportError:
    try:
        import pymupdf as fitz  # Try alternative import
    except ImportError:
        print("Error: PyMuPDF not installed correctly. Try: pip install pymupdf")
import argparse
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Union

# Load environment variables
load_dotenv()

# Configure LLM API
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Default to Ollama, can be 'openai', 'anthropic', etc.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # API key for external providers
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/chat")  # Updated Ollama API endpoint
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")  # Default model


class ArticleSummarizer:
    def __init__(self, model_name: str = LLM_MODEL, api_url: str = LLM_API_URL,
                 provider: str = LLM_PROVIDER, api_key: str = LLM_API_KEY):
        self.model_name = model_name
        self.api_url = api_url
        self.provider = provider
        self.api_key = api_key

    def fetch_article(self, source: str) -> Tuple[str, str]:
        """
        Fetch article content from URL or PDF file
        Returns: Tuple of (title, content)
        """
        try:
            if os.path.isfile(source) and source.lower().endswith('.pdf'):
                return self.extract_from_pdf(source)
            elif source.startswith(('http://', 'https://')):
                return self.extract_from_url(source)
            else:
                raise ValueError("Source must be a valid URL or path to a PDF file")
        except Exception as e:
            print(f"Error fetching article: {e}")
            raise

    def extract_from_pdf(self, pdf_path: str) -> Tuple[str, str]:
        """Extract text from a PDF file"""
        doc = fitz.open(pdf_path)
        title = os.path.basename(pdf_path)
        content = ""

        # Extract title from PDF metadata if available
        pdf_title = doc.metadata.get("title")
        if pdf_title:
            title = pdf_title

        # Extract text from each page
        for page in doc:
            content += page.get_text()

        return title, content

    def extract_from_url(self, url: str) -> Tuple[str, str]:
        """Extract text from a URL (with special handling for arXiv)"""
        domain = urlparse(url).netloc

        # Special handling for arXiv
        if 'arxiv.org' in domain:
            return self.extract_from_arxiv(url)

        # General URL extraction
        try:
            response = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Try to extract title
            title_tag = soup.find('title')
            title = title_tag.get_text() if title_tag else "Unknown Title"

            # Try to extract main content (this is a simplified approach)
            # Real implementation would need to be more sophisticated
            article_content = ""

            # Look for common article containers
            article_tags = soup.find_all(['article', 'main', 'div'],
                                         class_=re.compile(r'(article|content|post|entry)'))

            if article_tags:
                for tag in article_tags:
                    # Remove scripts, styles, and comments
                    for unwanted in tag.find_all(['script', 'style', 'nav', 'header', 'footer']):
                        unwanted.decompose()
                    article_content += tag.get_text(separator='\n', strip=True)
            else:
                # Fallback to body content
                body = soup.find('body')
                if body:
                    for unwanted in body.find_all(['script', 'style', 'nav', 'header', 'footer']):
                        unwanted.decompose()
                    article_content = body.get_text(separator='\n', strip=True)
                else:
                    article_content = soup.get_text(separator='\n', strip=True)

            # If content is too short, it might not be properly extracted
            if len(article_content.split()) < 100:
                print(f"Warning: Extracted content seems short ({len(article_content.split())} words)")
                # Try an alternative approach - paragraphs
                paragraphs = soup.find_all('p')
                if paragraphs:
                    article_content = "\n\n".join(
                        [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])

            return title, article_content

        except Exception as e:
            print(f"Error extracting from URL {url}: {e}")
            raise

    def extract_from_arxiv(self, url: str) -> Tuple[str, str]:
        """Special handling for arXiv articles"""
        # Check if URL is for the PDF version
        if '/pdf/' in url:
            # This is a direct PDF URL, extract the ID
            arxiv_id_match = re.search(r'/pdf/(\d+\.\d+|[a-z\-]+\/\d+)', url)
            if arxiv_id_match:
                arxiv_id = arxiv_id_match.group(1)
            else:
                # Fallback to look for any numeric pattern in the URL
                arxiv_id_match = re.search(r'(\d+\.\d+|[a-z\-]+\/\d+)', url)
                if not arxiv_id_match:
                    raise ValueError(f"Could not extract arXiv ID from URL: {url}")
                arxiv_id = arxiv_id_match.group(1)

            # Use the PDF URL directly
            pdf_url = url
            if not pdf_url.endswith('.pdf'):
                pdf_url = f"{pdf_url}.pdf"
        else:
            # Extract arXiv ID from abstract URL
            arxiv_id_match = re.search(r'(\d+\.\d+|[a-z\-]+\/\d+)', url)
            if not arxiv_id_match:
                raise ValueError(f"Could not extract arXiv ID from URL: {url}")

            arxiv_id = arxiv_id_match.group(1)
            # Construct PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Now we have the PDF URL, download it
        try:
            print(f"Downloading PDF from: {pdf_url}")
            pdf_response = requests.get(
                pdf_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            )
            pdf_response.raise_for_status()

            # Save PDF temporarily
            temp_pdf = f"temp_{arxiv_id.replace('/', '_')}.pdf"
            with open(temp_pdf, 'wb') as f:
                f.write(pdf_response.content)

            # Extract content from PDF
            try:
                title, content = self.extract_from_pdf(temp_pdf)
                return title, content
            finally:
                # Clean up temporary file
                if os.path.exists(temp_pdf):
                    os.remove(temp_pdf)
        except Exception as e:
            print(f"Error downloading/processing PDF: {e}")

            # Fallback to abstract page if PDF download fails
            abstract_url = f"https://arxiv.org/abs/{arxiv_id}"
            print(f"Falling back to abstract page: {abstract_url}")

            response = requests.get(
                abstract_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title
            title_tag = soup.find('h1', class_='title')
            title = title_tag.get_text(strip=True).replace('Title:', '') if title_tag else f"arXiv: {arxiv_id}"

            # Extract abstract
            abstract_tag = soup.find('blockquote', class_='abstract')
            abstract = abstract_tag.get_text(strip=True).replace('Abstract:', '') if abstract_tag else ""

            # Extract authors
            authors_tag = soup.find('div', class_='authors')
            authors = authors_tag.get_text(strip=True).replace('Authors:', '') if authors_tag else ""

            content = f"Authors: {authors}\n\nAbstract: {abstract}"

            return title, content

    def preprocess_text(self, text: str, max_length: int = 8000) -> str:
        """Process and truncate text to prepare for the LLM"""
        # Clean up text: remove excessive whitespace, normalize newlines, etc.
        text = re.sub(r'\s+', ' ', text)

        # Truncate to max_length
        if len(text) > max_length:
            text = text[:max_length] + "... [truncated]"

        return text

    def generate_summary(self, title: str, content: str,
                         length: str = "medium",
                         format_type: str = "bullet") -> str:
        """
        Generate a summary using the LLM API

        Args:
            title: Article title
            content: Article content
            length: 'short', 'medium', or 'long'
            format_type: 'bullet', 'paragraph', or 'structured'

        Returns:
            Generated summary
        """
        # Prepare prompt based on requested length and format
        length_guide = {
            "short": "Provide a brief summary in about 3-5 sentences.",
            "medium": "Provide a moderately detailed summary with key points.",
            "long": "Provide a comprehensive summary with detailed analysis and all key points."
        }

        format_guide = {
            "bullet": "Format the summary as bullet points.",
            "paragraph": "Format the summary as paragraphs.",
            "structured": "Format the summary with sections for Introduction, Main Points, Methodology (if applicable), Results, and Conclusion."
        }

        prompt = f"""
        Summarize the following article titled "{title}".
        {length_guide.get(length, length_guide["medium"])}
        {format_guide.get(format_type, format_guide["bullet"])}
        Focus on the main ideas, key findings, and conclusions.

        Article content:
        {self.preprocess_text(content)}
        """

        try:
            # Choose API call format based on provider
            if self.provider.lower() == "ollama":
                # Updated Ollama API format (chat-based)
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "No summary generated.")

            elif self.provider.lower() == "openai":
                # OpenAI API format
                import openai
                openai.api_key = self.api_key

                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes articles."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content

            elif self.provider.lower() == "anthropic":
                # Anthropic API format
                headers = {
                    "x-api-key": self.api_key,
                    "content-type": "application/json"
                }

                response = requests.post(
                    "https://api.anthropic.com/v1/complete",
                    headers=headers,
                    json={
                        "model": self.model_name,
                        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                        "max_tokens_to_sample": 2000,
                        "temperature": 0.7
                    }
                )
                response.raise_for_status()
                return response.json().get("completion", "No summary generated.")

            else:
                # Generic API format as fallback
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json={
                        "model": self.model_name,
                        "prompt": prompt
                    }
                )
                response.raise_for_status()
                return response.json().get("text",
                                           response.json().get("response",
                                                               response.json().get("output", "No summary generated.")))

        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return f"Failed to generate summary: {str(e)}"

    def summarize(self, source: str, length: str = "medium",
                  format_type: str = "bullet") -> Dict[str, str]:
        """
        Main method to summarize an article from a URL or PDF

        Args:
            source: URL or path to PDF file
            length: 'short', 'medium', or 'long'
            format_type: 'bullet', 'paragraph', or 'structured'

        Returns:
            Dictionary with title and summary
        """
        try:
            title, content = self.fetch_article(source)
            summary = self.generate_summary(title, content, length, format_type)

            return {
                "title": title,
                "summary": summary,
                "source": source,
                "length": length,
                "format": format_type
            }

        except Exception as e:
            print(f"Error summarizing article: {e}")
            return {
                "title": "Error",
                "summary": f"Failed to summarize the article: {str(e)}",
                "source": source,
                "length": length,
                "format": format_type
            }


def main():
    parser = argparse.ArgumentParser(description='Summarize articles from URLs or PDFs')
    parser.add_argument('source', help='URL or path to PDF file')
    parser.add_argument('--model', default=MODEL_NAME, help=f'LLM model name (default: {MODEL_NAME})')
    parser.add_argument('--length', choices=['short', 'medium', 'long'], default='medium',
                        help='Summary length (default: medium)')
    parser.add_argument('--format', choices=['bullet', 'paragraph', 'structured'], default='bullet',
                        help='Summary format (default: bullet)')

    args = parser.parse_args()

    summarizer = ArticleSummarizer(model_name=args.model)
    result = summarizer.summarize(args.source, args.length, args.format)

    print(f"\n{'-' * 50}")
    print(f"Title: {result['title']}")
    print(f"{'-' * 50}")
    print(f"Summary ({result['length']}, {result['format']} format):")
    print(f"{'-' * 50}")
    print(result['summary'])
    print(f"{'-' * 50}")
    print(f"Source: {result['source']}")
    print("\n")


if __name__ == "__main__":
    main()