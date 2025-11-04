import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import asyncio
import re
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    """
    A web scraper for extracting content from web pages with rate limiting.

    Attributes:
        rate_limit: Minimum seconds between requests to the same domain
        last_request_time: Dictionary tracking last request time per domain
    """

    def __init__(self, rate_limit: int = 1) -> None:
        """
        Initialize the web scraper.

        Args:
            rate_limit: Minimum seconds between requests to the same domain (default: 1)
        """
        self.rate_limit: int = rate_limit
        self.last_request_time: Dict[str, float] = {}
        
    async def respect_rate_limit(self, domain: str) -> None:
        """
        Implement rate limiting per domain.

        Args:
            domain: The domain to apply rate limiting to
        """
        current_time = asyncio.get_event_loop().time()
        if domain in self.last_request_time:
            time_since_last_request = current_time - self.last_request_time[domain]
            if time_since_last_request < self.rate_limit:
                await asyncio.sleep(self.rate_limit - time_since_last_request)
        self.last_request_time[domain] = current_time
        
    async def scrape_page(self, url: str) -> Optional[Dict[str, str]]:
        """
        Scrape content from a single webpage.

        Args:
            url: The URL to scrape

        Returns:
            Dictionary containing extracted content or None if scraping failed
        """
        domain = urlparse(url).netloc
        await self.respect_rate_limit(domain)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0, follow_redirects=True)
                response.raise_for_status()
                return await self.extract_content(response.text, url)
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
            
    async def extract_content(self, html: str, url: str) -> Dict[str, str]:
        """
        Extract and clean content from HTML.

        Args:
            html: The HTML content to parse
            url: The source URL

        Returns:
            Dictionary containing extracted title, description, content, and metadata
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        # Get title
        title = soup.title.string if soup.title else ""
        
        # Find main content
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_=re.compile(r'content|main|article'))
        )
        
        # Extract paragraphs
        if main_content:
            paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        else:
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
        # Clean and join text
        text = ' '.join(p.get_text().strip() for p in paragraphs)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Extract metadata
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc else ""
        
        return {
            "url": url,
            "title": title,
            "description": description,
            "content": text[:5000],  # Limit content length
            "source": urlparse(url).netloc
        }
        
    async def scrape_multiple(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Scrape multiple pages concurrently.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of dictionaries containing extracted content from successful scrapes
        """
        tasks = [self.scrape_page(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
