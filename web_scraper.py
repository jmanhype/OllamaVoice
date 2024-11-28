import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import asyncio
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, rate_limit=1):
        self.rate_limit = rate_limit
        self.last_request_time = {}
        
    async def respect_rate_limit(self, domain):
        """Implement rate limiting per domain."""
        current_time = asyncio.get_event_loop().time()
        if domain in self.last_request_time:
            time_since_last_request = current_time - self.last_request_time[domain]
            if time_since_last_request < self.rate_limit:
                await asyncio.sleep(self.rate_limit - time_since_last_request)
        self.last_request_time[domain] = current_time
        
    async def scrape_page(self, url: str) -> dict:
        """Scrape content from a single webpage."""
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
            
    async def extract_content(self, html: str, url: str) -> dict:
        """Extract and clean content from HTML."""
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
        
    async def scrape_multiple(self, urls: list[str]) -> list[dict]:
        """Scrape multiple pages concurrently."""
        tasks = [self.scrape_page(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
