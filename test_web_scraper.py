"""
Unit tests for the WebScraper class.
"""
import pytest
import asyncio
from web_scraper import WebScraper
from unittest.mock import Mock, patch, AsyncMock


class TestWebScraper:
    """Test suite for WebScraper functionality."""

    def test_init(self):
        """Test WebScraper initialization."""
        scraper = WebScraper(rate_limit=2)
        assert scraper.rate_limit == 2
        assert scraper.last_request_time == {}

    def test_init_default_rate_limit(self):
        """Test WebScraper initialization with default rate limit."""
        scraper = WebScraper()
        assert scraper.rate_limit == 1

    @pytest.mark.asyncio
    async def test_respect_rate_limit_first_request(self):
        """Test rate limiting on first request to a domain."""
        scraper = WebScraper(rate_limit=1)
        domain = "example.com"

        # First request should not wait
        start_time = asyncio.get_event_loop().time()
        await scraper.respect_rate_limit(domain)
        elapsed = asyncio.get_event_loop().time() - start_time

        assert elapsed < 0.1  # Should be nearly instant
        assert domain in scraper.last_request_time

    @pytest.mark.asyncio
    async def test_respect_rate_limit_subsequent_request(self):
        """Test rate limiting enforces delay on subsequent requests."""
        scraper = WebScraper(rate_limit=0.5)
        domain = "example.com"

        # First request
        await scraper.respect_rate_limit(domain)

        # Second request should wait
        start_time = asyncio.get_event_loop().time()
        await scraper.respect_rate_limit(domain)
        elapsed = asyncio.get_event_loop().time() - start_time

        assert elapsed >= 0.4  # Should wait at least ~0.5 seconds

    @pytest.mark.asyncio
    async def test_extract_content_basic(self):
        """Test basic content extraction from HTML."""
        scraper = WebScraper()
        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Title</h1>
                <p>This is a test paragraph.</p>
                <p>Another paragraph here.</p>
            </body>
        </html>
        """
        url = "https://example.com/test"

        result = await scraper.extract_content(html, url)

        assert result["url"] == url
        assert result["title"] == "Test Page"
        assert "Main Title" in result["content"]
        assert "test paragraph" in result["content"]
        assert result["source"] == "example.com"

    @pytest.mark.asyncio
    async def test_extract_content_removes_scripts(self):
        """Test that script tags are removed during content extraction."""
        scraper = WebScraper()
        html = """
        <html>
            <body>
                <p>Visible content</p>
                <script>alert('should not appear');</script>
                <style>.hidden { display: none; }</style>
            </body>
        </html>
        """
        url = "https://example.com/test"

        result = await scraper.extract_content(html, url)

        assert "Visible content" in result["content"]
        assert "alert" not in result["content"]
        assert "display: none" not in result["content"]

    @pytest.mark.asyncio
    async def test_extract_content_with_meta_description(self):
        """Test extraction of meta description."""
        scraper = WebScraper()
        html = """
        <html>
            <head>
                <meta name="description" content="This is a test description">
            </head>
            <body><p>Content</p></body>
        </html>
        """
        url = "https://example.com/test"

        result = await scraper.extract_content(html, url)

        assert result["description"] == "This is a test description"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_scrape_page_success(self, mock_client):
        """Test successful page scraping."""
        scraper = WebScraper()

        # Mock response
        mock_response = Mock()
        mock_response.text = "<html><body><p>Test content</p></body></html>"
        mock_response.raise_for_status = Mock()

        mock_client_instance = Mock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client.return_value = mock_client_instance

        result = await scraper.scrape_page("https://example.com/test")

        assert result is not None
        assert result["url"] == "https://example.com/test"
        assert "Test content" in result["content"]

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_scrape_page_error(self, mock_client):
        """Test error handling during page scraping."""
        scraper = WebScraper()

        # Mock client to raise an exception
        mock_client_instance = Mock()
        mock_client_instance.get = AsyncMock(side_effect=Exception("Network error"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client.return_value = mock_client_instance

        result = await scraper.scrape_page("https://example.com/test")

        assert result is None

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_scrape_multiple(self, mock_client):
        """Test scraping multiple URLs concurrently."""
        scraper = WebScraper()

        # Mock response
        mock_response = Mock()
        mock_response.text = "<html><body><p>Test content</p></body></html>"
        mock_response.raise_for_status = Mock()

        mock_client_instance = Mock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client.return_value = mock_client_instance

        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3"
        ]

        results = await scraper.scrape_multiple(urls)

        assert len(results) == 3
        assert all(r["source"] == "example.com" for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
