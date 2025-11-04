"""
Unit tests for the ResearchEngine class.
"""
import pytest
from research_engine import ResearchEngine
from unittest.mock import Mock, AsyncMock


class MockOllamaClient:
    """Mock Ollama client for testing."""

    def __init__(self):
        self.responses = []
        self.call_count = 0

    def set_responses(self, responses):
        """Set pre-defined responses for generate calls."""
        self.responses = responses
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        """Mock generate method."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "Mock response"


class TestResearchEngine:
    """Test suite for ResearchEngine functionality."""

    def test_init(self):
        """Test ResearchEngine initialization."""
        mock_client = MockOllamaClient()
        engine = ResearchEngine(mock_client)
        assert engine.ollama_client == mock_client

    @pytest.mark.asyncio
    async def test_generate_focus_areas_success(self):
        """Test successful focus area generation."""
        mock_client = MockOllamaClient()
        mock_client.set_responses([
            "1. Machine learning algorithms\n2. Neural networks basics"
        ])

        engine = ResearchEngine(mock_client)
        focus_areas = await engine._generate_focus_areas("AI")

        assert len(focus_areas) == 2
        assert "Machine learning" in focus_areas[0]
        assert "Neural networks" in focus_areas[1]

    @pytest.mark.asyncio
    async def test_generate_focus_areas_empty_response(self):
        """Test focus area generation with empty response."""
        mock_client = MockOllamaClient()
        mock_client.set_responses([""])

        engine = ResearchEngine(mock_client)

        with pytest.raises(ValueError, match="Failed to generate research focus areas"):
            await engine._generate_focus_areas("AI")

    @pytest.mark.asyncio
    async def test_generate_focus_areas_no_valid_areas(self):
        """Test focus area generation with no valid areas."""
        mock_client = MockOllamaClient()
        mock_client.set_responses(["\n\n\n"])

        engine = ResearchEngine(mock_client)

        with pytest.raises(ValueError, match="No valid research focus areas generated"):
            await engine._generate_focus_areas("AI")

    @pytest.mark.asyncio
    async def test_research_focus_area_success(self):
        """Test successful focus area research."""
        mock_client = MockOllamaClient()
        mock_client.set_responses([
            "Machine learning is a subset of AI. Key concepts include supervised learning, unsupervised learning, and reinforcement learning."
        ])

        engine = ResearchEngine(mock_client)
        result = await engine._research_focus_area("Machine Learning Basics")

        assert result is not None
        assert result["area"] == "Machine Learning Basics"
        assert "Machine learning" in result["content"]
        assert "title" in result

    @pytest.mark.asyncio
    async def test_research_focus_area_empty_response(self):
        """Test focus area research with empty response."""
        mock_client = MockOllamaClient()
        mock_client.set_responses([""])

        engine = ResearchEngine(mock_client)
        result = await engine._research_focus_area("Machine Learning")

        assert result is None

    @pytest.mark.asyncio
    async def test_research_focus_areas_multiple(self):
        """Test researching multiple focus areas."""
        mock_client = MockOllamaClient()
        mock_client.set_responses([
            "Content about machine learning",
            "Content about neural networks"
        ])

        engine = ResearchEngine(mock_client)

        async def mock_progress(msg: str):
            pass

        focus_areas = ["Machine Learning", "Neural Networks"]
        results = await engine._research_focus_areas(focus_areas, mock_progress)

        assert len(results) == 2
        assert results[0]["area"] == "Machine Learning"
        assert results[1]["area"] == "Neural Networks"

    @pytest.mark.asyncio
    async def test_research_focus_areas_with_errors(self):
        """Test researching focus areas when some fail."""
        mock_client = MockOllamaClient()
        mock_client.set_responses([
            "Content about machine learning",
            "",  # Empty response - should be skipped
            "Content about deep learning"
        ])

        engine = ResearchEngine(mock_client)

        async def mock_progress(msg: str):
            pass

        focus_areas = ["Machine Learning", "Failed Topic", "Deep Learning"]
        results = await engine._research_focus_areas(focus_areas, mock_progress)

        # Should only have 2 results (empty response is filtered out)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_generate_final_script_with_data(self):
        """Test script generation with research data."""
        mock_client = MockOllamaClient()
        mock_client.set_responses([
            "This is a comprehensive video script about AI. Introduction: AI is transforming the world..."
        ])

        engine = ResearchEngine(mock_client)
        research_data = [
            {"area": "ML", "content": "Machine learning content", "title": "ML Research"},
            {"area": "NN", "content": "Neural networks content", "title": "NN Research"}
        ]

        script = await engine._generate_final_script("AI", research_data)

        assert script
        assert "video script" in script.lower()

    @pytest.mark.asyncio
    async def test_generate_final_script_empty_data(self):
        """Test script generation with no research data (fallback mode)."""
        mock_client = MockOllamaClient()
        mock_client.set_responses([
            "This is a brief educational script about AI..."
        ])

        engine = ResearchEngine(mock_client)
        script = await engine._generate_final_script("AI", [])

        assert script
        assert len(script) > 0

    @pytest.mark.asyncio
    async def test_generate_final_script_failure(self):
        """Test script generation failure."""
        mock_client = MockOllamaClient()
        mock_client.set_responses([""])

        engine = ResearchEngine(mock_client)
        research_data = [{"area": "ML", "content": "Content", "title": "Title"}]

        with pytest.raises(ValueError, match="Failed to generate video script"):
            await engine._generate_final_script("AI", research_data)

    @pytest.mark.asyncio
    async def test_research_topic_full_workflow(self):
        """Test full research topic workflow."""
        mock_client = MockOllamaClient()
        mock_client.set_responses([
            "1. Machine Learning\n2. Deep Learning",  # Focus areas
            "Machine learning content",  # Research area 1
            "Deep learning content",  # Research area 2
            "Final comprehensive video script"  # Final script
        ])

        engine = ResearchEngine(mock_client)

        async def mock_progress(msg: str):
            pass

        script = await engine.research_topic("AI", progress_callback=mock_progress)

        assert script == "Final comprehensive video script"
        assert mock_client.call_count == 4

    @pytest.mark.asyncio
    async def test_research_topic_with_error(self):
        """Test research topic with error handling."""
        mock_client = Mock()
        mock_client.generate = AsyncMock(side_effect=Exception("API error"))

        engine = ResearchEngine(mock_client)

        with pytest.raises(ValueError, match="Research failed"):
            await engine.research_topic("AI")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
