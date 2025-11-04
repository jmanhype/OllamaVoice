# OllamaVoice

An AI-powered research and video generation tool that combines Ollama's language capabilities with OuteTTS for high-quality text-to-speech generation.

## Features

- Text-to-Speech generation using OuteTTS
- Research capabilities powered by Ollama
- Video generation with AI narration
- Support for multiple languages (English, Japanese, Korean, Chinese)
- Real-time audio generation
- Beautiful web interface

## Requirements

- Python 3.12+
- Ollama running locally
- FFmpeg for video processing
- PyTorch (with CUDA support recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/OllamaVoice.git
cd OllamaVoice
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: For development, install testing dependencies:
```bash
pip install pytest pytest-asyncio pytest-cov
```

4. Create necessary directories:
```bash
mkdir -p static/temp static/videos
```

5. Create `.env` file with your configuration:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

## Usage

1. Start the server:
```bash
uvicorn main:app --reload
```

2. Open your browser and navigate to `http://localhost:8000`

3. Use the interface to:
   - Generate TTS audio
   - Research topics
   - Create AI-narrated videos

## Testing

Run the test suite:
```bash
pytest
```

Run tests with coverage report:
```bash
pytest --cov=. --cov-report=html
```

Run specific test file:
```bash
pytest test_web_scraper.py -v
```

## Architecture

- FastAPI backend
- OuteTTS for text-to-speech
- Ollama for AI research
- FFmpeg for video processing
- Modern web interface

## Documentation

See the `docs` directory for detailed documentation:
- [Token Handling](docs/token_handling.md)

## License

MIT License
