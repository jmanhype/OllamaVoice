import os
from dotenv import load_dotenv
import inspect
import torch
import outetts.interface as interface
from outetts.version.v1.interface import ModelOutput
from outetts.version.v1.model import GenerationConfig
from outetts.interface import HFModelConfig_v1
from fastapi import FastAPI, WebSocket, HTTPException, Query, Request, Form, File, UploadFile, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from loguru import logger
from typing import Optional
import json
import uuid
import shutil
import base64
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Load environment variables
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import httpx
import asyncio
from moviepy.editor import AudioFileClip, VideoFileClip, ColorClip, TextClip, CompositeVideoClip
import tempfile
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from research_engine import ResearchEngine
import logging
import time
import warnings

# Filter specific warnings
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set.*")
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:None.*")
warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred from input.*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.add("app.log", rotation="500 MB")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create necessary directories
os.makedirs("static/videos", exist_ok=True)
os.makedirs("static/temp", exist_ok=True)

tts = None
speaker = None

def init_tts_interface():
    """Initialize the text-to-speech interface with proper configuration."""
    try:
        logger.info("=== Starting Diagnostic Checks ===")
        
        # Check directories exist and are writable
        for dir_path in ['static/temp', 'static/videos']:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            writable = os.access(path, os.W_OK)
            logger.info(f"Directory {dir_path}: exists={path.exists()}, writable={writable}")

        # Initialize OuteTTS with HF model configuration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_config = HFModelConfig_v1(
            model_path='OuteAI/OuteTTS-0.2-500M',
            language='en',
            device=device,
            dtype=None,   # Will use default dtype
            additional_model_config={
                'use_cache': True,
                'return_dict_in_generate': True,
                'do_sample': True,
                'pad_token_id': 1,  # Different from eos_token_id
                'eos_token_id': 2,  # Different from pad_token_id
            }
        )
        
        # Initialize tokenizer with proper padding configuration
        tokenizer_path = model_config.model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, 
            padding_side='right',
            truncation=True,
            model_max_length=4096
        )
        
        # Set tokenizer special tokens
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 2
        
        # Create generation config
        generation_config = GenerationConfig(
            pad_token_id=1,
            eos_token_id=2,
            max_length=4096,
            do_sample=True
        )
        model_config.additional_model_config['generation_config'] = generation_config
        
        # Initialize TTS interface
        tts = interface.InterfaceHF(model_version="0.2", cfg=model_config)
        logger.info("OuteTTS interface initialized successfully")
        logger.info(f"Model config: {model_config}")
        
        # Load speaker configuration
        speaker = tts.load_default_speaker("female_1")
        if not speaker:
            raise ValueError("Failed to load speaker configuration")
        logger.info("Speaker loaded successfully")
        
        # Log interface parameters
        sig = inspect.signature(tts.generate)
        logger.info(f"OuteTTS interface parameters: {sig}")
        
        return tts, speaker
        
    except Exception as e:
        logger.error(f"Failed to initialize TTS interface: {str(e)}")
        raise

def get_tts_interface():
    try:
        return init_tts_interface()
    except Exception as e:
        logger.error(f"Error initializing TTS interface: {str(e)}")
        return None, None

tts, speaker = get_tts_interface()

class OllamaClient:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "mistral"
        self.config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["User:", "\n\n"],
            "num_ctx": 32000,
            "num_predict": 32000
        }
        self.last_request_time = 0
        self.min_request_interval = 60  # Increased to 60 seconds between requests
        self.max_retries = 3
        self.retry_delay = 60  # Increased to 60 seconds base retry delay
        self.backoff_factor = 2  # Each retry will wait 2x longer

    async def generate(self, prompt: str) -> str:
        """Generate text using Ollama API with rate limiting and retries"""
        for attempt in range(self.max_retries):
            try:
                # Enforce rate limiting
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    wait_time = self.min_request_interval - time_since_last
                    logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={"model": self.model, "prompt": prompt, **self.config},
                        timeout=300.0
                    )
                    
                    if response.status_code == 429:  # Rate limit exceeded
                        retry_after = self.retry_delay * (self.backoff_factor ** attempt)
                        logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds before retry.")
                        await asyncio.sleep(retry_after)
                        continue
                        
                    response.raise_for_status()
                    self.last_request_time = time.time()
                    
                    # Process streaming response
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    full_response += data["response"]
                            except json.JSONDecodeError:
                                continue
                    return full_response.strip()

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    retry_after = self.retry_delay * (self.backoff_factor ** attempt)
                    logger.info(f"Retrying in {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                else:
                    raise ValueError(f"Failed to generate after {self.max_retries} attempts: {str(e)}")

ollama_client = OllamaClient()
research_engine = ResearchEngine(ollama_client)

async def get_ollama_response(message: str) -> str:
    """Get response from Ollama API"""
    return await ollama_client.generate(message)

async def text_to_speech(text: str) -> tuple[bytes, str]:
    """
    Convert text to speech using OuteTTS.
    
    Args:
        text (str): The text to convert to speech
        
    Returns:
        tuple[bytes, str]: The generated audio as WAV bytes and the path to the audio file
    """
    audio_path = "static/temp/temp_audio.wav"
    try:
        # Log the text being processed
        logger.info(f"Generating speech for text: {text}...")
        
        # Get TTS interface and speaker
        if not tts:
            raise ValueError("TTS interface not initialized")
        
        # Process text and create input with attention mask
        input_text = text.strip()
        if not input_text:
            raise ValueError("Empty text input")
            
        # Generate speech with OuteTTS's supported parameters
        output = tts.generate(
            input_text,
            speaker=speaker,
            temperature=0.1,
            repetition_penalty=1.1,
            max_length=4096  # Maximum supported length from model config
        )
        
        # Check if we have valid audio
        if output.audio is None or output.audio.numel() == 0:
            raise ValueError("Generated audio is empty")
            
        # Save to temporary file
        output.save(audio_path)
        
        # Get file size for logging
        file_size = os.path.getsize(audio_path)
        logger.info(f"Successfully generated audio file of size {file_size} bytes")
        
        # Read the file
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            
        return audio_bytes, audio_path
            
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        raise

async def create_video(title: str, script: str, background_video: str = None) -> str:
    """Create a video with AI narration"""
    audio_path = None
    video = None
    final_video = None
    output_path = None
    
    try:
        # Clean and validate input
        script = script.strip()
        if not script:
            raise ValueError("Empty script provided")
            
        # Generate audio narration
        try:
            _, audio_path = await text_to_speech(script)
        except Exception as e:
            logger.error(f"Failed to generate audio: {str(e)}")
            raise ValueError(f"Audio generation failed: {str(e)}")

        # Verify audio file exists
        if not audio_path or not os.path.exists(audio_path):
            raise ValueError("Audio file not generated")

        # Load audio and get duration
        audio = AudioFileClip(audio_path)
        
        # Get audio duration and validate it
        duration = audio.duration
        if duration is None or not isinstance(duration, (int, float)) or duration <= 0:
            # Fallback to a minimum duration if audio duration is invalid
            logger.warning("Invalid audio duration, using fallback duration")
            duration = 10.0  # 10 seconds fallback
        
        # Create video
        try:
            if background_video and os.path.exists(background_video):
                video = VideoFileClip(background_video)
                # Loop video if shorter than audio
                if video.duration < duration:
                    video = video.loop(duration=duration)
                # Trim video if longer than audio
                else:
                    video = video.subclip(0, duration)
            else:
                # Create blank video with text
                video = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
                txt_clip = TextClip(
                    script[:1000],  # Limit text length to avoid overflow
                    fontsize=30,
                    color='white',
                    size=(1800, 1000),
                    method='caption'
                ).set_duration(duration)
                txt_clip = txt_clip.set_position('center')
                video = CompositeVideoClip([video, txt_clip])
        except Exception as e:
            logger.error(f"Failed to create video: {str(e)}")
            raise ValueError(f"Video creation failed: {str(e)}")
        
        # Combine audio and video
        try:
            final_video = video.set_audio(audio)
            output_path = f"static/videos/{title.replace(' ', '_')}.mp4"
            final_video.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                logger=None  # Disable moviepy logging
            )
            
            # Verify output file exists and has content
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise ValueError("Failed to write video file")
                
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to write video file: {str(e)}")
            raise ValueError(f"Video file creation failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in create_video: {str(e)}")
        raise ValueError(str(e))
    finally:
        # Clean up resources
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if video:
                video.close()
            if final_video:
                final_video.close()
            if background_video and os.path.exists(background_video):
                os.remove(background_video)
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

@app.get("/test_tts")
async def test_tts():
    """Test endpoint for text-to-speech generation with JSON response"""
    try:
        text = "Hello, how are you?"  # Removed clear markers
        audio_result = await text_to_speech(text)
        
        # Read the generated audio file and encode as base64
        with open(audio_result[1], 'rb') as f:
            audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
        return JSONResponse({
            "status": "success",
            "message": "Text-to-speech test completed successfully",
            "audio": audio_base64
        })
    except Exception as e:
        logger.error(f"Text-to-speech test failed: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/test_tts_raw")
async def test_tts_raw():
    """Test endpoint for text-to-speech generation with raw audio response"""
    try:
        text = "Hello, how are you?"  # Removed clear markers
        audio_result = await text_to_speech(text)
        
        # Read the generated audio file
        with open(audio_result[1], 'rb') as f:
            audio_data = f.read()
            
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=test.wav"
            }
        )
    except Exception as e:
        logger.error(f"Text-to-speech test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech generation failed: {str(e)}")

@app.get("/videos/{video_name}")
async def get_video(video_name: str):
    video_path = f"static/videos/{video_name}"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path)

@app.post("/create_video")
async def create_video_endpoint(
    title: str = Form(...),
    script: str = Form(...),
    background_video: UploadFile = File(None)
):
    try:
        bg_video_path = None
        if background_video:
            # Save uploaded video temporarily
            bg_video_path = f"static/temp/{background_video.filename}"
            with open(bg_video_path, "wb") as f:
                f.write(await background_video.read())
        
        # Create video with the script
        video_path = await create_video(title=title, script=script, background_video=bg_video_path)
        
        # Get the filename from the path
        video_name = os.path.basename(video_path)
        
        # Clean up
        if bg_video_path:
            os.remove(bg_video_path)
        
        return {
            "status": "success", 
            "video_path": f"/videos/{video_name}",
            "video_name": video_name,
            "script": script
        }
    except Exception as e:
        logger.error(f"Error in create_video_endpoint: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/research_video")
async def research_video(topic: str = Query(...)):
    async def generate():
        try:
            # Send initial status
            yield f"data: {{\"status\": \"started\", \"message\": \"Starting research...\"}}\n\n"
            await asyncio.sleep(0.1)
            
            # Research phase with longer timeout
            yield f"data: {{\"status\": \"researching\", \"message\": \"Researching topic: {topic}...\"}}\n\n"
            try:
                # Use a longer timeout for research (5 minutes)
                script = await asyncio.wait_for(
                    research_engine.research_topic(topic, 
                        progress_callback=lambda msg: generate_progress(msg)),
                    timeout=300  # Increased to 5 minutes
                )
                if not script:
                    error_msg = "Failed to generate research content"
                    logger.error(error_msg)
                    yield f"data: {{\"status\": \"error\", \"message\": \"{error_msg}\"}}\n\n"
                    return
            except asyncio.TimeoutError:
                error_msg = "Research timed out. Please try again."
                logger.error(error_msg)
                yield f"data: {{\"status\": \"error\", \"message\": \"{error_msg}\"}}\n\n"
                return
            except ValueError as e:
                error_msg = str(e)
                logger.error(f"Research error: {error_msg}")
                yield f"data: {{\"status\": \"error\", \"message\": \"{error_msg}\"}}\n\n"
                return
            except Exception as e:
                error_msg = f"Unexpected error during research: {str(e)}"
                logger.error(error_msg)
                yield f"data: {{\"status\": \"error\", \"message\": \"{error_msg}\"}}\n\n"
                return
                
            # Video creation phase
            yield f"data: {{\"status\": \"creating\", \"message\": \"Creating video...\"}}\n\n"
            try:
                video_path = await asyncio.wait_for(create_video(topic, script), timeout=300)
                if not video_path:
                    error_msg = "Failed to create video"
                    logger.error(error_msg)
                    yield f"data: {{\"status\": \"error\", \"message\": \"{error_msg}\"}}\n\n"
                    return
            except asyncio.TimeoutError:
                error_msg = "Video creation timed out. Please try again."
                logger.error(error_msg)
                yield f"data: {{\"status\": \"error\", \"message\": \"{error_msg}\"}}\n\n"
                return
            except Exception as e:
                error_msg = f"Video creation failed: {str(e)}"
                logger.error(error_msg)
                yield f"data: {{\"status\": \"error\", \"message\": \"{error_msg}\"}}\n\n"
                return
                
            video_name = os.path.basename(video_path)
            
            # Success response
            success_response = {
                "status": "success",
                "video_path": f"/videos/{video_name}",
                "video_name": video_name,
                "script": script
            }
            yield f"data: {json.dumps(success_response)}\n\n"
            
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            logger.error(f"Error in research_video: {error_msg}")
            yield f"data: {{\"status\": \"error\", \"message\": \"{error_msg}\"}}\n\n"
    
    def generate_progress(message: str):
        return f"data: {{\"status\": \"researching\", \"message\": \"{message}\"}}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Error handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"An error occurred: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)}
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            
            # Get AI response from Ollama
            ai_response = await get_ollama_response(message)
            
            # Convert response to speech
            audio_data, _ = await text_to_speech(ai_response)
            
            # Send both text and audio back to client
            await websocket.send_json({
                "text": ai_response,
                "audio": base64.b64encode(audio_data).decode()
            })
            
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
