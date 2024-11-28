import logging
from typing import List, Dict
import json
import asyncio

logger = logging.getLogger(__name__)

class ResearchEngine:
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client

    async def research_topic(self, topic: str, progress_callback=None) -> str:
        try:
            async def send_progress(message: str):
                if progress_callback:
                    progress_callback(message)

            # Step 1: Generate focus areas
            await send_progress("Generating research focus areas...")
            focus_areas = await self._generate_focus_areas(topic)
            logger.info(f"Generated focus areas: {focus_areas}")

            # Step 2: Research each area
            research_data = await self._research_focus_areas(focus_areas, send_progress)
            
            # Step 3: Generate final script
            await send_progress("Generating final script...")
            script = await self._generate_final_script(topic, research_data)
            
            return script

        except Exception as e:
            logger.error(f"Error in research_topic: {str(e)}")
            raise ValueError(f"Research failed: {str(e)}")

    async def _generate_focus_areas(self, topic: str) -> list:
        focus_areas_prompt = f"""Generate 2 specific research focus areas for the topic: {topic}.
Focus on key aspects that would make an engaging educational video.
Format as a numbered list. Keep each focus area concise."""

        focus_areas_text = await self.ollama_client.generate(focus_areas_prompt)
        if not focus_areas_text:
            raise ValueError("Failed to generate research focus areas")
            
        focus_areas = [area.strip() for area in focus_areas_text.split('\n') if area.strip()]
        if not focus_areas:
            raise ValueError("No valid research focus areas generated")
        
        return focus_areas

    async def _research_focus_area(self, area: str) -> dict:
        content_prompt = f"""Generate detailed, factual information about this topic: {area}
Include:
1. Key concepts (2-3 points)
2. Important facts (2-3 points)
3. One real-world example
Keep the content concise and engaging."""

        content = await self.ollama_client.generate(content_prompt)
        if not content:
            return None
            
        return {
            'area': area,
            'content': content,
            'title': f"Research on {area}"
        }

    async def _research_focus_areas(self, focus_areas: list, send_progress) -> list:
        research_data = []
        for i, area in enumerate(focus_areas, 1):
            await send_progress(f"Researching area {i} of {len(focus_areas)}: {area[:50]}...")
            try:
                # Add delay between requests to respect rate limits
                await asyncio.sleep(2)
                
                result = await self._research_focus_area(area)
                if result:
                    research_data.append(result)
            except Exception as e:
                logger.error(f"Error researching focus area {area}: {str(e)}")
                continue

        return research_data

    async def _generate_final_script(self, topic: str, research_data: list) -> str:
        await asyncio.sleep(2)  # Rate limiting delay

        if research_data:
            script_prompt = f"""Create a concise video script about {topic} using this research:
{json.dumps(research_data, indent=2)}

Structure:
1. Brief introduction
2. Main points from research
3. Quick conclusion

Keep it engaging but brief."""

            script = await self.ollama_client.generate(script_prompt)
            if not script:
                raise ValueError("Failed to generate video script")
            return script
        else:
            # Fallback to simpler script
            backup_prompt = f"""Create a brief educational video script about {topic}.
Include:
1. Quick introduction
2. 2-3 main points
3. Brief conclusion
Keep it concise and engaging."""

            script = await self.ollama_client.generate(backup_prompt)
            if not script:
                raise ValueError("Failed to generate video script using backup prompt")
            return script
