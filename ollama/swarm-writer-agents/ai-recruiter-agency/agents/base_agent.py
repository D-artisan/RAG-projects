import os
from typing import Dict, Any
import json
from openai import OpenAI

from dotenv import load_dotenv

# Load .env from parent directory if not found in current directory

from pathlib import Path
dotenv_path = Path(__file__).parent / '.env'
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path)

base_url = os.getenv("OLLAMA_API_URL")


ollama_model = os.getenv("OLLAMA_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")


class BaseAgent:
    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions
        self.ollama_client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # required but unused
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        """Default run method to be overridden by child classes"""
        raise NotImplementedError("Subclasses must implement run()")

    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama model with the given prompt"""
        try:
            response = self.ollama_client.chat.completions.create(
                model=ollama_model, 
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying Ollama: {str(e)}")
            raise

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON from text, handling potential errors"""
        try:
            # Try to find JSON-like content between curly braces
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start : end + 1]
                return json.loads(json_str)
            return {"error": "No JSON content found"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON content"}
