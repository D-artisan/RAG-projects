from pathlib import Path
from dotenv import load_dotenv
import os

print("Current working directory:", os.getcwd())
print("Script location:", __file__)

# Test the same logic as base_agent.py
dotenv_path = Path(__file__).parent / '.env'
print(f'Looking for: {dotenv_path}')
print(f'Exists: {dotenv_path.exists()}')

if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent.parent / '.env'
    print(f'Fallback to: {dotenv_path}')
    print(f'Exists: {dotenv_path.exists()}')

load_dotenv(dotenv_path)
print(f'OLLAMA_BASE_URL: {os.getenv("OLLAMA_BASE_URL")}')
print(f'OLLAMA_MODEL: {os.getenv("OLLAMA_MODEL")}')
