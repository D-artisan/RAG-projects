import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the base URL and construct the tags endpoint
base_url = os.getenv("OLLAMA_BASE_URL")
tags_url = f"{base_url}/api/tags"

print(f"Checking models at: {tags_url}")
print("-" * 50)

try:
    response = requests.get(tags_url)
    
    if response.status_code == 200:
        result = response.json()
        
        if "models" in result and result["models"]:
            print("📋 Available Models:")
            print("=" * 50)
            
            for i, model in enumerate(result["models"], 1):
                name = model.get("name", "Unknown")
                size = model.get("size", 0)
                modified_at = model.get("modified_at", "Unknown")
                
                # Convert size to human readable format
                if size > 0:
                    if size >= 1024**3:  # GB
                        size_str = f"{size / (1024**3):.1f} GB"
                    elif size >= 1024**2:  # MB
                        size_str = f"{size / (1024**2):.1f} MB"
                    else:
                        size_str = f"{size} bytes"
                else:
                    size_str = "Unknown size"
                
                print(f"{i}. 🤖 {name}")
                print(f"   📦 Size: {size_str}")
                print(f"   📅 Modified: {modified_at}")
                print("-" * 30)
                
            print(f"\n✅ Found {len(result['models'])} model(s)")
        else:
            print("📭 No models found on this Ollama instance")
            
    else:
        print(f"❌ Error checking models: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.RequestException as e:
    print(f"❌ Connection error: {e}")
    print("Make sure the Ollama server is running and accessible")
except json.JSONDecodeError as e:
    print(f"❌ Error parsing response: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
