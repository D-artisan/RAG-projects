import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the base URL and construct the pull endpoint
base_url = os.getenv("OLLAMA_BASE_URL")
pull_url = f"{base_url}/api/pull"
model_name = os.getenv("GET_MODEL")

print(f"Pulling model: {model_name}")
print(f"Using endpoint: {pull_url}")

data = {
    "name": model_name
}

try:
    response = requests.post(pull_url, json=data, stream=True)
    
    if response.status_code == 200:
        print(f"Pulling model '{model_name}'...")
        
        # Iterate over the streaming response to show progress
        for line in response.iter_lines():
            if line:
                try:
                    # Decode the line and parse the JSON
                    decoded_line = line.decode("utf-8")
                    result = json.loads(decoded_line)
                    
                    # Show status updates
                    if "status" in result:
                        status = result["status"]
                        print(f"Status: {status}")
                        
                        # Show progress if available
                        if "completed" in result and "total" in result:
                            completed = result["completed"]
                            total = result["total"]
                            percentage = (completed / total) * 100 if total > 0 else 0
                            print(f"Progress: {completed}/{total} ({percentage:.1f}%)")
                    
                    # Check if pull is complete
                    if result.get("status") == "success":
                        print(f"✅ Model '{model_name}' pulled successfully!")
                        break
                        
                except json.JSONDecodeError:
                    # Some lines might not be valid JSON, skip them
                    continue
                    
    else:
        print(f"❌ Error pulling model: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.RequestException as e:
    print(f"❌ Connection error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
