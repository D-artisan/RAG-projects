import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

url = os.getenv("OLLAMA_API_URL")

data = {
    "model": os.getenv("OLLAMA_MODEL"), 
    "prompt": "How can AI benefit Insrance",
}

response = requests.post(url, json=data, stream=True)

# check the response status
if response.status_code == 200:
    print("Generated Text:", end=" ", flush=True)
    # Iterate over the streaming response
    for line in response.iter_lines():
        if line:
            # Decode the line and parse the JSON
            decoded_line = line.decode("utf-8")
            result = json.loads(decoded_line)
            # Get the text from the response
            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)
else:
    print("Error:", response.status_code, response.text)
