import ollama
import os
from dotenv import load_dotenv


# Load .env from parent directory if not found in current directory
from pathlib import Path
dotenv_path = Path(__file__).parent / '.env'
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path)


base_url = os.getenv("OLLAMA_BASE_URL")
model_name = os.getenv("OLLAMA_MODEL")
input_file_path = os.getenv("INPUT_FILE_PATH")
output_file_path = os.getenv("OUTPUT_FILE_PATH")

if not base_url:
    print("Error: OLLAMA_BASE_URL is not set in the .env file.")
    exit(1)
if not model_name:
    print("Error: OLLAMA_MODEL is not set in the .env file.")
    exit(1)

# Connectivity check
import requests
try:
    resp = requests.get(base_url)
    if resp.status_code != 200:
        print(f"Error: Ollama server at {base_url} is not responding (status {resp.status_code}).")
        exit(1)
except Exception as e:
    print(f"Error: Could not connect to Ollama server at {base_url}. Exception: {e}")
    print("Please ensure the Ollama server is running and accessible.")
    exit(1)

client = ollama.Client(host=base_url)


input_file = input_file_path
output_file = output_file_path

if not os.path.exists(input_file):
    print(f"Input file {input_file} does not exist.")
    exit(1)

with open(input_file, 'r') as f:
    items = f.read().strip()


prompt = f"""
You are a grocery categoriser. Categorise the following grocery items into appropriate categories:
{items}
Return the categories in a structured format, like this:
- Fruits: [item1, item2]
- Vegetables: [item3, item4]
- Dairy: [item5, item6]
- Grains: [item7, item8]
- Proteins: [item9, item10]
- Snacks: [item11, item12]
- Beverages: [item13, item14]
- Condiments: [item15, item16]      

Sort the items alphabetically within each category.
"""

try:
    response = client.generate(model=model_name, prompt=prompt, think=False, stream=True)
    # generated_text = response.get("response", "")
    # print("=== Categorised Grocery List ===")
    # print(generated_text)

    # with open(output_file, 'w') as f:
    #     f.write(generated_text.strip())

    for chunk in response:
        print(chunk.get("response", ""), end="", flush=True)

    # print(f"Categorised grocery list saved to {output_file}")
    print("\n")
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)
