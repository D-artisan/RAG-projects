import requests
import json
import os
from dotenv import load_dotenv
import re

# ANSI color codes for styling
class Colors:
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    DIM = '\033[2m'

def format_markdown_text(text):
    """Convert basic markdown formatting to ANSI codes"""
    # Bold text (**text** or __text__)
    text = re.sub(r'\*\*(.*?)\*\*', f'{Colors.BOLD}\\1{Colors.RESET}', text)
    text = re.sub(r'__(.*?)__', f'{Colors.BOLD}\\1{Colors.RESET}', text)
    
    # Italic text (*text* or _text_)
    text = re.sub(r'\*(.*?)\*', f'{Colors.ITALIC}\\1{Colors.RESET}', text)
    text = re.sub(r'_(.*?)_', f'{Colors.ITALIC}\\1{Colors.RESET}', text)
    
    # Headers (# ## ###)
    text = re.sub(r'^### (.*?)$', f'{Colors.YELLOW}{Colors.BOLD}\\1{Colors.RESET}', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', f'{Colors.CYAN}{Colors.BOLD}\\1{Colors.RESET}', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$', f'{Colors.GREEN}{Colors.BOLD}\\1{Colors.RESET}', text, flags=re.MULTILINE)
    
    # Code blocks (`code`)
    text = re.sub(r'`(.*?)`', f'{Colors.MAGENTA}\\1{Colors.RESET}', text)
    
    return text

# Load environment variables from .env file
load_dotenv()

url = os.getenv("OLLAMA_API_URL")
model_name = os.getenv("OLLAMA_MODEL")

print(f"{Colors.CYAN}{Colors.BOLD}🚀 Ollama REST API Client{Colors.RESET}")
print(f"{Colors.WHITE}{'=' * 50}{Colors.RESET}")
print(f"{Colors.GREEN}📡 API Endpoint:{Colors.RESET} {Colors.DIM}{url}{Colors.RESET}")
print(f"{Colors.BLUE}🧠 Using model:{Colors.RESET} {Colors.BOLD}{model_name}{Colors.RESET}")
print(f"{Colors.WHITE}{'=' * 50}{Colors.RESET}")

prompt = "What is the difference between supervised and unsupervised learning?"
print(f"\n{Colors.YELLOW}💬 Prompt:{Colors.RESET} {Colors.ITALIC}{prompt}{Colors.RESET}")
print(f"\n{Colors.CYAN}🔄 Sending request to API...{Colors.RESET}\n")

data = {
    "model": model_name, 
    "prompt": prompt,
}

response = requests.post(url, json=data, stream=True)

print(f"{Colors.GREEN}{Colors.BOLD}📝 Response:{Colors.RESET}")
print(f"{Colors.WHITE}{'-' * 30}{Colors.RESET}")

# check the response status
if response.status_code == 200:
    try:
        response_text = ""
        # Iterate over the streaming response
        for line in response.iter_lines():
            if line:
                # Decode the line and parse the JSON
                decoded_line = line.decode("utf-8")
                result = json.loads(decoded_line)
                # Get the text from the response
                generated_text = result.get("response", "")
                response_text += generated_text
                # Format and print the content with styling
                formatted_text = format_markdown_text(generated_text)
                print(formatted_text, end="", flush=True)
        
        print(f"\n{Colors.WHITE}{'-' * 30}{Colors.RESET}")
        print(f"{Colors.GREEN}✅ Response completed successfully!{Colors.RESET}")
        print(f"{Colors.BLUE}📊 Total characters:{Colors.RESET} {Colors.BOLD}{len(response_text)}{Colors.RESET}")
        
    except json.JSONDecodeError as e:
        print(f"\n{Colors.RED}❌ JSON parsing error:{Colors.RESET} {Colors.BOLD}{e}{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Unexpected error:{Colors.RESET} {Colors.BOLD}{e}{Colors.RESET}")
        
else:
    print(f"{Colors.RED}❌ HTTP Error:{Colors.RESET} {Colors.BOLD}{response.status_code}{Colors.RESET}")
    print(f"{Colors.YELLOW}Response:{Colors.RESET} {response.text}")

print(f"\n{Colors.WHITE}{'=' * 50}{Colors.RESET}")
print(f"{Colors.MAGENTA}{Colors.BOLD}🎉 API session completed!{Colors.RESET}")
