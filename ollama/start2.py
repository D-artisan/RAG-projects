import ollama
import os
from dotenv import load_dotenv
import time
import re
import datetime

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

# Configure ollama client with custom base URL
base_url = os.getenv("OLLAMA_BASE_URL")
model_name = os.getenv("OLLAMA_MODEL")

print(f"{Colors.CYAN}{Colors.BOLD}🤖 Ollama Chat Assistant{Colors.RESET}")
print(f"{Colors.WHITE}{'=' * 50}{Colors.RESET}")
print(f"{Colors.GREEN}📡 Connected to:{Colors.RESET} {Colors.DIM}{base_url}{Colors.RESET}")
print(f"{Colors.BLUE}🧠 Using model:{Colors.RESET} {Colors.BOLD}{model_name}{Colors.RESET}")
print(f"{Colors.WHITE}{'=' * 50}{Colors.RESET}")

# Create ollama client with custom host
client = ollama.Client(host=base_url)

question = "Explain Deep Learning to an absolute beginner."
print(f"\n{Colors.YELLOW}💬 Question:{Colors.RESET} {Colors.ITALIC}{question}{Colors.RESET}")
print(f"\n{Colors.CYAN}🔄 Generating response...{Colors.RESET}\n")
print(f"{Colors.GREEN}{Colors.BOLD}📝 Response:{Colors.RESET}")
print(f"{Colors.WHITE}{'-' * 30}{Colors.RESET}")

# Ensure 'responses' directory exists
responses_dir = os.path.join(os.getcwd(), "responses-2")
os.makedirs(responses_dir, exist_ok=True)

try:
    res = client.chat(
        model=model_name, 
        messages=[
            {"role": "user", "content": question}
        ],
        stream=True
    )

    response_text = ""
    for chunk in res:
        content = chunk["message"]["content"]
        response_text += content
        # Format and print the content with styling
        formatted_content = format_markdown_text(content)
        print(formatted_content, end="", flush=True)
    
    print(f"\n{Colors.WHITE}{'-' * 30}{Colors.RESET}")
    print(f"{Colors.GREEN}✅ Response completed successfully!{Colors.RESET}")
    print(f"{Colors.BLUE}📊 Total characters:{Colors.RESET} {Colors.BOLD}{len(response_text)}{Colors.RESET}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    response_file = os.path.join(responses_dir, f"response_{timestamp}.md")
    with open(response_file, "w", encoding="utf-8") as f:
        f.write("# Response\n\n")
        f.write(response_text.strip() + "\n")
    print(f"Response written to {response_file}")
    
except Exception as e:
    print(f"\n{Colors.RED}❌ Error occurred:{Colors.RESET} {Colors.BOLD}{e}{Colors.RESET}")
    print(f"{Colors.YELLOW}Please check your connection and try again.{Colors.RESET}")

print(f"\n{Colors.WHITE}{'=' * 50}{Colors.RESET}")
print(f"{Colors.MAGENTA}{Colors.BOLD}🎉 Chat session completed!{Colors.RESET}")


# ==================================================================================
# ==== The Ollama Python library's API is designed around the Ollama REST API ====
# ==================================================================================
