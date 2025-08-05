#!/usr/bin/env python3
"""
AI Learning Quest - Launcher Script
Run this script to start the educational AI game with Streamlit
"""
import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'requests', 'python-dotenv', 'plotly', 
        'scikit-learn', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:", missing_packages)
        print("Installing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)

def main():
    print("🤖 AI Learning Quest - Starting...")
    
    # Check dependencies
    check_dependencies()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Check if .env file exists
    env_file = script_dir / ".env"
    if not env_file.exists():
        print("⚠️  .env file not found. Creating a default one...")
        default_env = """# Ollama Configuration
OLLAMA_API_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3.2

# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo"""
        
        with open(env_file, 'w') as f:
            f.write(default_env)
        
        print("✅ Default .env file created. Please update it with your API settings.")
    
    # Run Streamlit app
    app_file = script_dir / "ai_learning_game.py"
    
    if not app_file.exists():
        print("❌ ai_learning_game.py not found!")
        return
    
    print("🚀 Launching AI Learning Quest...")
    print("📱 The app will open in your browser automatically.")
    print("🛑 Press Ctrl+C to stop the application.")
    
    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(app_file), 
        "--server.port", "8501",
        "--server.headless", "false"
    ])

if __name__ == "__main__":
    main()
