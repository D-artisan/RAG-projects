import sys
import os
from pathlib import Path

# Add the agents directory to the path
sys.path.append(str(Path(__file__).parent))

from base_agent import BaseAgent

# Test creating a BaseAgent instance
agent = BaseAgent("Test Agent", "Test instructions")

print("BaseAgent created successfully!")
print(f"base_url used in BaseAgent: {agent.ollama_client.base_url}")
print(f"OLLAMA_BASE_URL from environment: {os.getenv('OLLAMA_BASE_URL')}")

# Test if we can access the client
try:
    # Just test if the client object is properly configured
    print(f"Client configured with base_url: {agent.ollama_client.base_url}")
    print("BaseAgent initialization successful!")
except Exception as e:
    print(f"Error with BaseAgent: {e}")
