## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

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
doc_path = os.getenv("DOC_PATH")

if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("done loading...")
else:
    print("Upload a PDF file to start the process.")

content = data[0].page_content
print(content[:100])