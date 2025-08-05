# app.py

import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# from langchain_community.embeddings.ollama import OllamaEmbeddings: deprecated
from langchain_ollama import OllamaEmbeddings

from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_community.chat_models.ollama import ChatOllama: deprecated
from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
from dotenv import load_dotenv

# Load .env from parent directory if not found in current directory

from pathlib import Path
dotenv_path = Path(__file__).parent / '.env'
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent.parent / '.env'
st.write("Using dotenv path:", dotenv_path)
st.write("Dotenv exists:", dotenv_path.exists())
load_dotenv(dotenv_path)
st.write("OLLAMA_BASE_URL after load_dotenv:", os.getenv("OLLAMA_BASE_URL"))

# base_url = os.getenv("OLLAMA_BASE_URL")
base_url = os.getenv("OLLAMA_BASE_URL") 


model_name = os.getenv("OLLAMA_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")
doc_path = os.getenv("DOC_PATH")
# Set a default PDF path if DOC_PATH is not set
if not doc_path:
    doc_path = "./data/BOI.pdf"  # Default to a sample PDF in the data folder
    st.warning(f"DOC_PATH environment variable not set. Using default: {doc_path}")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = doc_path
MODEL_NAME = model_name
EMBEDDING_MODEL = embedding_model
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    # Pull the embedding model if not already available

    embedding = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=base_url,          
)


    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Vector database created and persisted.")
    return vector_db


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain


def main():
    st.title("Document Assistant")

    st.write("OLLAMA_BASE_URL:", os.getenv("OLLAMA_BASE_URL"))

    # User input
    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Initialize the language model
                llm = ChatOllama(
                    model=model_name,
                    base_url=base_url,           
                )

                # Load the vector database
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                # Create the retriever
                retriever = create_retriever(vector_db, llm)

                # Create the chain
                chain = create_chain(retriever, llm)

                # Get the response
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()