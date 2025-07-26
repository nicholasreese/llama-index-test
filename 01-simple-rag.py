import nltk  # Natural Language Toolkit for text processing
import os  # Provides functions to interact with the operating system
from dotenv import load_dotenv  # Loads environment variables from a .env file
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex  # LlamaIndex classes for reading data and building an index
from llama_index.llms.openai import OpenAI  # OpenAI LLM integration for LlamaIndex

# Download required NLTK data packages for text processing
nltk.download("punkt")  # Sentence tokenizer
nltk.download("punkt_tab")  # Tokenizer data    
nltk.download("stopwords")  # Common stopwords for various languages
nltk.download("averaged_perceptron_tagger")  # Part-of-speech tagger
nltk.download("maxent_treebank_pos_tagger")  # Another POS tagger
nltk.download("maxent_ne_chunker_tab")  # Named entity chunker data
nltk.download("maxent_ne_chunker_tab")  # (Duplicate) Named entity chunker data

# Load environment variables from a .env file (e.g., API keys)
load_dotenv()

# Initialize the OpenAI LLM (Large Language Model) with a specific model name
OpenAI(model="gpt-4o-mini")

# Read all documents from the 'pdf/' directory using LlamaIndex's SimpleDirectoryReader
# This loads and parses the PDF files for further processing
documents = SimpleDirectoryReader("pdf/").load_data()

# Create a vector store index from the loaded documents
# This enables efficient semantic search and retrieval
index = VectorStoreIndex.from_documents(documents)

# Create a query engine from the index to handle natural language queries
query_engine = index.as_query_engine()

# Query the engine with a specific question and get a response
response = query_engine.query("What is machine learning in one paragraph?")

# Print the response to the console
print(response)
