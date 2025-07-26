from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import retriever
from llama_index.llms.openai import OpenAI
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

load_dotenv()

# 1. Load data
documents = SimpleDirectoryReader("pdf/").load_data()

# Create a Chroma database
db = chromadb.PersistentClient(path="./chromadb")
# Create a colletion for the database
chroma_collection = db.get_or_create_collection("quickstart")
# Create a chroma vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# Create a storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# Create a vector store index from the chroma database - non persistent
# index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
# Create an index that loads and persists the index to the chroma database - we don't need
# to pass in documents because they already exist in the chroma database
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
# Create the query engine
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
# Create a response synthesizer
response_synthesizer = get_response_synthesizer()
# Create a retriever query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)
# Query the engine
response = query_engine.query("What is machine learning in one paragraph?")
print(response)

