# Observability is what is happening behind the scenes
# We can use the llama_index.core.observability module to track the performance of the RAG pipeline
# Whenever we run something with an LLM, we want to know how much it costs, 
# how long it takes, and how many tokens it uses 

# After running this code, go to the phoenix dashboard and you will see the traces


from dotenv import load_dotenv
import os
from llama_index.core import set_global_handler, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage

load_dotenv()

PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://api.us.phoenix.cloud.unity3d.com"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"api_key={PHOENIX_API_KEY}"
# This next URL is provide by phoenix when the API key is created
set_global_handler("arize_phoenix", endpoint="https://app.phoenix.arize.com/s/nicholas-reese2")

# Everything is set up now - now we need to code the RAG pipeline

documents = SimpleDirectoryReader("pdf/").load_data()

if os.path.exists("storage"):
    print("Loading index from storage ...")
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
else:
    print("Create the new index ...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="storage")

query_engine = index.as_query_engine()

response = query_engine.query("What is machine learning in one paragraph?")
print(response)

print(response)