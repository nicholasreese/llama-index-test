# from tabnanny import verbose
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
# from openai import chat

load_dotenv()

llm = OpenAI(model="gpt-4o-mini")
data = SimpleDirectoryReader("pdf/").load_data()
index = VectorStoreIndex.from_documents(data)

chat_engine = index.as_chat_engine(chat_mode="best",llm=llm, verbose=True)

response = chat_engine.chat("Explain about training on multiple GPUs")

print(response)

