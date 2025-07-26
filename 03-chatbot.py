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

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = chat_engine.chat(text_input)
    print(f"Agent: {response}")