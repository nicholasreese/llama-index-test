# Use LlamaParse to parse PDFs and other files - this is a paid service
# but it is free for a limited time
# https://llamaparse.com/
# LlamaParse is good for parsing complex PDF documents that are not well structured
# and have a lot of information - especially graphs and tables

import nest_asyncio

nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from dotenv import load_dotenv

load_dotenv()
# The api key is provided by llama_parse - it could also be stored in a .env file
# as an environment variable
parser = LlamaParse(
    api_key="llx-2CTNonAORzexB3Mrty4hRTeXZi0Hy63fbjcQoHyGkfxXa4fK",
    result_type="markdown",
    verbose=True,
)

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("pdf/", file_extractor=file_extractor).load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("What is the main idea of the document?")

print(response)