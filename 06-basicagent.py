# This code has come from the youtube channel "Tyler AI"
# Everything you need to know about LlamaIndex - Tyler AI
# Creating a basic agent
import os
from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

load_dotenv()

llm = OpenAI(model="gpt-4o-mini")

# Create some functions to use as tools
def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    return a / b    

# Create a tool for each function in a way that the LLM can understand
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
multiply_tool = FunctionTool.from_defaults(fn=multiply)
divide_tool = FunctionTool.from_defaults(fn=divide)

# Create a ReActAgent with the tools
agent = ReActAgent.from_tools([add_tool, subtract_tool, multiply_tool, divide_tool], llm=llm, verbose=True)

# Create a chat engine with the agent
# chat_engine = agent.as_chat_engine()

# Chat with the agent
response = agent.chat("What is 10 + 5?")
print(response)


