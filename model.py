from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Loading Environment Variablees
load_dotenv()

# Getting credentials
API = os.getenv("OPENROUTER_API_KEY")
URL = os.getenv("BASE_URL")

# Instancing Model
def ChatOpenAIModel(name : str = "gpt-oss-20b", temperature : float = 1.0):
    
    llm = ChatOpenAI(
        model=name,
        api_key=API,
        base_url=URL,
        temperature=temperature,
        default_headers={
        "HTTP-Referer": "http://localhost",   
        "X-Title": "Agent for project"
        }
    )
    
    return llm

# #Testing
# model = ChatOpenAIModel().invoke("What is AI ?").content
# print(model)