from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv(r"B:\Gen AI\langchain\.env")
api_key = os.getenv("GROQ_API_KEY")
print(f"Groq API Key: {api_key is not None}")
print("Groq API Key loaded")

# Use compound-mini as the model for testing
model = ChatGroq(model="groq/compound-mini", groq_api_key=api_key)

# Test the model
try:
    response = model.invoke("Say hello")
    print("Model test successful:", response.content)
except Exception as e:
    print(f"Model test failed: {e}")

# Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser = StrOutputParser()

# Create chain
chain = prompt_template | model | parser

# App definition
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces"
)

# Adding chain routes 
add_routes(
    app,
    chain,
    path="/chain" 
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)