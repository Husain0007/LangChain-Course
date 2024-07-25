from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title = "Lanchain Server",
    version = "1.0",
    description="A simple API Server"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)
model = ChatOpenAI()
# Ollama Llama3
llm = Ollama(model="llama3")

prompt1 = ChatPromptTemplate.from_template("Write an essay about {topic} in 100 words.")
prompt2 = ChatPromptTemplate.from_template("Write a poem about {topic} in 100 words.")

add_routes(
    app,
    prompt1|model,
    path = "/essay"
)

add_routes(
    app,
    prompt2|llm,
    path = "/poem"
)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port = 8000)