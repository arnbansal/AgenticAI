from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import logging

log= logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

ollama_model = ChatOllama(model ="llama3", temperature=0.7, base_url="http://localhost:56414")

# response = ollama_model.invoke([
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="Explain the theory of relativity in simple terms."),
# ])
# log.debug("Ollama model invoked successfully.")
# log.debug(response)
# print("🔹 LangChain Ollama Response:")
# print(response.content)