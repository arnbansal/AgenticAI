from langchain_core.prompts import ChatPromptTemplate as promptTemplate
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

try:
    prompt = promptTemplate.from_messages([
         ("system", "You are a world-renowned chef that specializes in {cuisine}."),
        ("human", "What is the best dish to prepare with {ingredient}?"),
    ])
    # log.debug("Prompt template created successfully.")
    # log.debug(prompt)
   # prompt_text  = prompt.invoke({"cuisine": "Maxican", "ingredient": "pasta"})
    # log.debug(prompt_text)
    # print("🔹 LangChain Prompt Template Result:")
    # print(prompt_text.to_string())
except Exception as e:
    print(f"Error: {e}")