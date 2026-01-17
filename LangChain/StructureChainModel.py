from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class StructureChainModel(BaseModel):
    dish_name: str
    ingredients: list[str]
    prep_time_minutes: int


parser = PydanticOutputParser(pydantic_object=StructureChainModel)

ollama_model = ChatOllama(model="llama3", temperature=0, base_url="http://localhost:56414")

# 3. Define the Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert chef. Your goal is to extract recipe information from the user's text and format it perfectly as JSON according to the following schema:\n{format_instructions}"),
        ("human", "{user_input}")
    ]
)

prompt_template = prompt.partial(format_instructions=parser.get_format_instructions())

chain = prompt_template | ollama_model | parser

user_input = "Tell me about a quick recipe for scrambled eggs. It should take about 10 minutes."

recipe_object = chain.invoke({"user_input": user_input})

print(f"Dish Name (Type: {type(recipe_object.dish_name)}): {recipe_object.dish_name}")
print(f"Ingredients List (Type: {type(recipe_object.ingredients)}): {recipe_object.ingredients}")
print(f"Preparation Time (Type: {type(recipe_object.prep_time_minutes)}): {recipe_object.prep_time_minutes}")





