from SimpleLangChainModel import ollama_model
from PromptLangChain import prompt
from langchain_core.output_parsers import StrOutputParser

class LangChainModel:

    @staticmethod
    def InvokeModels():

        response = prompt | ollama_model | StrOutputParser()

        result = response.invoke({"cuisine": "Indian", "ingredient": "lentils"})
        print(result)


if __name__ == "__main__":
    LangChainModel.InvokeModels()

