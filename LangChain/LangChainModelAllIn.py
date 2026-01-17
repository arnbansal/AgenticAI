from langchain_core.prompts import ChatPromptTemplate as promptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

class LangChainModelAllIn:
    
    @staticmethod
    def ExecuteLangChainModelwithChain():
        try:

            ollama = ChatOllama(model="llama3", temperature="0.7", base_url="http://localhost:56414")

            template = "Write a short Movie review {movie_name}"
            prompt = promptTemplate.from_template(template)
            chain = prompt | ollama | StrOutputParser()
            user_input ={"movie_name":"Munna Bhai MBBS"}
            result = chain.invoke (user_input)
            print("🔹 LangChain Model with Chain Response:")
            print(result)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    LangChainModelAllIn.ExecuteLangChainModelwithChain()

