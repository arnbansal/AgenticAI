from langchain_community.document_loaders import CSVLoader # <-- NEW
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load CSV file with medical records
file_path = r"E:\Github\AgenticAI\RAG\MedicalRecords.csv"

# Verify file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

# Load documents from CSV
loader = CSVLoader(file_path=file_path,
                    csv_args={
                        'delimiter': ',',
                        'quotechar': '"',
                    }
                )

documents = loader.load()

embedding  = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:57592")

#Each row of the CSV is now a LangChain Document. We still chunk for better retrieval.
split_text = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = split_text.split_documents(documents)

#Create FAISS Vector Store
vectorstore  = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- C. RAG Chain Definition ---
#  Initialize Ollama LLM (llama3)
ollama_llm = ChatOllama(model="llama3", temperature=0, base_url="http://localhost:57592")

RAG_PROMPT_TEMPLATE = """
You are a highly specialized medical assistant. Your task is to accurately and concisely answer the question
based ONLY on the medical records provided in the context below. Do not use external knowledge.
If the information is not in the context, state that explicitly.

CONTEXT:
{context}

QUESTION: {question}
"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ollama_llm
    | StrOutputParser()
)

user_query = "What condition does patient P1003 have?"
final_answer = chain.invoke(user_query)

print(f"\n✅ LLM (Ollama) Answer:")
print(final_answer)

user_query_2 = "What were the vitals for P1004 during her last checkup?"


final_answer_2 = chain.invoke(user_query_2)

print(f"\n✅ LLM (Ollama) Answer:")
print(final_answer_2)













