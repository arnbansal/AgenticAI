from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

medical_records = [
    """
    Patient ID: P1001, Name: John Doe
    Date of Visit: 2025-10-20
    Diagnosis: Type 2 Diabetes Mellitus (ICD-10: E11.9)
    Medication: Metformin 500mg, twice daily. 
    Notes: Patient's A1C level is 7.5%. Advised diet modification and increased physical activity.
    """,
    """
    Patient ID: P1001, Name: John Doe
    Date of Visit: 2025-11-15
    Chief Complaint: Persistent joint pain in the knees (Osteoarthritis).
    Test Results: X-ray confirmed moderate cartilage wear.
    Treatment: Started on Celecoxib 200mg daily. Follow-up scheduled in 4 weeks.
    """,
    """
    Patient ID: P1002, Name: Alice Smith
    Date of Visit: 2025-09-01
    Diagnosis: Seasonal Allergic Rhinitis.
    Medication: Cetirizine 10mg PRN (as needed). 
    Notes: Allergies primarily triggered by pollen. No asthma history.
    """,
    """
    Patient ID: P1002, Name: Alice Smith
    Date of Visit: 2025-11-20
    Chief Complaint: Routine follow-up. Blood pressure recorded at 120/80 mmHg.
    Vitals: Healthy heart rate (72 bpm). Patient maintains a healthy lifestyle.
    No changes to current medication recommended.
    """
]

# Convert strings into LangChain Document objects
documents = [Document(page_content=record)for record in medical_records]


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_text = text_splitter.split_documents(documents)


#  Initialize Ollama Embeddings (Used nomic-embed-text)
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url ="http://localhost:57592")


# print("Creating FAISS index (Embedding documents)...")
vectorstore = FAISS.from_documents(split_text, ollama_embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

ollama_llm = ChatOllama(model="llama3", temperature=0, base_url ="http://localhost:57592")

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

user_query = "What medications is patient P1001 currently taking and for what conditions?"

print(f"User Query: {user_query}")

final_answer = chain.invoke(user_query)

print(final_answer)


