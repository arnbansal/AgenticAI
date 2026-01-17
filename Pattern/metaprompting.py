import requests
OLLAMA_URL ="http://localhost:56368/api/generate"
def ollama_client(message):
   payload ={
         "model": "llama3",
         "prompt": message,
         "options": {"temperature": 0.7, "num_predict": 300},
         "stream": False
   }
   client = requests.post(OLLAMA_URL, json=payload)
   return client.json()["response"]

   
def meta_prompting(question):
   meta_prompt = f"""
You are a prompt-engineering assistant.
Improve the user prompt by making it:
– clearer
– more detailed
– with constraints
– with better structure

User prompt:
\"\"\"{question}\"\"\"
"""
   answer = ollama_client(meta_prompt)
   print("🔹 Meta-Prompting Result:")
   print(answer)


if __name__ == "__main__":
   meta_prompting("How to become a ai engineer?")