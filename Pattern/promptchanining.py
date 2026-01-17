
import requests
Ollam_URL= "http://localhost:56368/api/generate"


def ollama_chat(prompt):
    payload ={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(Ollam_URL, json=payload)
    return response.json()["response"]


def chain_summarize(text):
    try:
        p = f"Summarize the following text in 5 bullet points:\n\n{text}"
        return ollama_chat(p)
    except Exception as e:  
        return f"Error: {e}"
    


def chain_generate_quiz(summary):
    try:
        p = f"Generate 5 MCQ quiz questions based on this summary:\n\n{summary}"
        return ollama_chat(p)
    except Exception as e:  
        return f"Error: {e}"

if __name__ == "__main__":
    # Run chain
    text = """
        Large Language Models are deep neural networks trained on large corpora of text...
    """
summary = chain_summarize(text)
quiz = chain_generate_quiz(summary)

print("Summary:\n", summary)
print("\nQuiz Questions:\n", quiz)