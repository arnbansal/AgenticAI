import ollama
import json
import time


def send_completion(prompt, model=DEFAULT_MODEL, temperature=0.7, max_tokens=300):
    """
    Sends a prompt to a local Ollama model and returns text output.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

few_shot_template = """
You are an assistant that classifies movie reviews as Positive or Negative.

Examples:
Review: "I loved the movie. The story was touching and the acting superb."
Label: Positive

Review: "Boring, too long, and predictable."
Label: Negative

Now classify the following review:
Review: "{review}"
Label:
"""

def classify_review(review):
    prompt = few_shot_template.format(review=review)
    return send_completion(prompt, temperature=0.0, max_tokens=20)


# ---------------------------------------------
# Demo
# ---------------------------------------------
if __name__ == "__main__":
    print("🔹 LLM Fundamentals Demo — Ollama Version")
    
    review = "The plot was amazing and the visuals were stunning."
    print("\nFew-shot classification:")
    print(classify_review(review))
    
   
