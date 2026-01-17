import requests
import json

OLLAMA_URL = "http://localhost:56368/api/generate"
MODEL = "llama3"

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Error evaluating expression."
    
def ollama_chat(prompt):
    try:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
        r = requests.post(OLLAMA_URL, json=payload)
        return r.json()["response"]
    except Exception as e:
        return f"Error: {e}"
    
def react(question):
    reasoning_prompt = f"""
You are using ReAct.
Think step by step, then decide an action.

Available Tools:
1. calculator[expression]

Your response format:
Thought: ...
Action: <tool>[<input>]
"""
    thoughts = ollama_chat(reasoning_prompt + "\nUser query: " + question)
    print("LLM Thoughts + Action:\n", thoughts)

    # 2. Extract the action
    import re
    match = re.search(r"Action:\s*(\w+)\[(.*?)\]", thoughts)
    if not match:
        return "No tool used. Final answer: " + thoughts
    
    action, param = match.group(1), match.group(2)
    if action == "calculator":
        tool_output = calculator(param)
    else:
        tool_output = "Unknown tool"
    
    print("\nTool Output:", tool_output)

    final_prompt = f"""
        You previously said:
        {thoughts}

        Observation from tool:
        {tool_output}

        Give final answer now.
        """

    final_answer = ollama_chat(final_prompt)
    print("\nFinal Answer:")
    print(final_answer)
    return final_answer

if __name__ == "__main__":
    react("What is 11 * 11?")