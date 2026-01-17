OLLAMA_URL = "http://localhost:56368/api/generate"
MODEL = "llama3"   # or any model you installed via `ollama pull`
import logging
import requests

log = logging.getLogger("ToolUseOllama")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


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

def get_weather(city):
    fake_data = {
        "bangalore": "28°C | Cloudy",
        "mumbai": "32°C | Humid",
        "delhi": "26°C | Clear skies",
        "chandigarh": "24°C | Sunny",
        "chennai": "30°C | Rain showers",
        "kolkata": "31°C | Partly cloudy"
    }
    return fake_data.get(city.lower(), "No data")
    
def agent_weather(question):
    react_prompt = f"""
You are an agent with access to:

Tool: weather[city]

If user asks for weather, respond:
Action: weather[city]

Else say:
Action: none

User query: {question}
"""
    query = react_prompt.format(question=question)
    plan = ollama_chat(react_prompt)
    print("Plan:\n", plan)   

    import re
    match = re.search(r"Action:\s*(\w+)\[(.*?)\]", plan)
    if not match:
        return "Could not parse action"
    
    log.debug(match.groups())
    action, param = match.group(1), match.group(2)
    if action == "weather":
        observation = get_weather(param)
    else:
        observation = "No action taken."

    final_prompt = f"""
You said:
{plan}

Tool observation:
{observation}

Give final answer to user.
""" 
    answer = ollama_chat(final_prompt)
    print("\nFinal Answer:\n", answer)
    return answer

    
if __name__ == "__main__":
    agent_weather("What is the weather in Chandigarh today?")

