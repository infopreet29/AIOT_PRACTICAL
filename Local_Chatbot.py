import requests
import json

url = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}

# Define your strict rules as a string variable
system_prompt = """
You are a strict, concise college viva examiner. 
Provide highly precise, accurate answers. 
Never write more than 4-5 sentences. Do not use conversational filler.
"""

print("==================================================")
print("  BASIC PROGRAMMING CHATBOT (Type 'exit' to quit)     ")
print("==================================================\n")

while True:
    user_prompt = input("Your Question: ")
    
    if user_prompt.lower() == 'exit':
        print("Good luck for your study! Goodbye.")
        break
        
    if not user_prompt.strip():
        continue

    # Construct the payload including the SYSTEM instructions
    payload = {
        "model": "llama3.2:3b",
        "prompt": user_prompt,
        "system": system_prompt,  # <-- THIS INJECTS THE PERMANENT RULE
        "temperature": 1.0,
        "stream": False
    }

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            print(f"\nAnswer: {response.json()['response']}\n")
        else:
            print(f"\n[Error: Status code {response.status_code}]\n")
    except requests.exceptions.ConnectionError:
        print("\n[Error: Could not connect to Ollama. Is the server running?]\n")
