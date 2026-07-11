import requests

url = "https://api.groq.com/openai/v1/chat/completions"

# Paste your GROQ API key here
GROQ_API_KEY = "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"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}",
}

# Define your strict rules as a string variable
system_prompt = """
You are a strict, concise college viva examiner.
Provide highly precise, accurate answers.
Never write more than 4-5 sentences. Do not use conversational filler.
"""

# Multi-turn chat history, starts with the system instructions
messages = [{"role": "system", "content": system_prompt}]

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

    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 1.0,
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            print(f"\nAnswer: {answer}\n")
            messages.append({"role": "assistant", "content": answer})
        else:
            print(f"\n[Error: Status code {response.status_code} - {response.text}]\n")
            messages.pop()  # drop the last user message so the failed turn isn't remembered
    except requests.exceptions.ConnectionError:
        print("\n[Error: Could not connect to GROQ API. Check your internet connection.]\n")
        messages.pop()
