# practical_3_ai_assistant.py
"""
Unit 2 Practical (iii): Build a Prompt-Based AI Assistant using LangChain

A simple conversational assistant that:
  - Uses a ChatPromptTemplate to define its persona/behavior
  - Keeps track of conversation history so it can hold a multi-turn chat
  - Takes the user's questions dynamically at runtime (a loop, not hardcoded)
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from a .env file in the same folder

LLM_MODEL = "llama-3.3-70b-versatile"  # any current Groq-hosted model name

llm = ChatGroq(
    model=LLM_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
)

# The assistant's persona / behavior is defined once, here.
ASSISTANT_PERSONA = (
    "You are a friendly and knowledgeable AI assistant for BCA students. "
    "Explain concepts clearly, use simple examples, and keep answers concise "
    "unless the student asks for more detail."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", ASSISTANT_PERSONA),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# LangChain Expression Language (LCEL) chain: prompt -> model
chain = prompt | llm


def main():
    history = []  # holds HumanMessage/AIMessage objects across turns

    print("===== AI Assistant (LangChain + Ollama) =====")
    print("Ask me anything. Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit", ""):
            print("Assistant: Goodbye!")
            break

        response = chain.invoke({
            "history": history,
            "question": question,
        })

        answer = response.content
        print(f"Assistant: {answer}\n")

        # Update conversation history so follow-up questions have context
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))


if __name__ == "__main__":
    main()
