# pip install -U langchain-community bs4

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def answer_from_url(url: str, question: str):
    # 1. Scrape content from the provided URL
    print(f"--- Scraping webpage content from: {url} ---")
    loader = WebBaseLoader(url)
    docs = loader.load()
    web_content = docs[0].page_content

    # 2. Initialize Groq model
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0  # Temperature=0 makes answers strictly factual
    )

    # 3. Create a strict grounding prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict data assistant. You MUST answer user questions based ONLY "
            "on the provided CONTEXT. If the answer cannot be directly derived from the CONTEXT, "
            "you MUST reply with: 'I cannot answer this based on the provided source.' "
            "Do NOT use any outside knowledge."
        ),
        (
            "human",
            "CONTEXT:\n{context}\n\n"
            "QUESTION:\n{question}"
        )
    ])

    # 4. Construct LCEL pipeline
    chain = prompt | llm | StrOutputParser()

    # 5. Execute
    response = chain.invoke({
        "context": web_content,
        "question": question
    })

    print("\n--- AI Response ---")
    print(response)

if __name__ == "__main__":
    #target_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    #user_question = "When was Python first released and who created it?"
    
    target_url = "https://vtpoddar.com/?page_id=19"
    #user_question = "When the vimal tormal poddar bca college started?"
    user_question = "How many colleges are running in Vimal tormal poddar college?"
    answer_from_url(target_url, user_question)
