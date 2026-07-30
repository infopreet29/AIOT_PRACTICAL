import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables from .env file
load_dotenv()


def answer_from_custom_text(custom_text: str, question: str) -> str:
    # Initialize Groq LLM with temperature=0.0 for strict facts
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )

    # System instruction locks down the AI to your text
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer questions using ONLY the text provided under SOURCE TEXT. "
            "If the answer is not mentioned in the source text, state: "
            "'Information not available in the provided text.'"
        ),
        ("human", "SOURCE TEXT:\n{text}\n\nQUESTION:\n{question}")
    ])

    # Build and execute the chain
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": custom_text, "question": question})


if __name__ == "__main__":
    # Custom grounded context
    my_text = """
    Company XYZ was founded in 2021 by Sarah Jenkins. 
    It produces eco-friendly water bottles made from 100% recycled ocean plastic. 
    The company is headquartered in Seattle, Washington.
    """

    print("--- Test 1 (Information present) ---")
    #q1 = "Where is Company XYZ located?"
    q1 = "કંપની XYZ ક્યાં આવેલી છે?"
    print(f"Question: {q1}")
    print(f"Answer: {answer_from_custom_text(my_text, q1)}\n")

    print("--- Test 2 (Information missing) ---")
    q2 = "How much do the water bottles cost?"
    print(f"Question: {q2}")
    print(f"Answer: {answer_from_custom_text(my_text, q2)}")
