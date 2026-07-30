# pip install -U pypdf
import os
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Reads a PDF file from your local disk and extracts all text."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    return full_text

def answer_from_pdf(pdf_path: str, question: str):
    print(f"--- Reading PDF from: {pdf_path} ---")
    pdf_text = extract_text_from_pdf(pdf_path)

    # Initialize Groq LLM with temperature=0.0 for strict fact-checking
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )

    # Strict grounding prompt prevents hallucinations
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a strict document assistant. You MUST answer user questions based ONLY "
            "on the provided PDF CONTENT. If the answer cannot be directly found in the content, "
            "you MUST reply with: 'Information not found in the provided PDF file.' "
            "Do NOT make up any answers or use outside knowledge."
        ),
        (
            "human",
            "PDF CONTENT:\n{pdf_text}\n\n"
            "QUESTION:\n{question}"
        )
    ])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "pdf_text": pdf_text,   #[:8000],  # Sending the extracted text content
        "question": question
    })

    print("\n--- Answer from PDF ---")
    print(response)

if __name__ == "__main__":
    # Replace with the actual path to your PDF file
    local_pdf_path = r"D:\Ollama\IC UNIT 2.pdf"  # Use 'r' before path string for Windows backslashes
    
    #user_question = "What is the main summary of this document?"
    #user_question = "What is computer architecture?"
    #user_question = "GPU Kya Hai?"
    user_question = "What is Ollama?"
    
    answer_from_pdf(local_pdf_path, user_question)

    
