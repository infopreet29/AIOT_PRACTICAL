# practical_2_prompt_comparison.py
"""
Unit 2 Practical (ii): Prompt Comparison Experiments

Runs the SAME user question through several DIFFERENT prompt strategies
against the same local Ollama model, and prints the responses side by
side so students can compare how prompt design changes the output.

Strategies compared:
  A. Direct / no-instruction prompt
  B. Instruction-based prompt (explicit style guidance)
  C. Role-based prompt (assistant given a persona)
  D. Chain-of-thought style prompt (asked to reason step by step)
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from a .env file in the same folder

LLM_MODEL = "llama-3.3-70b-versatile"  # any current Groq-hosted model name

llm = ChatGroq(
    model=LLM_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)


# ---------- Prompt strategies ----------

direct_prompt = PromptTemplate.from_template("{question}")

instruction_prompt = PromptTemplate.from_template(
    "Answer the following question in 2-3 clear, simple sentences.\n\nQuestion: {question}"
)

role_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a patient, expert teacher explaining things to a beginner."),
    ("human", "{question}"),
])

cot_prompt = PromptTemplate.from_template(
    "Answer the following question. Think through it step by step before giving "
    "your final answer.\n\nQuestion: {question}\n\nStep-by-step reasoning:"
)


STRATEGIES = {
    "A": ("Direct (no instruction)", direct_prompt),
    "B": ("Instruction-based", instruction_prompt),
    "C": ("Role-based (teacher persona)", role_prompt),
    "D": ("Chain-of-thought (step by step)", cot_prompt),
}


def run_strategy(key, question):
    label, template = STRATEGIES[key]
    if isinstance(template, ChatPromptTemplate):
        messages = template.format_messages(question=question)
        response = llm.invoke(messages)
    else:
        prompt = template.format(question=question)
        response = llm.invoke(prompt)
    return label, response.content


def compare_all(question):
    print(f"\n########## Comparing prompts for: \"{question}\" ##########")
    for key in STRATEGIES:
        label, answer = run_strategy(key, question)
        print(f"\n--- Strategy {key}: {label} ---")
        print(answer)
    print("\n" + "#" * 60)


def main():
    print("===== Prompt Comparison Experiment =====")
    print("Available strategies:")
    for key, (label, _) in STRATEGIES.items():
        print(f"  {key}. {label}")
    print("\nType a question to compare all strategies at once.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit", ""):
            print("Exiting.")
            break
        compare_all(question)


if __name__ == "__main__":
    main()
