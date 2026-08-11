"""
Multi_Agent_Research.py

Demonstrates multi-agent collaboration with AutoGen:
  Researcher (uses a web_search TOOL) -> Writer -> Reviewer

The Reviewer ends the conversation by saying "TERMINATE".
The Writer's article is then saved as a PDF (via ReportLab) and opened.

Install dependencies first:
    pip install autogen-agentchat autogen-ext[openai] ddgs reportlab python-dotenv
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from ddgs import DDGS

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import OpenAIChatCompletionClient

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

load_dotenv()


# ---------------------------------------------------------------------------
# Tool: given to the Researcher agent. AutoGen automatically wraps a plain
# Python function into a callable tool the LLM can invoke.
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return a formatted summary of results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r.get('title', '')}\n   {r.get('href', '')}\n   {r.get('body', '')}")
    return "\n\n".join(lines)


def sanitize_text(text: str) -> str:
    """Keep PDF text safe for ReportLab's default (Latin-1) fonts."""
    if not text:
        return ""
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def save_as_pdf(topic: str, content: str, output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in topic).strip().replace(" ", "_")
    output_path = os.path.join(output_dir, f"{safe_name}_article.pdf")

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(sanitize_text(f"Article: {topic}"), styles["Title"]), Spacer(1, 14)]

    for para in content.split("\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(sanitize_text(para), styles["BodyText"]))
            story.append(Spacer(1, 8))

    doc.build(story)
    return output_path


def open_file(path: str):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')


async def main():
    model_client = OpenAIChatCompletionClient(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "unknown",
        },
        temperature=0,  # more deterministic tool-call formatting
    )

    researcher = AssistantAgent(
        name="Researcher",
        model_client=model_client,
        tools=[web_search],
        system_message=(
            "You research topics using the web_search tool. "
            "Call the tool exactly once, then summarize the key facts "
            "for the Writer in 4-6 bullet points."
        ),
    )

    writer = AssistantAgent(
        name="Writer",
        model_client=model_client,
        system_message=(
            "You write a clear, well-structured short article (3-4 paragraphs) "
            "based on the Researcher's findings. Do not invent facts not "
            "provided by the Researcher."
        ),
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        model_client=model_client,
        system_message=(
            "You review the Writer's article for clarity, accuracy, and flow. "
            "Suggest at most 2 improvements if needed, then on a new line "
            "write exactly: TERMINATE"
        ),
    )

    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat(
        [researcher, writer, reviewer],
        termination_condition=termination,
        max_turns=10,
    )

    topic = input("Enter a topic for the agents to research and write about: ").strip()
    if not topic:
        topic = "Generative AI in education"
        print(f"No input given, using default topic: {topic}")

    final_article = ""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            async for message in team.run_stream(task=f"Research and write a short article about: {topic}"):
                if isinstance(message, TaskResult):
                    print(f"\n--- Conversation ended: {message.stop_reason} ---")
                else:
                    print(f"\n--- {message.source} ---")
                    print(message.content)
                    if message.source == "Writer":
                        final_article = message.content  # keep the latest Writer draft
            break  # success, exit retry loop
        except RuntimeError as e:
            print(f"\n[Attempt {attempt}/{max_attempts}] The model produced a malformed tool call: {e}")
            if attempt == max_attempts:
                print("Giving up after repeated tool-call failures. Try running again.")
            else:
                print("Retrying...")
                await team.reset()

    await model_client.close()

    if final_article:
        pdf_path = save_as_pdf(topic, final_article)
        print(f"\nPDF saved to: {pdf_path}")
        open_file(pdf_path)
    else:
        print("\nNo article was produced, so no PDF was created.")


if __name__ == "__main__":
    asyncio.run(main())
