"""
Agentic AI Competitor Analysis Generator (Groq free-tier version)
===================================================================
Domain   : Electric Vehicle (EV) manufacturers in India
Problem  : Generate a comprehensive competitor analysis for EV manufacturers in India.
Pattern  : Multi-agent pipeline (Planner -> Researcher -> Analyst ->
           Synthesizer -> Validator), each stage a separate LLM call
           with its own role/system prompt, sharing a common "state"
           dict (a simple blackboard architecture).

Difference from the Anthropic version:
- Uses Groq's free API (OpenAI-compatible) instead of Anthropic.
- Groq does not host a built-in web_search tool, so this script
  implements a small custom "search_web" tool using DuckDuckGo
  (free, no API key) and a manual tool-calling loop -- a good
  teaching example of how function calling actually works under
  the hood, versus the hosted tool in the Anthropic version.

Setup:
    pip install groq ddgs
    export GROQ_API_KEY="your-groq-key-here"   # from console.groq.com
    python ev_competitor_agentic_analysis_groq.py

Notes for classroom use:
- Groq's free tier is rate-limited (roughly 30 requests/min, ~1000
  requests/day). This script makes ~2-3 calls per manufacturer, so
  keep the manufacturer list short (5-7) to stay comfortably inside
  the limit.
- Groq only serves open-source models (no GPT/Claude/Gemini), so
  quality will differ from the Anthropic version -- useful itself
  as a discussion point on model choice trade-offs.
"""

import os
import json
import time
import datetime
from dotenv import load_dotenv
from groq import Groq, BadRequestError
from ddgs import DDGS

load_dotenv()  # reads variables from a .env file in the same folder into os.environ

MODEL = "openai/gpt-oss-120b"
client = Groq()  # now finds GROQ_API_KEY via os.environ, loaded from .env above

# ---------------------------------------------------------------------
# Custom tool: web search via DuckDuckGo (free, no API key needed)
# ---------------------------------------------------------------------
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current information and return short snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}


def search_web(query: str, max_results: int = 5) -> str:
    """Runs a live DuckDuckGo search and returns text snippets with sources."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        return "\n".join(
            f"- {r.get('title')}: {r.get('body')} (source: {r.get('href')})"
            for r in results
        )
    except Exception as e:
        return f"Search failed: {e}"


# ---------------------------------------------------------------------
# Low-level helper: one agent "turn" with an optional tool-calling loop
# ---------------------------------------------------------------------
def run_agent(system_prompt: str, user_prompt: str, use_search: bool = True,
              max_tokens: int = 1200, max_tool_rounds: int = 3) -> str:
    """
    Sends a request to Groq. If the model wants to call search_web,
    executes the search and feeds the result back, looping until the
    model produces a final text answer (or max_tool_rounds is hit).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tools = [SEARCH_TOOL_SCHEMA] if use_search else None

    for round_num in range(max_tool_rounds):
        kwargs = dict(model=MODEL, max_tokens=max_tokens, messages=messages)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            # Llama models on Groq occasionally emit a malformed function-call
            # string instead of a proper structured tool call ("tool_use_failed").
            # This is usually transient -- retry once, then fall back to
            # answering from the model's own knowledge without tools.
            if "tool_use_failed" in str(e) and tools:
                print("  (tool call malformed, retrying without search for this step...)")
                tools = None
                continue
            raise

        choice = response.choices[0].message

        if choice.tool_calls:
            messages.append(choice)
            for call in choice.tool_calls:
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = search_web(args.get("query", ""))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
            continue  # let the model see the search results and respond

        return choice.content or ""

    return "No final answer produced within tool-call limit."


# ---------------------------------------------------------------------
# Sub-task 1: Identify the manufacturers to analyze
# ---------------------------------------------------------------------
def identify_manufacturers(state: dict) -> dict:
    print("[Agent 1/5] Planner: identifying manufacturers to cover...")
    system = (
        "You are a market research planner specializing in India's "
        "automotive industry. Be precise and current."
    )
    user = (
        "List the top 5-7 electric vehicle manufacturers currently "
        "active in India's passenger EV market (include both domestic "
        "players like Tata Motors and Mahindra, and foreign entrants "
        "like MG, Hyundai, BYD, etc. where relevant). "
        "Return ONLY a JSON array of company names, nothing else."
    )
    raw = run_agent(system, user)
    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        manufacturers = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        manufacturers = [line.strip("-* ") for line in raw.splitlines() if line.strip()]
    state["manufacturers"] = manufacturers
    return state


# ---------------------------------------------------------------------
# Sub-task 2: Gather sales figures and pricing per manufacturer
# ---------------------------------------------------------------------
def gather_market_data(state: dict) -> dict:
    print("[Agent 2/5] Researcher: gathering sales & pricing data...")
    system = (
        "You are a research analyst. Use the search_web tool to find "
        "current, sourced facts. Always note the source of any figure "
        "you report. If you cannot verify a number, say so explicitly."
    )
    market_data = {}
    for company in state["manufacturers"]:
        user = (
            f"For '{company}' in the Indian EV passenger vehicle market, "
            "find: (a) recent sales volume trend, (b) key model line-up "
            "and price range (INR), (c) approximate market share if "
            "available. Summarize in 4-6 sentences and cite sources."
        )
        market_data[company] = run_agent(system, user)
    state["market_data"] = market_data
    return state


# ---------------------------------------------------------------------
# Sub-task 3: Analyze customer sentiment per manufacturer
# ---------------------------------------------------------------------
def analyze_reviews(state: dict) -> dict:
    print("[Agent 3/5] Analyst: analyzing customer sentiment...")
    system = (
        "You are a customer insights analyst. Use the search_web tool "
        "to find recent owner reviews, forum discussions, and news "
        "coverage. Distinguish recurring themes from one-off complaints."
    )
    sentiment = {}
    for company in state["manufacturers"]:
        user = (
            f"Based on recent owner reviews and news coverage of '{company}' "
            "EVs in India, summarize: common praise points, common "
            "complaints (e.g. service network, range, software), and "
            "overall sentiment trend. Keep it to 3-5 sentences."
        )
        sentiment[company] = run_agent(system, user)
    state["sentiment"] = sentiment
    return state


# ---------------------------------------------------------------------
# Sub-task 4: Synthesize everything into one report
# ---------------------------------------------------------------------
def synthesize_report(state: dict) -> dict:
    print("[Agent 4/5] Synthesizer: building comparative report...")
    system = (
        "You are a senior strategy consultant. Write clearly and "
        "concisely for an executive audience. Use Markdown."
    )
    user = (
        "Using the research notes below, produce a competitor analysis "
        "report on EV manufacturers in India with:\n"
        "1. A one-paragraph executive summary.\n"
        "2. A Markdown comparison table (Manufacturer | Key Models | "
        "Price Range | Sales Trend | Market Share | Strengths | Weaknesses).\n"
        "3. A short 'Key Takeaways' bullet list.\n\n"
        f"MARKET DATA:\n{json.dumps(state['market_data'], indent=2)}\n\n"
        f"CUSTOMER SENTIMENT:\n{json.dumps(state['sentiment'], indent=2)}"
    )
    state["report_draft"] = run_agent(system, user, use_search=False, max_tokens=2500)
    return state


# ---------------------------------------------------------------------
# Sub-task 5: Validate the draft before it's finalized
# ---------------------------------------------------------------------
def validate_report(state: dict) -> dict:
    print("[Agent 5/5] Validator: checking claims are backed by research...")
    system = (
        "You are a fact-checking editor. You do not invent new facts. "
        "You only check whether claims in the draft are consistent with "
        "the supplied source notes, and flag anything unsupported."
    )
    user = (
        "Review this draft report against the source notes. List any "
        "unsupported claims or inconsistencies as bullet points. If "
        "everything is adequately supported, say so explicitly.\n\n"
        f"DRAFT REPORT:\n{state['report_draft']}\n\n"
        f"SOURCE NOTES:\n{json.dumps(state['market_data'], indent=2)}"
    )
    state["validation_notes"] = run_agent(system, user, use_search=False, max_tokens=800)
    return state


# ---------------------------------------------------------------------
# Orchestrator: runs the pipeline end-to-end
# ---------------------------------------------------------------------
def run_pipeline() -> str:
    state = {}
    for step in (identify_manufacturers, gather_market_data,
                 analyze_reviews, synthesize_report, validate_report):
        state = step(state)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"EV_India_Competitor_Analysis_Groq_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(state["report_draft"])
        f.write("\n\n---\n\n## Validator Notes\n\n")
        f.write(state["validation_notes"])

    print(f"\nDone. Report saved to: {filename}")
    return filename


if __name__ == "__main__":
    run_pipeline()
