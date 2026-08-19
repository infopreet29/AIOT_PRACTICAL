r"""
Interactive Self-Correcting Python Debugger Agent (LangGraph + Groq)
=====================================================================

Paste in a buggy Python program (and, if it needs runtime input, the
input values it should read). The agent runs it, and if it fails or
gives the wrong output, asks the LLM to fix it — repeating until it
runs cleanly or a max-iteration limit is hit.

Setup
-----
    pip install -U langgraph langchain-groq langchain-core langgraph-checkpoint-sqlite python-dotenv

    Create a file named ".env" in the SAME FOLDER as this script
    containing one line:

        GROQ_API_KEY=your_key_here

Requirements on PATH:
    Python : python (or python3)

Run
---
    python interactive_debugger_agent.py

Then paste your code, paste any runtime input it needs (or leave
blank), optionally give the expected output, and the agent takes it
from there.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Load .env from the same directory as this script (not the current
# working directory), so it works no matter where/how you launch it.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("debugger_agent")

MODEL_NAME = "openai/gpt-oss-120b"   # Groq model id
DEFAULT_MAX_ITERATIONS = 5
EXEC_TIMEOUT_SECONDS = 10
CHECKPOINT_DB_PATH = "agent_checkpoints.sqlite"


def get_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a '.env' file next to this "
            "script containing: GROQ_API_KEY=your_key_here "
            "(see the module docstring)."
        )
    return ChatGroq(model=MODEL_NAME, api_key=api_key, temperature=0, max_retries=2, timeout=60)


# --------------------------------------------------------------------------
# Python runner
# --------------------------------------------------------------------------

def run_python(code: str, work_dir: str, stdin_input: str) -> dict:
    """
    Runs the given Python source in a scratch directory and returns a
    dict with stdout / stderr / returncode / timed_out.

    stdin_input is ALWAYS passed explicitly (defaulting to ""). This
    is important: without an explicit input=, a program that calls
    input() blocks forever waiting on a stream nothing will ever
    write to, and every attempt just times out instead of actually
    running.
    """
    path = os.path.join(work_dir, "main.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    python_exe = shutil.which("python") or shutil.which("python3") or sys.executable

    try:
        result = subprocess.run(
            [python_exe, "main.py"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
            input=stdin_input,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": (
                f"Timed out after {EXEC_TIMEOUT_SECONDS}s running the program.\n"
                "If this program calls input(), make sure you supplied "
                "enough input lines — a program that tries to read more "
                "input than it was given will hang exactly like this."
            ),
            "returncode": -1,
            "timed_out": True,
        }
    except FileNotFoundError as e:
        return {
            "stdout": "",
            "stderr": f"Required tool not found on PATH: {e}",
            "returncode": -1,
            "timed_out": False,
        }


def extract_code_block(text: str) -> str:
    match = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# --------------------------------------------------------------------------
# Agent state
# --------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    original_code: str     # the buggy code the user supplied (kept for reference)
    code: str               # current candidate (starts == original_code)
    stdin_input: str        # exact text fed to the program's stdin on every run
    expected_output: str    # optional: exact stdout the program SHOULD produce.
                              # "" means "not checked" — only crashes are
                              # treated as bugs.
    stdout: str
    stderr: str
    success: bool
    iteration: int
    max_iterations: int


# --------------------------------------------------------------------------
# Graph nodes
# --------------------------------------------------------------------------

FIXER_SYSTEM_PROMPT = """You are an expert Python debugger.
You will be given a Python program that fails to run correctly, along
with the exact error output (and, if relevant, the runtime input it
was given). Rules:
- Diagnose the ROOT CAUSE from the error message, don't guess randomly.
- Output ONLY one ```python fenced code block containing the FULL
  corrected program — no explanation outside the code block.
- Preserve the original program's intent and structure as much as
  possible; make the minimal change needed to fix the bug.
- The corrected program must be complete and runnable on its own.
- If runtime input is shown below, make sure your program reads
  exactly that many values, in that order/format (e.g. matching
  input() calls to the input layout) — a mismatch there is a common
  bug in its own right.
"""


def execute_code_node(state: AgentState) -> dict:
    logger.info(
        "Attempt %d/%d: running Python code...",
        state["iteration"] + 1,
        state["max_iterations"],
    )
    with tempfile.TemporaryDirectory() as work_dir:
        result = run_python(state["code"], work_dir, state.get("stdin_input", ""))
    ran_cleanly = result["returncode"] == 0 and not result["timed_out"]

    expected = state.get("expected_output", "")
    if ran_cleanly and expected:
        success = result["stdout"].strip() == expected.strip()
        if not success:
            result["stderr"] = (
                f"Program ran without error but produced the wrong output.\n"
                f"Expected:\n{expected.strip()}\n\nGot:\n{result['stdout'].strip()}"
            )
    else:
        success = ran_cleanly

    if success:
        logger.info("Run succeeded.")
    else:
        logger.warning("Run failed (returncode=%s, timed_out=%s).", result["returncode"], result["timed_out"])

    return {
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "success": success,
        "iteration": state["iteration"] + 1,
    }


def fix_code_node(state: AgentState) -> dict:
    llm = get_llm()
    logger.info("Asking the model to fix the Python code...")

    system = SystemMessage(content=FIXER_SYSTEM_PROMPT)
    expected_note = (
        f"\nExpected stdout:\n{state['expected_output'].strip()}\n"
        if state.get("expected_output") else ""
    )
    stdin_note = (
        f"\nRuntime input given to the program (in order):\n{state['stdin_input']}\n"
        if state.get("stdin_input") else "\nThis program is given no runtime input (empty stdin).\n"
    )
    human = HumanMessage(
        content=(
            f"Buggy Python program:\n```python\n{state['code']}\n```\n"
            f"{stdin_note}"
            f"{expected_note}\n"
            f"stdout:\n{state['stdout']}\n\nstderr:\n{state['stderr']}\n\n"
            "Fix it and return the complete corrected program."
        )
    )

    response: AIMessage = llm.invoke([system, human])
    fixed_code = extract_code_block(response.content)

    return {"messages": [human, response], "code": fixed_code}


def route_after_execution(state: AgentState) -> str:
    if state["success"]:
        return "success"
    if state["iteration"] >= state["max_iterations"]:
        return "give_up"
    return "retry"


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

def build_graph(checkpointer):
    graph = StateGraph(AgentState)
    graph.add_node("execute_code", execute_code_node)
    graph.add_node("fix_code", fix_code_node)
    graph.set_entry_point("execute_code")
    graph.add_conditional_edges(
        "execute_code",
        route_after_execution,
        {"retry": "fix_code", "success": END, "give_up": END},
    )
    graph.add_edge("fix_code", "execute_code")
    return graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------
# Programmatic API
# --------------------------------------------------------------------------

def debug_code(
    buggy_code: str,
    stdin_input: str = "",
    expected_output: str = "",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    thread_id: str = "debug-thread-1",
) -> AgentState:
    initial_state: AgentState = {
        "messages": [],
        "original_code": buggy_code,
        "code": buggy_code,
        "stdin_input": stdin_input,
        "expected_output": expected_output,
        "stdout": "",
        "stderr": "",
        "success": False,
        "iteration": 0,
        "max_iterations": max_iterations,
    }
    config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH) as checkpointer:
        app = build_graph(checkpointer)
        final_state = app.invoke(initial_state, config=config)

    return final_state


# --------------------------------------------------------------------------
# Interactive CLI
# --------------------------------------------------------------------------

def _read_multiline(prompt: str) -> str:
    print(prompt)
    print('(Paste your text, then type a line containing only  ###END###  and press Enter)')
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "###END###":
            break
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print("Interactive Self-Correcting Python Debugger")
    print("=" * 70)

    code = _read_multiline("\nPaste your Python code below.")
    if not code.strip():
        print("No code entered. Exiting.")
        return

    needs_input = input(
        "\nDoes this program read runtime input (input())? [y/N]: "
    ).strip().lower().startswith("y")
    stdin_input = ""
    if needs_input:
        stdin_input = _read_multiline(
            "\nPaste the input values, one per line, in the exact order the "
            "program will read them."
        )

    wants_expected = input(
        "\nDo you want to specify the EXACT expected output, to catch bugs "
        "that run without crashing but give the wrong answer? [y/N]: "
    ).strip().lower().startswith("y")
    expected_output = ""
    if wants_expected:
        expected_output = _read_multiline("\nPaste the exact expected output.")

    max_iter_raw = input(
        f"\nMax fix attempts [default {DEFAULT_MAX_ITERATIONS}]: "
    ).strip()
    max_iterations = int(max_iter_raw) if max_iter_raw.isdigit() else DEFAULT_MAX_ITERATIONS

    print("\nRunning...\n")
    final = debug_code(
        code,
        stdin_input=stdin_input,
        expected_output=expected_output,
        max_iterations=max_iterations,
        thread_id="interactive-session",
    )

    print("\n" + "=" * 70)
    print(f"Success: {final['success']}  (after {final['iteration']} attempt(s))")
    print("=" * 70)
    print("\nFinal code:\n")
    print(final["code"])
    print("\nProgram output:\n")
    print(final["stdout"] or "(no stdout)")
    if not final["success"]:
        print("\nLast error:\n")
        print(final["stderr"])


if __name__ == "__main__":
    main()
