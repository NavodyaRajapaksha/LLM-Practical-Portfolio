import subprocess
import time
import re
import pyautogui
import sqlite3

from rag_pipeline import ask_university

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Create the memory database
def initialize_database():

    with sqlite3.connect("memory.db") as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                information TEXT
            )
        """)


initialize_database()

# Save user information
def save_memory(information):

    with sqlite3.connect("memory.db") as conn:

        conn.execute(
            "INSERT INTO memories (information) VALUES (?)",
            (information,)
        )

    print("[MEMORY SAVED]", information)



@tool
def calculator(expression: str) -> str:
    """Perform a mathematical calculation using Windows Calculator."""

    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")
    expression = expression.replace(" ", "")

    if not re.fullmatch(r"\d+(\.\d+)?[+\-*/]\d+(\.\d+)?", expression):
        return "Provide a simple calculation such as 25*4."

    subprocess.Popen("calc.exe")

    time.sleep(3)

    pyautogui.press("esc")
    pyautogui.write(expression, interval=0.2)
    pyautogui.press("enter")

    return f"Calculation entered in Windows Calculator: {expression}"

# Memory Search Tool
@tool
def search_memory(query: str) -> str:
    """Search previously stored information about the user."""

    with sqlite3.connect("memory.db") as conn:

        results = conn.execute(
            "SELECT information FROM memories "
            "WHERE information LIKE ? ORDER BY id DESC",
            (f"%{query}%",)
        ).fetchall()

    if results:
        return "\n".join(row[0] for row in results)

    return "No matching memories found."

# RAG Search Tool
@tool
def search_university_knowledge(query: str) -> str:
    """Search university documents for academic information,
    degree programs, modules, regulations and requirements."""

    print("[EXECUTING] Searching university knowledge:", query)

    result = ask_university(query)

    answer = result["answer"]

    sources = "\n".join(
        f"Source: {source['document']}, Page: {source['page']}"
        for source in result["sources"]
    )

    return f"{answer}\n\n{sources}"


tools = [
    calculator,
    search_memory,
    search_university_knowledge
]

llm_with_tools = llm.bind_tools(tools)

tool_map = {
    t.name: t for t in tools
}

# Short-term memory
chat_history = []


def run_agent(user_input):

    # Detect and save user information
    if user_input.lower().startswith(
        ("my name is", "i am studying", "i like", "i live in")
    ):
        save_memory(user_input)

    messages = [
        SystemMessage(
            content="You are an intelligent university AI agent. "
                    "Use calculator for mathematical calculations. "
                    "Use search_memory for previously stored user information. "
                    "Search memory using keywords such as name or studying. "
                    "Use search_university_knowledge for university-related "
                    "questions, degree programs, modules and regulations. "
                    "Include document sources and page numbers when available. "
                    "Prefer the most recently stored personal information. "
                    "Do not invent information or tool results."
        ),
        *chat_history,
        HumanMessage(content=user_input)
    ]

    for _ in range(5):

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            chat_history.extend([
                HumanMessage(content=user_input),
                response
            ])

            return response.content

        for call in response.tool_calls:

            tool_name = call["name"]
            arguments = call["args"]

            print("[TOOL SELECTED]", tool_name)
            print("[TOOL ARGUMENTS]", arguments)

            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(arguments)
            else:
                result = "Unknown tool."

            print("[TOOL RESULT]", result)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

    return "Tool execution limit reached."

if __name__ == "__main__":

    print("================================")
    print("RAG POWERED UNIVERSITY AI AGENT")
    print("CALCULATOR TOOL")
    print("SHORT-TERM MEMORY")
    print("LONG-TERM SQLITE MEMORY")
    print("WORKING MEMORY")
    print("UNIVERSITY KNOWLEDGE BASE")
    print("================================")

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        response = run_agent(user_input)

        print("Agent:", response)