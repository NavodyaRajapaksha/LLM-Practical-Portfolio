import subprocess
import time
import re
import pyautogui
from datetime import datetime
from pathlib import Path

import platform
import psutil

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

# Date and Time Tool
@tool
def date_time() -> str:
    """Get the current date and time."""

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

# the Notepad tool
@tool
def open_notepad(text: str) -> str:
    """Open Windows Notepad and write the provided text."""

    subprocess.Popen("notepad.exe")

    time.sleep(3)

    pyautogui.write(text, interval=0.1)

    return "Text entered into Notepad."

# the File Search tool
@tool
def search_files(directory: str, pattern: str) -> str:
    """Search files in a directory using a pattern such as *.pdf."""

    folder = Path(directory)

    if not folder.is_dir():
        return "Directory not found."

    files = [
        str(file)
        for file in folder.rglob(pattern)
        if file.is_file()
    ]

    return "\n".join(files) if files else "No files found."

# the File Creation tool
@tool
def create_file(
    directory: str,
    filename: str,
    content: str
) -> str:
    """Create a new text file with the given content."""

    if Path(filename).name != filename:
        return "Invalid filename."

    if not filename.lower().endswith(".txt"):
        return "Only text files are supported."

    path = Path(directory) / filename

    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(content)

        return f"File created: {path}"

    except FileExistsError:
        return "File already exists."

    except OSError as error:
        return str(error)

# the System Information tool
@tool
def system_info() -> str:
    """Retrieve computer operating system, CPU and RAM information."""

    return (
        f"Operating System: {platform.system()}\n"
        f"OS Version: {platform.release()}\n"
        f"Processor: {platform.processor()}\n"
        f"CPU Cores: {psutil.cpu_count()}\n"
        f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB"
    )


# Register tools
tools = [
    calculator,
    date_time,
    open_notepad,
    search_files,
    create_file,
    system_info
]

llm_with_tools = llm.bind_tools(tools)

tool_map = {
    t.name: t for t in tools
}

def run_agent(user_input):

    messages = [
        SystemMessage(
           content="You are a helpful AI agent. "
                   "Use the calculator tool for calculations. "
                   "Use the date_time tool for date and time questions. "
                   "Select the appropriate tool for each request."
                   "user request. Do not invent tool results."
            ),
        HumanMessage(content=user_input)
    ]

    for _ in range(5):

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
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

    print("===== CALCULATOR AI AGENT =====")

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        response = run_agent(user_input)

        print("Agent:", response)