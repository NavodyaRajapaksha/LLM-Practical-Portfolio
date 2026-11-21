import subprocess
import time
import re
import pyautogui
import sqlite3

from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


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

# Google Calendar configuration
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events"
]

BASE_DIR = Path(__file__).resolve().parent

CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"


# Google Calendar authentication
def get_calendar_service():

    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(
            str(TOKEN_FILE), SCOPES
        )

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE),
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(
            creds.to_json(),
            encoding="utf-8"
        )

    return build(
        "calendar",
        "v3",
        credentials=creds
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

    try:
        from rag_pipeline import ask_university

        result = ask_university(query)

        answer = result["answer"]

        sources = "\n".join(
            f"Source: {source['document']}, Page: {source['page']}"
            for source in result["sources"]
        )

        return f"{answer}\n\n{sources}"

    except Exception as error:
        return f"RAG search failed: {error}"

# Google Calendar - Today's Schedule
@tool
def check_today_schedule() -> str:
    """Retrieve today's events from Google Calendar."""

    service = get_calendar_service()

    timezone = ZoneInfo("Asia/Colombo")

    now = datetime.now(timezone)

    start = now.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    end = start + timedelta(days=1)

    events_result = service.events().list(
        calendarId="primary",
        timeMin=start.isoformat(),
        timeMax=end.isoformat(),
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    events = events_result.get("items", [])

    if not events:
        return "You have no events scheduled for today."

    schedule = []

    for event in events:

        event_start = event["start"].get(
            "dateTime",
            event["start"].get("date")
        )

        title = event.get("summary", "Untitled Event")

        schedule.append(
            f"{title} - {event_start}"
        )

    return "\n".join(schedule)

# Google Calendar - Create Event
@tool
def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str
) -> str:
    """Create a Google Calendar event.
    Use YYYY-MM-DDTHH:MM:SS for start_time and end_time.
    """

    timezone = ZoneInfo("Asia/Colombo")

    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)

        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone)

        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone)

        if end <= start:
            return "End time must be after start time."

    except ValueError:
        return "Invalid date or time format."

    # Ask for user approval
    print("\n===== EVENT DETAILS =====")
    print("Title:", title)
    print("Start:", start.isoformat())
    print("End:", end.isoformat())

    approval = input("Create this event? (yes/no): ")

    if approval.strip().lower() != "yes":
        return "Event creation cancelled."

    service = get_calendar_service()

    event = {
        "summary": title,
        "start": {
            "dateTime": start.isoformat(),
            "timeZone": "Asia/Colombo"
        },
        "end": {
            "dateTime": end.isoformat(),
            "timeZone": "Asia/Colombo"
        }
    }

    created_event = service.events().insert(
        calendarId="primary",
        body=event
    ).execute()

    return (
        f"Event created successfully: {title}\n"
        f"Link: {created_event.get('htmlLink')}"
    )


tools = [
    calculator,
    search_memory,
    search_university_knowledge,
    check_today_schedule,
    create_calendar_event
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
                    "Use search_memory for personal information. "
                    "Use search_university_knowledge for university questions. "
                    "Use check_today_schedule to retrieve today's events. "
                    "Use create_calendar_event to create calendar events. "
                    "Ask for missing event details before calling the tool. "
                    "Use Asia/Colombo as the timezone. "
                    f"Current local date and time: "
                    f"{datetime.now(ZoneInfo('Asia/Colombo')).isoformat()}. "
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
    print("INTELLIGENT UNIVERSITY AI AGENT")
    print("CALCULATOR TOOL")
    print("SHORT-TERM MEMORY")
    print("LONG-TERM SQLITE MEMORY")
    print("WORKING MEMORY")
    print("UNIVERSITY KNOWLEDGE BASE")
    print("GOOGLE CALENDAR INTEGRATION")
    print("================================")

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        response = run_agent(user_input)

        print("Agent:", response)
