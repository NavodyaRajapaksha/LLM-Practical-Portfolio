# Lab Sheet 06 - Building a Tool-Using AI Agent with Ollama

## Objective

Build a Python command-line agent that uses LangChain and Ollama's `llama3.2:3b` model to select and run tools for desktop tasks, conversation memory, university document retrieval, and Google Calendar.

## Tasks and Implementation

| Part | Script | Implementation |
| --- | --- | --- |
| A – Calculator | [calculator_agent.py](LabSheet6/calculator_agent.py) | Registered a calculator tool and used PyAutoGUI to enter calculations in Windows Calculator. |
| A – Multiple tools | [multi_tool_agent.py](LabSheet6/multi_tool_agent.py) | Added tools for date and time, Notepad, file search, text file creation, and system information. |
| B – Memory | [memory_agent.py](LabSheet6/memory_agent.py) | Used conversation history for short-term context and SQLite for persistent memory. |
| C – RAG | [rag_agent.py](LabSheet6/rag_agent.py) | Connected the agent to Lab 5's `ask_university` function to retrieve information from university documents. |
| D – Google Calendar | [calendar_agent.py](LabSheet6/calendar_agent.py) | Used Google OAuth to authorize tools that check today's schedule and create events. |

### Run on Windows

Start Ollama and open PowerShell in the `Lab-06-AI-Agents` folder with your Python environment active.

```powershell
cd LabSheet6
python -m pip install -r requirements_rag_backup.txt
python -m pip install pyautogui pygetwindow google-api-python-client google-auth-oauthlib
ollama pull llama3.2:3b
python calculator_agent.py
```

Try `Calculate 25 * 4` and type `exit` to close the agent. Run the other scripts from the same folder.

- **RAG.** Follow the [Lab 5 setup](../Lab-05-RAG-Pipeline/) to prepare the document index and make `rag_pipeline.py` available to the Lab 6 scripts.
- **Google Calendar.** Enable the Calendar API and place a desktop OAuth client file named `credentials.json` beside `calendar_agent.py`. Authorize access on the first run. Keep `credentials.json` and `token.json` local.

## Results and Evidence

See the [Screenshots folder](Screenshots/) for the documented runs.

| Part | Evidence |
| --- | --- |
| A | Calculator operations for addition, subtraction, division, and multiplication, plus outputs from the five additional tools. |
| B | Recall within the conversation and retrieval of information saved in SQLite. |
| C | A university knowledge-base response, plus demonstrations of memory search and calculation. |
| D | A schedule lookup and a newly created Google Calendar event. |


## Observations

- The calculator and Notepad tools depend on the Windows desktop because the scripts interact with those applications directly.
- Conversation history provides context during a session. SQLite allows saved information to be retrieved after the agent restarts.
- University answers depend on the Lab 5 document index. Calendar access requires completing the browser authorization step on the first run.

## Learning Outcomes

- Register Python functions as tools and handle tool calls from a local language model.
- Add short-term conversation context and persistent SQLite memory to an agent.
- Reuse a RAG pipeline and connect an agent to Google Calendar through OAuth.
