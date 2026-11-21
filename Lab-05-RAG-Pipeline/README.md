# Lab Sheet 05 — RAG Pipeline

## Laboratory Session

**Lab 05 — Retrieval-Augmented Generation (RAG) and the University AI Assistant**  

## Objective

For this lab, I built a university assistant that answers questions using PDFs uploaded through a Streamlit chat page. I can add several documents, such as an assessment brief and a course handbook, then ask questions without opening each PDF and searching through it myself. The app also shows the filenames and page numbers it retrieved so I can check the answer against the original documents.

The main idea is **Retrieval-Augmented Generation (RAG)**. When I ask a question, the app searches the uploaded documents for useful passages and gives those passages to the language model before it answers. Adding a new PDF means processing and indexing it. The language model itself does not need to be retrained.

## Tasks and Implementation

### 1. Build the PDF ingestion and retrieval pipeline

1. **Read and split the PDFs.** PyMuPDF extracts text page by page. LangChain splits it into chunks of up to 1,000 characters with 150 characters of overlap, so nearby information can carry across chunk boundaries.
2. **Store the chunks.** Ollama runs Qwen3 Embedding 0.6B to turn each chunk into a vector, a list of numbers representing its text. ChromaDB stores the vectors alongside the text and source details. The app uses a hash of each PDF's contents to skip exact duplicates.
3. **Find and rerank passages.** When I ask a question, Chroma retrieves up to 10 chunks through vector similarity search. A MiniLM cross-encoder scores each question–chunk pair, and the app keeps the three highest-scoring chunks.
4. **Generate and check the answer.** Llama 3.2 3B receives the question and those passages. The prompt asks it to answer from the retrieved text, cite sources using labels such as `[Source 1]`, and say when the information is unavailable. The **View Retrieved Sources** section shows the filename and PDF page number for each retrieved chunk.

| Tool | What it does in this lab |
| --- | --- |
| Python and LangChain | Connect PDF processing, retrieval and answer generation |
| PyMuPDF | Extracts text from PDFs |
| Ollama | Runs the chat and embedding models locally |
| `qwen3-embedding:0.6b` | Creates embeddings for PDF chunks and questions |
| ChromaDB | Stores and searches document chunks |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranks the chunks returned by Chroma |
| `llama3.2:3b` | Writes the answer using the selected passages |
| Streamlit | Provides the PDF uploader, chat and source display |

### 2. Add a Streamlit chat interface and multi-PDF uploads

I added a sidebar where I can select and process several PDFs in one batch. The chat area displays the answer and an expandable list of retrieved sources.

| GitHub folder | Contents |
| --- | --- |
| [RAG_test](notebooks/RAG_test/) | Source for the PDF assistant |
| [RAG_multilingual](notebooks/RAG_multilingual/) | Source for the multilingual group activity |

#### Main files in RAG_test

| File or folder | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and the file to run for this lab |
| `ingest.py` | Saves PDFs, extracts and splits text, checks for duplicates and indexes chunks |
| `rag_pipeline.py` | Retrieves chunks, reranks them and prompts Llama to answer |
| `requirements.txt` | Dependency snapshot from the wider project environment |
| `requirements_rag_backup.txt` | Backup of the dependency snapshot |
| `data/` | Local copies of uploaded PDFs |
| `chroma_db/` | Local Chroma index created when the app uses the vector store |

### 3. Set up and run the PDF assistant in VS Code

#### Step 1 — Open the project folder

Open the folder containing `app.py`, `ingest.py` and `rag_pipeline.py` in VS Code. Open a PowerShell terminal in that folder. Have Python 3.12 and Ollama installed before continuing.

#### Step 2 — Activate the Python environment

If the project already has a working `rag_env`, reuse it. If you are setting up a fresh copy, create it first.

```powershell
py -3.12 -m venv rag_env
```

```powershell
.\rag_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install streamlit langchain-community langchain-ollama langchain-chroma langchain-text-splitters chromadb pymupdf sentence-transformers
```

The install command covers the packages used by this app. If you want to try the project's full dependency snapshot, use `python -m pip install -r requirements.txt`. The supplied snapshot is UTF-16 and pins packages from the wider environment. If pip cannot read it or cannot resolve a pinned version, use the app-specific install above in a clean environment.

To check that Python can import ChromaDB:

```powershell
python -c "import chromadb; print('ChromaDB installed successfully')"
```

#### Step 3 — Download the Ollama models

Open Ollama, then run these commands in PowerShell. If both models already appear in `ollama list`, you can skip the downloads.

```powershell
ollama pull llama3.2:3b
ollama pull qwen3-embedding:0.6b
ollama list
```

If the app cannot connect to Ollama, make sure the Ollama desktop app is running. If the service is stopped, you can start it with `ollama serve` in a separate terminal. The reranker may also download its model from Hugging Face the first time the app starts.

#### Step 4 — Start the Streamlit app

```powershell
python -m streamlit run app.py
```

Open the local address printed in the terminal, usually `http://localhost:8501`.

#### Step 5 — Upload and test your PDFs

1. In the **Knowledge Base** sidebar, select one or more text-based PDFs.
2. Click **Process Documents** and wait for the indexing messages.
3. Check that the PDFs appear under **Stored Documents**.
4. Ask a question that you know one of the PDFs answers.
5. Expand **View Retrieved Sources** and check the cited pages yourself.

If you reopen VS Code later, activate `rag_env` and run the Streamlit command again. Previously indexed documents stay available as long as you keep the same `data/` and `chroma_db/` folders.

### 4. Explore multilingual retrieval in the group activity

The group activity explored how well the assistant could retrieve useful evidence when a question was asked in **English, Sinhala or Tamil**, especially when the relevant document was in a different language. The results show a separate version of the app with a question-language selector and three retrieval strategies.

| Strategy | What the comparison is meant to test |
| --- | --- |
| Baseline | Search with the question as written and use that result as the reference point. |
| Multilingual | Test whether multilingual embeddings can connect a question with relevant passages across languages. |
| Translated Query | Translate the question into English before retrieval and see whether it finds better evidence in English documents. |

We used the app’s question-language and retrieval-strategy controls to try the approaches. The group version is in [RAG_multilingual](notebooks/RAG_multilingual/).

## Results and Evidence

[Open the lab evidence screenshots](Screenshots/).

| Test | What the lab results show |
| --- | --- |
| Multiple PDFs | The Streamlit sidebar shows a computing award handbook and an in-class maths activity PDF stored together. |
| A question covered by the documents | Asking what to do in the in-class activity produced a summary of the smart-city maths tasks. I would still check the brief for the exact part numbers and timings. |
| A question outside the available context | In the early test, the assistant reported that it could not find information on machine learning in the retrieved passages. |
| Checking sources | The UI displays retrieved document names and page numbers so I can verify the answer. A retrieved page may contain related information without supporting every statement in the response. |

| Group activity evidence | What I observed |
| --- | --- |
| Sinhala and Tamil questions | The screenshots show questions in Sinhala and Tamil receiving answers in English. |
| Retrieval strategy comparison | The same Sinhala question about AI was tried with Baseline, Multilingual and Translated Query selected. The results are a qualitative record of the demo. |

<!-- To embed any screenshot here, copy its exact filename from GitHub and replace YOUR-FILE.png in the line below. Use %20 for spaces, and keep the folder name capitalized.
![Description of the evidence](Screenshots/YOUR-FILE.png)
-->

## Observations

The assistant could summarize the in-class activity from the uploaded material. Some task labels and details in the generated summary needed checking against the brief. Having the PDF name and page number made that check easier. Reranking can move useful passages higher, but it can only score the chunks Chroma already found.

The multilingual screenshots show that a question can be asked in Sinhala or Tamil and receive a response in English. Retrieval quality and answer language are separate things to measure. To compare the three strategies fairly, I would use the same PDF set and questions, record the expected answer and supporting page for each question, then compare evidence retrieval, answer accuracy and response time. The screenshots alone do not establish a winning strategy.

### Troubleshooting

| If you see this | Try this |
| --- | --- |
| `ModuleNotFoundError` | Confirm that `python -m pip --version` points to `rag_env`, then install the package with that environment active. |
| Ollama connection failure or model missing | Check that Ollama is running and verify both exact model tags with `ollama list`. |
| The app is slow to start | Let the first reranker download finish. Model loading and reranking can also take longer on a CPU. |
| “No university documents are indexed yet” | Select PDFs and click **Process Documents**. Check for an indexing error in the sidebar before asking again. |
| “No extractable text found in PDF” | Open the PDF and try selecting its text. For an image-only scan, use an OCR-processed copy with selectable text. |
| The answer is missing details or looks wrong | Open **View Retrieved Sources**, check the relevant pages and try a more specific question that includes the module, assessment or document name. |

## Learning Outcomes

- I learned how to turn several PDFs into a searchable knowledge base using text chunks, embeddings and ChromaDB.
- I learned how to use Ollama for local embedding and answer generation and Streamlit for the upload and chat interface.
- I learned how similarity search and reranking work together and how to check a generated answer against its source page.
- I learned how translated queries and multilingual embeddings can help search across languages and how to set up a fair comparison.
- I learned that the basic app displays earlier chat messages, while the RAG pipeline receives only the latest question and its retrieved passages. For follow-up questions, I can repeat the relevant context.
