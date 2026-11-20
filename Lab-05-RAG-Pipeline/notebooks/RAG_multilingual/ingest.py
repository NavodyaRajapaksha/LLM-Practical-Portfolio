
from pathlib import Path
import hashlib
import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION = "university_knowledge"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Multilingual E5 embedding integration
# --------------------------------------------------

class MultilingualE5Embeddings(Embeddings):

    def __init__(self):
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )

    def embed_documents(self, texts):
        texts = [
            "passage: " + text
            for text in texts
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(
            "query: " + text,
            normalize_embeddings=True
        )

        return embedding.tolist()


# --------------------------------------------------
# Select embedding model
# --------------------------------------------------

def get_embeddings(mode="multilingual"):

    if mode == "baseline":

        return OllamaEmbeddings(
            model="qwen3-embedding:0.6b"
        )

    return MultilingualE5Embeddings()


# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

def get_vectorstore(mode="multilingual"):

    if mode == "baseline":
        collection = "university_baseline"

    else:
        collection = "university_multilingual"

    embeddings = get_embeddings(mode)

    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(DB_PATH)
    )


# --------------------------------------------------
# Process uploaded PDF
# --------------------------------------------------

def ingest_pdf(uploaded_file, mode="multilingual"):

    pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError(
            "The uploaded file is not a valid PDF."
        )

    file_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    filename = Path(
        uploaded_file.name
    ).name

    filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        filename
    )

    saved_name = (
        f"{file_hash[:12]}_{filename}"
    )

    pdf_path = DATA_DIR / saved_name

    vectorstore = get_vectorstore(mode)

    existing = vectorstore.get(
        where={"file_hash": file_hash},
        limit=1
    )

    if existing["ids"]:

        return (
            f"Already indexed: {filename}"
        )

    pdf_path.write_bytes(pdf_bytes)

    try:

        loader = PyMuPDFLoader(
            str(pdf_path)
        )

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(
            documents
        )

        if not chunks:

            raise ValueError(
                "No extractable text found in PDF."
            )

        for chunk in chunks:

            chunk.metadata["file_hash"] = file_hash

            chunk.metadata["filename"] = filename

            chunk.metadata["source"] = saved_name

            chunk.metadata["embedding_mode"] = mode

        chunk_ids = [

            f"{file_hash}_{i}"

            for i in range(len(chunks))

        ]

        vectorstore.add_documents(

            documents=chunks,

            ids=chunk_ids

        )

        return (

            f"Successfully indexed {filename} "

            f"({len(chunks)} chunks)"

        )

    except Exception:

        partial = vectorstore.get(

            where={"file_hash": file_hash}

        )

        if partial["ids"]:

            vectorstore.delete(
                ids=partial["ids"]
            )

        pdf_path.unlink(
            missing_ok=True
        )

        raise


# --------------------------------------------------
# List stored PDF documents
# --------------------------------------------------

def list_documents():

    return sorted(
        DATA_DIR.glob("*.pdf")
    )
