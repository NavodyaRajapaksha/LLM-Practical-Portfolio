
import streamlit as st

from ingest import (
    ingest_pdf,
    list_documents
)

from rag_pipeline import ask_university


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multilingual University AI Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title(
    "🎓 Multilingual University AI Assistant"
)

st.caption(
    "University Knowledge Retrieval in "
    "English, Sinhala and Tamil"
)


# --------------------------------------------------
# Sidebar: Document management
# --------------------------------------------------

with st.sidebar:

    st.header("📚 Knowledge Base")

    uploaded_files = st.file_uploader(

        "Upload university PDF documents",

        type=["pdf"],

        accept_multiple_files=True

    )

    if st.button(
        "Process Documents",
        type="primary"
    ):

        if uploaded_files:

            progress = st.progress(0)

            total = len(uploaded_files) * 2

            completed = 0

            for uploaded_file in uploaded_files:

                for mode in [
                    "baseline",
                    "multilingual"
                ]:

                    try:

                        with st.spinner(
                            f"Indexing {uploaded_file.name} "
                            f"using {mode}..."
                        ):

                            message = ingest_pdf(
                                uploaded_file,
                                mode
                            )

                        st.success(message)

                    except Exception as e:

                        st.error(
                            f"{uploaded_file.name}: {e}"
                        )

                    completed += 1

                    progress.progress(
                        completed / total
                    )

        else:

            st.warning(
                "Please select PDF files first."
            )

    st.divider()

    st.subheader("Stored Documents")

    documents = list_documents()

    if documents:

        st.write(
            f"PDF files: {len(documents)}"
        )

        for doc in documents:

            st.write(
                f"📄 {doc.name}"
            )

    else:

        st.info(
            "No PDFs stored yet."
        )


# --------------------------------------------------
# Retrieval settings
# --------------------------------------------------

st.header(
    "💬 Ask the University Assistant"
)

language = st.selectbox(

    "Question language",

    [

        "English",

        "Sinhala",

        "Tamil"

    ]

)

retrieval_mode = st.selectbox(

    "Retrieval strategy",

    [

        "Baseline",

        "Multilingual",

        "Translated Query"

    ]

)


mode_mapping = {

    "Baseline": "baseline",

    "Multilingual": "multilingual",

    "Translated Query": "translated"

}


# --------------------------------------------------
# Chat interface
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            with st.expander(
                "View Retrieved Sources"
            ):

                for i, source in enumerate(

                    message["sources"],

                    start=1

                ):

                    st.write(

                        f"[Source {i}] "

                        f"{source['document']} "

                        f"— Page {source['page']}"

                    )


question = st.chat_input(

    "Ask a question about the university..."

)


if question:

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })

    with st.chat_message("user"):

        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching university documents..."
        ):

            try:

                result = ask_university(

                    question,

                    mode=mode_mapping[
                        retrieval_mode
                    ]

                )

                st.markdown(
                    result["answer"]
                )

                if result["sources"]:

                    with st.expander(
                        "View Retrieved Sources"
                    ):

                        for i, source in enumerate(

                            result["sources"],

                            start=1

                        ):

                            st.write(

                                f"[Source {i}] "

                                f"{source['document']} "

                                f"— Page {source['page']}"

                            )

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": result["answer"],

                    "sources": result["sources"]

                })

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )
