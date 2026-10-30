
import os
import re
import uuid
import hashlib
import streamlit as st
import chromadb
import pymupdf

from sentence_transformers import SentenceTransformer
from google import genai


# --------------------------------
# 1. Application configuration
# --------------------------------

st.set_page_config(
    page_title="Advanced RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Advanced RAG Document Assistant")
st.caption("Ask questions and get answers from your PDF documents.")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

COLLECTION_NAME = "employee_handbook"
UPLOADED_COLLECTION_NAME = "uploaded_documents_v2"

THRESHOLD = 0.70


# --------------------------------
# 2. Load AI components
# --------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_database():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION_NAME)


@st.cache_resource
def load_uploaded_database():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=UPLOADED_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


@st.cache_resource
def load_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    return genai.Client(api_key=api_key)


embedding_model = load_embedding_model()
collection = load_database()
uploaded_collection = load_uploaded_database()
gemini_client = load_gemini()


# --------------------------------
# 3. PDF processing
# --------------------------------

def extract_uploaded_pdf(uploaded_file):
    extracted_pages = []

    with pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    ) as document:

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            text = re.sub(r"\s+", " ", text).strip()

            if text:
                extracted_pages.append({
                    "page": page_number,
                    "text": text
                })

    return extracted_pages


def chunk_pages(pages, chunk_size=1000, overlap=200):
    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page["text"]

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"]
                })

    return chunks


def store_uploaded_pdf(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    existing = uploaded_collection.get(
        where={"file_hash": file_hash},
        limit=1
    )

    if existing["ids"]:
        return 0, "This PDF has already been processed."

    pages = extract_uploaded_pdf(uploaded_file)

    if not pages:
        return 0, "No extractable text found in this PDF."

    chunks = chunk_pages(pages)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False
    )

    uploaded_collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "document_name": uploaded_file.name,
                "page": chunk["page"],
                "file_hash": file_hash
            }
            for chunk in chunks
        ]
    )

    return len(chunks), "PDF processed successfully."


# --------------------------------
# 4. Get available documents
# --------------------------------

def get_document_names():
    names = ["University of Southern Indiana Handbook"]

    stored = uploaded_collection.get(
        include=["metadatas"]
    )

    if stored["metadatas"]:
        uploaded_names = sorted(set(
            metadata["document_name"]
            for metadata in stored["metadatas"]
            if metadata.get("document_name")
        ))

        names.extend(uploaded_names)

    return ["All documents"] + names


# --------------------------------
# 5. Retrieve relevant chunks
# --------------------------------

def retrieve_relevant_chunks(
    question,
    top_k=5,
    selected_document="All documents"
):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    relevant_chunks = []

    if selected_document in [
        "All documents",
        "University of Southern Indiana Handbook"
    ]:

        handbook_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        for i, text in enumerate(
            handbook_results["documents"][0]
        ):

            distance = handbook_results["distances"][0][i]

            if distance <= THRESHOLD:
                metadata = handbook_results["metadatas"][0][i]

                relevant_chunks.append({
                    "text": text,
                    "page": metadata["page"],
                    "document_name":
                        "University of Southern Indiana Handbook",
                    "distance": distance
                })

    if selected_document in [
        "All documents",
        *get_document_names()[2:]
    ]:

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k
        }

        if selected_document != "All documents":
            query_args["where"] = {
                "document_name": selected_document
            }

        uploaded_results = uploaded_collection.query(
            **query_args
        )

        if uploaded_results["documents"]:
            for i, text in enumerate(
                uploaded_results["documents"][0]
            ):

                distance = uploaded_results["distances"][0][i]

                if distance <= THRESHOLD:
                    metadata = uploaded_results["metadatas"][0][i]

                    relevant_chunks.append({
                        "text": text,
                        "page": metadata["page"],
                        "document_name": metadata["document_name"],
                        "distance": distance
                    })

    relevant_chunks.sort(
        key=lambda chunk: chunk["distance"]
    )

    return relevant_chunks[:top_k]


# --------------------------------
# 6. Generate answer using Gemini
# --------------------------------

def generate_answer(question, chunks):

    if not chunks:
        return "I couldn't find sufficient evidence in the selected documents."

    context = ""

    for chunk in chunks:
        context += (
            f"[Document: {chunk['document_name']}, "
            f"PDF page: {chunk['page']}]\n"
            f"{chunk['text']}\n\n"
        )

    prompt = f"""
You are a careful AI document assistant.

Follow these rules:
1. Answer only using the supplied document evidence.
2. Never invent facts or policies.
3. Read section headings carefully.
4. If documents contain conflicting information, explain the differences.
5. Do not combine unrelated statements into one answer.
6. Cite important claims using [Source: document name, PDF page N].
7. If the evidence is insufficient, clearly say so.
8. Give clear and concise answers.

DOCUMENT EVIDENCE:

{context}

USER QUESTION:

{question}
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response.text or "The model returned an empty response."

    except Exception as error:
        error_message = str(error)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            return (
                "Gemini API quota has been exceeded. "
                "The relevant document evidence was retrieved successfully, "
                "but answer generation is temporarily unavailable. "
                "Please try again later."
            )

        return f"Gemini answer generation failed: {error_message}"


# --------------------------------
# 7. Sidebar and document upload
# --------------------------------

with st.sidebar:

    st.header("📚 Document Assistant")

    st.write("Handbook chunks:", collection.count())
    st.write("Uploaded PDF chunks:", uploaded_collection.count())

    st.write(f"Similarity threshold: {THRESHOLD}")

    st.divider()

    st.subheader("Upload PDF documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "Process PDFs",
        disabled=not uploaded_files
    ):

        for uploaded_file in uploaded_files:

            with st.spinner(
                f"Processing {uploaded_file.name}..."
            ):

                try:
                    chunk_count, message = store_uploaded_pdf(
                        uploaded_file
                    )

                    if chunk_count > 0:
                        st.success(
                            f"{uploaded_file.name}: "
                            f"{chunk_count} chunks stored."
                        )
                    else:
                        st.info(
                            f"{uploaded_file.name}: {message}"
                        )

                except Exception as error:
                    st.error(
                        f"{uploaded_file.name}: {error}"
                    )

        st.rerun()

    st.divider()

    st.subheader("Choose a document")

    document_names = get_document_names()

    selected_document = st.selectbox(
        "Select document to search",
        document_names
    )

    st.divider()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------
# 8. Chat interface
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            source_text = ", ".join(
                f"{name}, PDF page {page}"
                for name, page in message["sources"]
            )

            st.caption("Sources: " + source_text)


question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner("Searching your documents..."):

                chunks = retrieve_relevant_chunks(
                    question,
                    selected_document=selected_document
                )

            if not chunks:

                answer = (
                    "I couldn't find sufficient evidence "
                    "in the selected documents."
                )

            else:

                with st.spinner("Generating your answer..."):

                    answer = generate_answer(
                        question,
                        chunks
                    )

            st.markdown(answer)

            source_list = sorted(set(
                (
                    chunk["document_name"],
                    chunk["page"]
                )
                for chunk in chunks
            ))

            if source_list:
                source_text = ", ".join(
                    f"{name}, PDF page {page}"
                    for name, page in source_list
                )

                st.caption("Retrieved sources: " + source_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_list
            })

        except Exception as error:

            st.error(f"Something went wrong: {error}")
