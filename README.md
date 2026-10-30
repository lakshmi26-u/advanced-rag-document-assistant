# Advanced RAG Document Assistant

An AI-powered document question-answering application that allows users
to upload PDF documents, ask questions, and receive answers grounded
in document content with source citations.

## Features

- Upload and process PDF documents
- Extract text from PDFs
- Split text into overlapping chunks
- Generate semantic embeddings
- Store and search document vectors
- Ask questions in natural language
- Generate answers using Google Gemini
- Display source documents and PDF page numbers
- Filter retrieval by selected document
- Handle questions when sufficient evidence is unavailable

## Technology Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- all-MiniLM-L6-v2
- ChromaDB
- Google Gemini API
- Google Colab

## Architecture

PDF Documents
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
ChromaDB Vector Storage
    ↓
Semantic Retrieval
    ↓
Gemini Answer Generation
    ↓
Answer with Source Citations

## How It Works

1. A user selects an existing document or uploads a PDF.
2. The application extracts text from the PDF.
3. The text is divided into overlapping chunks.
4. Sentence Transformers generates embeddings for the chunks.
5. ChromaDB stores the embeddings and associated metadata.
6. A user's question is converted into an embedding.
7. Relevant document chunks are retrieved using vector similarity.
8. Gemini generates an answer using the retrieved evidence.
9. The application displays the answer with source citations.

## Current Project Configuration

- Embedding model: all-MiniLM-L6-v2
- Embedding dimensions: 384
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Similarity distance threshold: 0.70
- Handbook chunks: 1524
- Uploaded sample policy chunks: 6

## Setup Notes

1. Install the required Python packages.
2. Configure the Gemini API key securely.
3. Ensure the ChromaDB database is available.
4. Run the Streamlit application.

Never commit API keys, passwords, or other secrets to GitHub.

## Testing

The application has been tested for:

- Answering questions using a handbook
- Retrieving information from an uploaded sample policy
- Filtering results by document
- Displaying source document and PDF page citations
- Responding when sufficient evidence is unavailable

## Future Improvements

- Improve retrieval quality and relevance
- Add automated evaluation of retrieval and answer quality
- Improve error handling and user feedback
- Add document management and deletion
- Evaluate performance on larger document collections
- Deploy the application for public demonstration

## Disclaimer

This is an educational portfolio project. Answers should be
verified against the cited source documents before being used
for important decisions.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Advanced_RAG_Document_Assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API key

Create a .env file using .env.example as a reference.
Add GEMINI_API_KEY=your_actual_api_key

Never upload your real API key to GitHub.

### 4. Prepare the vector database

Place or generate the ChromaDB database in the chroma_db folder next to app.py.

### 5. Run the application

```bash
streamlit run app.py
```
