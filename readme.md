# MultiPDF Chat App 📚

## Introduction
------------
The MultiPDF Chat App is a Python application that allows you to chat with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes Ollama LLM (local) and Cohere embeddings to generate accurate answers to your queries. The app will only respond to questions related to the loaded PDFs and provides source references with page numbers.

## Features
------------
- 📄 Upload and process multiple PDF documents simultaneously
- 💬 Interactive chat interface with conversation history
- 🔍 Source attribution showing which PDF and page number answers come from
- 🧠 Powered by Ollama (qwen2.5:0.5b) - lightweight local LLM
- 🎯 Cohere embeddings for accurate semantic search
- 📊 FAISS vector store for efficient document retrieval
- 🔒 Privacy-focused: runs locally with Ollama

## How It Works
------------

The application follows these steps to provide responses to your questions:

1. **PDF Loading**: The app reads multiple PDF documents and extracts their text content page by page.

2. **Text Chunking**: The extracted text is divided into smaller chunks (5000 characters) with overlap to maintain context.

3. **Embeddings**: The application uses Cohere's embed-english-light-v3.0 model to generate vector representations of the text chunks.

4. **Vector Storage**: Text chunks and their embeddings are stored in a FAISS vector database for efficient similarity search.

5. **Similarity Matching**: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

6. **Response Generation**: The selected chunks are passed to Ollama LLM (running locally), which generates a response based on the relevant content of the PDFs.

7. **Source Attribution**: The app displays which PDF and page number each answer is derived from.

## Dependencies and Installation
----------------------------
To install the MultiPDF Chat App, please follow these steps:

1. **Install Ollama**: 
   - Download and install Ollama from [ollama.ai](https://ollama.ai)
   - Pull the required model:
     ```bash
     ollama pull qwen2.5:0.5b
     ```
   - Verify Ollama is running at `http://localhost:11434`

2. Clone the repository to your local machine.

3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
```commandline
GEMINI_API_KEY=your_secrit_api_key
```

## Usage
-----
To use the MultiPDF Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the Google API key to the `.env` file.

2. Run the app using the Streamlit CLI:
   ```bash
   streamlit run app.py
   ```

3. The application will launch in your default web browser at `http://localhost:8501`.

4. Upload POllama (qwen2.5:0.5b) - Local LLM
  - *Alternative: Google Gemini 2.0 Flash (available in code, commented out)*
- **Embeddings**: Cohere embed-english-light-v3.0
- **Vector Store**: FAISS
- **PDF Processing**: PyPDF2
- **Framework**: LangChain

## Configuration Options
------------
The app supports two LLM options:

### Current: Ollama (Local)
```python
llm = Ollama(
    model="qwen2.5:0.5b",
    temperature=0.3,
    base_url="http://localhost:11434"
)
```
**Pros**: Privacy-focused, no API costs, runs offline  
**Cons**: Requires local resources, needs Ollama installation

### Alternative: Google Gemini (Cloud)
Uncomment the Google Gemini section in `app.py` and comment out the Ollama section:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
```
**Pros**: More powerful, no local resources needed  
**Cons**: Requires API key, costs per use, needs internet
5. Ask questions:
   - Type your question in the text input field
   - The AI will respond based on the content of your uploaded PDFs
   - View source references to see which documents and pages were used

## Technology Stack
------------
- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: Cohere embed-english-light-v3.0
- **Vector Store**: FAISS
- **PDF Processing**: PyPDF2
- **Framework**: LangChain

## Project Structure
------------
```
Chat with multiple PDFs/
├── app.py                 # Main application file
├── htmlTemplates.py       # HTML/CSS templates for chat UI
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in repo)
├── docs/                  # Documentation and assets
│   ├── AI.jpg            # Bot avatar image
│   └── Human.webp        # User avatar image
└── readme.md             # This file
```

## Contributing
------------
This repository is intended for educational purposes. Feel free to fork and enhance the app based on your own requirements.

## License
-------
The MultiPDF Chat App is released under the [MIT License](https://opensource.org/licenses/MIT).
