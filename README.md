# MCA Chatbot (Chandigarh University)

An intelligent, production-ready conversational agent designed for MCA (AI & ML) students at Chandigarh University. 

This chatbot uses an advanced **hybrid-retrieval system** consisting of Sentence-Transformer semantic intent matching and FAISS-powered Document Retrieval (RAG) to provide fast, accurate, and localized answers to student queries. It runs entirely on local machine resources—zero external API costs!

## ✨ Key Features

- **Semantic Intent Matching**: Uses `all-MiniLM-L6-v2` to understand the *meaning* of queries, not just keywords.
- **Typo-Tolerant Resolution**: Features a deterministic Levenshtein-distance spelling corrector (`rapidfuzz`) to smoothly handle misspellings before semantic matching.
- **Hybrid RAG Fallback**: If a query is highly specific (e.g., syllabus details), it falls back to a local FAISS vector index of embedded documents (`docs/`) to extract context.
- **Data Privacy & PII Masking**: Student identity data (`students.json`) is securely parsed and decoupled from chat logs. Sensitive PII like emails and roll numbers are NEVER echoed back in chat bubbles.
- **Production WSGI Deployment**: Powered by the robust Waitress WSGI server ensuring concurrent requests are handled safely, with custom thread-safe logging architectures replacing standard unscalable file writes.
- **Text-To-Speech (TTS)**: Leverages backend `gTTS` mapping directly into the browser's Web Audio API for a fast, responsive conversational voice.
- **Automated Gap Analysis**: Includes `analyze_gaps.py` to continuously learn what users are asking by automatically clustering "unmatched" queries into new intent suggestions.

## 🚀 Getting Started

### Prerequisites
- **Python 3.14+**
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Awinashkr31/Chat_bot.git
   cd Chat_bot
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Student Data**: Ensure you have a valid `students.json` in the root directory. You can copy the schema from `students.example.json` if needed.

### Running the Server

**For Production (Recommended):**
The application is securely wrapped in a Waitress WSGI server with thread-safe file handlers and HTTP security headers.
```bash
python run_prod.py
```

**For Development (Debug Mode):**
To leverage hot-reloading and debug traces:
```bash
python app.py
```

Once running, navigate to `http://127.0.0.1:5000` in your browser.

## 🧪 Testing

The repository contains a safety net of regression tests covering tricky NLP confusions (e.g., "fee" vs "fees") and API response schemas. 

To run the tests:
```bash
python -m pytest
```

## 🛠 Project Structure

- `app.py`: Core routing, PII handling, and semantic intent logic.
- `run_prod.py`: Entrypoint for the production Waitress server.
- `retrieval.py`: The FAISS-powered Document Retrieval / RAG system.
- `config.py`: Environment and constant definitions.
- `intents.json`: The database of categorized intents and responses.
- `static/` & `templates/`: Frontend UI (HTML, CSS, JS) boasting a dynamic CU-branded interface with animated typing indicators and clarification states.
- `analyze_gaps.py`: A utility script for analyzing unresolved user logs to suggest new intents.

## 🤝 Contributing
Please see `CONTRIBUTING.md` for details on how to add intents, update the document index, and adhere to our PII privacy policies.