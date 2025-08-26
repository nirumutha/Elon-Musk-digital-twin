🚀 AI Digital Twin: Elon Musk
This project is a sophisticated, multi-modal AI Digital Twin of Elon Musk, built using a Retrieval-Augmented Generation (RAG) pipeline. It ingests public interviews and writings to answer questions in his specific voice and style, providing both text and audio responses.

This application serves as a comprehensive portfolio piece demonstrating a full-cycle AI engineering project, from resilient data collection and processing to a polished, interactive web application.

Live Demo URL: https://elon-digital-twin-final.streamlit.app/

✨ Key Features & Solutions
This project goes beyond a standard RAG implementation by incorporating several features designed to solve real-world engineering challenges:

1. Upgraded & Resilient Architecture
The project was initially built with ChromaDB, but it ran into a critical, well-known sqlite3 version conflict during deployment to Streamlit Cloud.

The Solution: I re-engineered the AI core, upgrading the vector store to FAISS (developed by Meta AI). This not only solved the deployment bug by removing the problematic dependency but also made the application more robust and performant. This demonstrates the ability to pivot and upgrade a system's architecture to overcome environmental constraints.

2. Multi-Modal Output
To create a more engaging user experience, the application generates both text and audio.

Text Generation: Uses GPT-4-turbo for high-quality, context-aware responses.

Audio Generation: Integrates OpenAI's Text-to-Speech (TTS) API to convert the text answer into spoken words, which can be played directly in the user interface.

3. AI Transparency & Explainability
The user interface is designed to demystify the AI's process.

Source Citation: For every answer, the application clearly cites which source document(s) from the knowledge base were used as the primary context. This demonstrates a commitment to building "grounded" and verifiable AI systems.

🛠️ Tech Stack
AI Orchestration: LangChain

LLMs & APIs: OpenAI (GPT-4-turbo, Whisper, TTS, Embeddings)

Vector Database: FAISS (upgraded from ChromaDB)

Web Framework: Streamlit

Core Language: Python

⚙️ Setup and Installation
Clone the repository:

git clone https://github.com/nirumutha/Elon-Musk-digital-twin.git
cd Elon-Musk-digital-twin

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Download NLTK data:

python download_nltk.py

Set up your environment variables:

Create a .env file in the root directory.

Add your OpenAI API key: OPENAI_API_KEY="sk-..."

Run the application:

First, build the knowledge base: python data_collector_v2.py

Then, build the vector database: python ai_core_v2.py

Finally, launch the app: streamlit run app_v2.py
