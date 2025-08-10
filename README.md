# 🤖 ADGM-Compliant Corporate Agent

This project is an intelligent AI-powered legal assistant designed to review, validate, and help prepare documentation for business incorporation and compliance within the Abu Dhabi Global Market (ADGM) jurisdiction.
The agent leverages a Retrieval-Augmented Generation (RAG) pipeline to ensure its analysis is grounded in official ADGM legal documents, providing accurate, context-aware feedback. It runs entirely on a local machine, ensuring complete data privacy.

---

## ✨ Key Features

*   **Document Checklist Verification:** Automatically checks if all mandatory documents for a legal process (like Company Incorporation) are present in the user's submission.
*   **AI-Powered Red Flag Detection:** Uses a local RAG pipeline with **Llama 3** to analyze `.docx` files clause-by-clause.
*   **Detects Critical Issues:** Identifies non-compliance with ADGM templates, incorrect legal jurisdictions, missing clauses, improper formatting, and ambiguous language.
*   **Legal Citations:** Cites the specific ADGM rules or regulations that apply to a flagged issue, providing authority for its findings.
*   **Actionable Suggestions:** Offers clear, legally compliant suggestions to fix identified problems.
*   **Automated Commenting:** Generates a downloadable `.docx` version of the original file with all findings inserted as contextual comments at the relevant paragraphs.
*   **Structured JSON Report:** Outputs a comprehensive JSON summary of the entire analysis, suitable for logging or downstream processing.

---

## 🛠️ Technology Stack

*   **UI Framework:** Streamlit
*   **Backend Logic:** Python 3.9+
*   **AI Orchestration:** LangChain
*   **Generative LLM:** Llama 3 (running locally via **Ollama**)
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (local)
*   **Vector Store:** FAISS (local)
*   **Document Handling:** `python-docx`, `pypdf`

---

## 🚀 Setup and Usage Instructions

Follow these steps to set up and run the Corporate Agent on your local machine.

### Prerequisites

*   **Python 3.9+:** Ensure you have a compatible version of Python installed.
*   **Git:** For cloning the repository.
*   **Ollama:** This is required to run the Llama 3 model locally. [Download and install it from ollama.com](https://ollama.com/).

### Step 1: Clone the Repository

Open your terminal or command prompt and clone this repository:

```bash
git clone https://github.com/YourUsername/your-repo-name.git
cd your-repo-name
```
*(Replace with your actual repository URL)*

### Step 2: Create a `requirements.txt` File

This project's dependencies are managed through a `requirements.txt` file. Create a file named `requirements.txt` in the project root and add the following content:

```text
streamlit
langchain
langchain_community
sentence-transformers
faiss-cpu
python-docx
pypdf
torch
transformers
```

### Step 3: Install All Dependencies

Install all the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Step 4: Download the Local AI Model (Llama 3)

With Ollama installed, run the following command in your terminal. This will download the Llama 3 model (a one-time download of several gigabytes).

```bash
ollama pull llama3
```
Ollama will run as a background service, making the model available to the application.

### Step 5: Prepare the Knowledge Base

The agent's intelligence comes from the ADGM documents it studies.

1.  **Add Reference Documents:** Place all the official ADGM `.pdf` and `.docx` reference documents inside the `adgm_documents` folder.
2.  **Run the Ingestion Script:** This script reads the documents, creates embeddings, and builds the local vector store. Run it from the project's root directory:

    ```bash
    python ingest_data.py
    ```
    This will create an `adgm_faiss_index_local` folder. This step is only needed once, or whenever the reference documents are updated.

### Step 6: Run the Application

You are now ready to launch the agent!

```bash
streamlit run corporate_agent_app.py
```

A new tab should open in your web browser with the application running. You can now upload your `.docx` files for analysis.
