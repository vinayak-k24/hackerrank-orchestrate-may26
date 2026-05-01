# Support Triage Agent

This directory contains the code for the Support Triage Agent, built for the HackerRank Orchestrate hackathon.

## Architecture & Approach

The agent has been cleanly refactored into modular components to ensure high maintainability, clear separation of concerns, and robust reasoning:

- **`config.py`**: Handles environment variables and configuration loading.
- **`models.py`**: Defines standard data structures (`Ticket`, `Article`, `TriageResult`).
- **`text_utils.py`**: Manages tokenization and text normalization.
- **`retriever.py`**: Implements a robust `BM25Retriever` algorithm for finding the most relevant articles from the local corpus without external API dependencies.
- **`agent.py`**: Contains the core `SupportAgent` logic. It decides when to escalate high-risk or unsupported cases and routes valid cases to the appropriate product area. It can optionally use OpenAI for answer synthesis if `USE_OPENAI=true` and an API key is provided, but it includes an efficient heuristic sentence extractor as a fallback.
- **`main.py`**: The main entry point that processes the CSV file and outputs predictions.

## Setup and Running

1. **Install Dependencies**
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   pip install -r ../requirements.txt
   ```

2. **Configuration**
   Copy `.env.example` (if available) to `.env` or `.env.local` to override configurations.
   If you have an OpenAI key, you can add it to `.env`:
   ```env
   USE_OPENAI=true
   OPENAI_API_KEY=your-api-key-here
   ```
   If no key is provided, the agent will automatically use a fallback algorithm to extract relevant sentences from the retrieved corpus documents.

3. **Running the Agent**
   From the repository root (or the `code/` directory), run:
   ```bash
   python code/main.py
   ```
   The predictions will be saved to the configured output CSV path (usually `support_tickets/output.csv` or `support_issues/output.csv`).