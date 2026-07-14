# Live Streaming Anchor Recommender

A hybrid recommendation system for live streaming anchors. It combines structured tag matching with Chinese semantic retrieval using sentence embeddings, and optionally generates natural-language explanations via Google Gemini.

## Features

- Hybrid recommendation using:
  - structured tag inference from user query
  - semantic retrieval over Chinese anchor metadata
- Optional Gemini-based explanation generation with `GEMINI_API_KEY`
- Streamlit user interface for interactive filtering and ranking
- Local fallback explanation when Gemini is unavailable
- Diversity reranking to reduce overly similar results

## Repository Files

- `app.py` — Streamlit UI entry point
- `streamer_recommender.py` — hybrid recommendation engine and CLI fallback
- `anchors_100.csv` — anchor dataset used for recommendations
- `requirements.txt` — Python dependencies

## Requirements

- Python 3.11+ (recommended)
- `pip` or another Python package manager

## Installation

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The app can optionally use Google Gemini for explanation generation.

Create a `.env` file in the project root or export environment variables directly:

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
DATA_PATH=anchors_100.csv
CACHE_DIR=.cache
```

If `GEMINI_API_KEY` is not set or `google-genai` is unavailable, the system will use a local text-based fallback explanation.

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your browser.

## Run the CLI Fallback

```bash
python streamer_recommender.py
```

This will prompt for a query and display ranked anchor recommendations in the terminal.

## Data and Models

- `anchors_100.csv` is the default anchor metadata source.
- `EMBEDDING_MODEL_NAME` defaults to `shibing624/text2vec-base-chinese`.
- Sentence embedding caching is stored in `.cache` by default.

## Notes

- The Streamlit UI supports gender filtering, required tags, top-K selection, and diversity control.
- The recommendation engine performs a hybrid score when explicit tags are detected in the query, otherwise it relies on semantic similarity.
- Required tags are enforced with AND logic.

## License

This repository does not include a license file. Add one if you plan to share or publish the project.
