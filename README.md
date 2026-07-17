# Live Streaming Anchor Recommender

A hybrid recommendation system for live streaming anchors. It combines structured tag matching with Chinese semantic retrieval using sentence embeddings, and optionally generates natural-language explanations via Google Gemini.

## Dataset 
The dataset itself exhibits a clear gender imbalance. Therefore, the gender distribution of the recommendation results does not necessarily reflect model bias alone, but is also influenced by the composition of the original candidate pool.

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

1. Create and activate a Conda environment:

```bash
conda create -n streamer-recommender python=3.11 -y
conda activate streamer-recommender
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Installing PyTorch 

PyTorch wheels are platform- and CUDA-version-specific and therefore are not included directly in `requirements.txt`. Install `torch` / `torchvision` / `torchaudio` separately using one of the options below that matches your system.

- pip (CUDA 11.8):

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


## Configuration

The app can optionally use Google Gemini for explanation generation.

Create a `.env` file in the project root or export environment variables directly:

```bash
touch .env
code .env
```

add below arguments 

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
DATA_PATH=anchors_100.csv
CACHE_DIR=.cache
EMBEDDING_MODEL_NAME=shibing624/text2vec-base-chinese
```

If `GEMINI_API_KEY` is not set or `google-genai` is unavailable, the system will use a local text-based fallback explanation.

### Obtaining a Gemini API key (Generative AI Studio)
If you obtained your API key from AI Studio, use the API keys page:

`https://aistudio.google.com/api-keys`

Copy the key shown there into `GEMINI_API_KEY` in your `.env` file (or export it as an environment variable).


If you want, create a `.env` in the project root with the variable names (the actual secret should never be committed).

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your browser.

If you prefer to run the Docker image, use:

```bash
docker run --rm \
  --name streamer-recommender \
  --env-file .env \
  -p 8514:8501 \
  ghcr.io/vincenttsai0601/streamer-recommender:latest
```

Then open:

```text
http://localhost:8514
```

If you run the Docker container with port mapping `-p 8514:8501`, open:

```text
http://localhost:8514
```

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
