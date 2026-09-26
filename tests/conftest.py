"""Real ranking code with deterministic, offline dependencies."""

import sys
import numpy as np
import pandas as pd
import pytest

sys.modules["sentence_transformers"] = None
sys.modules["google.genai"] = None
import streamer_recommender as engine  # noqa: E402


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


@pytest.fixture
def csv_path(tmp_path):
    rows = []
    for pfid, gender, talents, style in [
        (900, "女", "歌唱", "互動熱絡"),
        (12, "男", "歌唱、舞蹈", "安靜"),
        (407, "女", "舞蹈", "互動熱絡"),
    ]:
        rows.append(
            dict(
                pfid=pfid,
                gender=gender,
                personality="溫柔",
                appearance="自然",
                talents=talents,
                featured_topics="音樂",
                live_streaming_style=style,
                overall_vibe="輕鬆",
                reasons='{"才藝": "歌唱證據"}',
                self_description="介紹",
            )
        )
    path = tmp_path / "anchors.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def recommender(csv_path, tmp_path):
    return engine.HybridStreamerRecommender(csv_path, cache_dir=tmp_path / "cache")


class FixedEncoder:
    def __init__(self, model_name):
        self.calls = []

    def encode(self, documents, **kwargs):
        self.calls.append(list(documents))
        if len(documents) == 1:
            return np.array([[1.0, 0.0]])
        return np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])


@pytest.fixture
def dense_recommender(monkeypatch, csv_path, tmp_path):
    monkeypatch.setattr(engine, "SentenceTransformer", FixedEncoder)
    return engine.HybridStreamerRecommender(csv_path, cache_dir=tmp_path / "cache")
