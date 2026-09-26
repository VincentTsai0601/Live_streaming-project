from pathlib import Path
import streamer_recommender as engine


def test_existing_ci_retrieval_smoke_offline(tmp_path):
    """Preserve the existing CI assertions using the real CSV and TF-IDF."""
    rec = engine.HybridStreamerRecommender(
        Path(__file__).resolve().parents[1] / "anchors_100.csv", cache_dir=tmp_path
    )
    results = rec.retrieve(
        "想看會唱歌、互動熱絡的主播", top_k=3, required_tags=["歌唱", "互動熱絡"]
    )
    assert len(results) == 3
    for result in results:
        assert "歌唱" in result["metadata"]["才藝"]
        assert "互動熱絡" in result["metadata"]["風格"]
