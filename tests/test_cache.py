import numpy as np
import pytest


def test_cache_hit_avoids_encoding(dense_recommender):
    rec = dense_recommender
    rec.encoder.calls.clear()
    np.testing.assert_array_equal(rec._load_or_create_embeddings(), rec.item_embeddings)
    assert rec.encoder.calls == []


def test_cache_key_tracks_data_and_model(dense_recommender):
    rec = dense_recommender
    original = rec._cache_path()
    rec.model_name += "-changed"
    assert rec._cache_path() != original
    model_path = rec._cache_path()
    with rec.csv_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert rec._cache_path() != model_path


def test_wrong_row_count_rebuilds(dense_recommender):
    rec = dense_recommender
    np.save(rec._cache_path(), np.zeros((1, 2)))
    rec.encoder.calls.clear()
    assert rec._load_or_create_embeddings().shape == (3, 2)
    assert len(rec.encoder.calls) == 1


def test_corrupt_cache_currently_raises(dense_recommender):
    rec = dense_recommender
    rec._cache_path().write_bytes(b"invalid numpy file")
    with pytest.raises(ValueError):
        rec._load_or_create_embeddings()
