import numpy as np
import pytest


def test_dense_mapping_and_semantic_only(dense_recommender):
    results = dense_recommender.retrieve("未知需求", diversity=0)
    assert [r["pfid"] for r in results] == [12, 407, 900]
    assert [r["score"] for r in results] == [1.0, 0.6, 0.0]
    assert [r["rank"] for r in results] == [1, 2, 3]
    assert all(r["score_breakdown"]["structured"] == 0 for r in results)


def test_hybrid_weights(dense_recommender):
    results = dense_recommender.retrieve("歌唱", diversity=0)
    assert {r["pfid"]: r["score"] for r in results} == {12: 1.0, 407: 0.39, 900: 0.35}


def test_structured_weights(recommender):
    scores, matches = recommender._structured_scores(
        {"talents": {"歌唱"}, "live_streaming_style": {"互動熱絡"}}
    )
    np.testing.assert_allclose(scores, [1, 1.3 / 2.5, 1.2 / 2.5])
    assert set(matches[0]) == {"歌唱", "互動熱絡"}


def test_required_tags_are_and(recommender):
    assert recommender._required_tag_mask({"歌唱", "互動熱絡"}).tolist() == [
        True,
        False,
        False,
    ]
    assert recommender._required_tag_mask(set()).all()
    assert not recommender._required_tag_mask({"不存在"}).any()


@pytest.mark.parametrize("fixture_name", ["recommender", "dense_recommender"])
def test_filters_and_mapping(request, fixture_name):
    rec = request.getfixturevalue(fixture_name)
    assert [
        r["pfid"] for r in rec.retrieve("歌唱", required_tags=["歌唱", "互動熱絡"])
    ] == [900]
    assert [r["pfid"] for r in rec.retrieve("女生唱歌", required_gender="男")] == [12]
    assert {r["pfid"] for r in rec.retrieve("女生唱歌")} == {900, 407}


@pytest.mark.parametrize("top_k,expected", [(0, 1), (-1, 1), (1, 1), (100, 3)])
def test_top_k(recommender, top_k, expected):
    assert len(recommender.retrieve("歌唱", top_k=top_k)) == expected


def test_errors(recommender):
    with pytest.raises(ValueError, match="空白"):
        recommender.retrieve("  ")
    with pytest.raises(ValueError, match="必要條件"):
        recommender.retrieve("歌唱", required_tags=["不存在"])


def test_backend_routes(recommender, dense_recommender):
    assert recommender.retrieval_backend == "tf-idf"
    assert dense_recommender.retrieval_backend == "sentence-transformer"
